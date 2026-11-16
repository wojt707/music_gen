import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from miditok import REMI
from notify_run import Notify
from transformers import (
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def send_notification(msg: str):
    endpoint = os.getenv("NOTIFY_ENDPOINT")
    if not endpoint or Notify is None:
        return
    try:
        notify = Notify(endpoint=endpoint)
        notify.send(msg)
    except Exception as e:
        log.warning(f"Could not send notification: {e}")

class LlamaGenrePrefixCollator:
    """
    Prepends the genre token ID to the beginning of the sequence.
    Shifts the MIDI tokens into the extended vocabulary space (after the original LLaMA text tokens).
    Sets padding targets to -100 so loss is ignored on padding.
    """
    def __init__(self, original_vocab_size: int, base_vocab_size: int, pad_token_id: int):
        self.original_vocab_size = original_vocab_size
        self.base_vocab_size = base_vocab_size
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids_list = [f["input_ids"] for f in features]
        labels_list    = [f["label"] for f in features]

        padded_input_ids = []
        padded_labels = []

        # Find max length in batch after prepending genre token
        max_len_in_batch = max(len(ids) + 1 for ids in input_ids_list)

        for ids, genre_id in zip(input_ids_list, labels_list):
            genre_token = self.original_vocab_size + self.base_vocab_size + genre_id
            
            # Shift MIDI tokens so they don't collide with pre-trained text tokens
            shifted_ids = [self.original_vocab_size + t for t in ids]
            
            # Prepend genre token
            seq = [genre_token] + shifted_ids
            
            pad_len = max_len_in_batch - len(seq)
            
            # Pad input_ids with pad_token_id
            padded_seq = seq + [self.pad_token_id] * pad_len
            padded_input_ids.append(padded_seq)
            
            # Pad labels with -100 so loss isn't computed on padding
            padded_label = seq + [-100] * pad_len
            padded_labels.append(padded_label)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": (torch.tensor(padded_input_ids, dtype=torch.long) != self.pad_token_id).long()
        }


def main():
    parser = argparse.ArgumentParser(description="Full fine-tune of LLaMA 3.2 1B on REMI MIDI chunks.")
    parser.add_argument("--multi-instrument", action="store_true", help="Train on all-instrument split.")
    parser.add_argument("--piano", action="store_true", help="Explicit piano flag (default).")
    parser.add_argument("extra_positional", nargs="*", help="Catch stray positional arguments like 'piano' or 'multi-instrument'.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU (keep 2 for 1B model on 40GB GPU).")
    parser.add_argument("--grad-accum", type=int, default=32, help="Gradient accumulation steps (2 * 32 = 64 effective batch size).")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (lower for fine-tuning).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (defaults to 10 for piano, 2 for multi-instrument).")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint path or True for latest.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: 256 samples, 5 steps.")
    parser.add_argument("--dataset-name", type=str, default="gt_1000_with_ties",
                        help="Name of the dataset directory in artist_genres (e.g. gt_1000_no_ties)")
    args = parser.parse_args()

    # If user passed 'multi-instrument' as a positional argument
    if "multi-instrument" in args.extra_positional:
        args.multi_instrument = True

    project_root = Path(__file__).resolve().parent.parent
    sync_dir = Path(os.getenv("MUSICGEN_SYNC_DIR", project_root))
    split_type = "multi-instrument" if args.multi_instrument else "piano"

    candidate_dirs = [
        sync_dir / "datasets_metadata" / args.dataset_name / "dataset" / split_type / "remi_bpe",
        sync_dir / "dataset" / args.dataset_name / split_type / "remi_bpe",
        sync_dir / "artist_genres" / args.dataset_name / "dataset" / split_type / "remi_bpe",
        project_root / "datasets_metadata" / args.dataset_name / "dataset" / split_type / "remi_bpe",
    ]
    if os.getenv("SCRATCH"):
        candidate_dirs.insert(0, Path(os.environ["SCRATCH"]) / "dataset" / args.dataset_name / split_type / "remi_bpe")

    dataset_dir = candidate_dirs[0]
    for cand in candidate_dirs:
        if cand.exists():
            dataset_dir = cand
            break

    tokenizer_path = dataset_dir / "tokenizer" / "tokenizer.json"
    
    output_dir = Path(f"llama_output/{split_type}")
    
    send_notification(f'Training LLaMA 3.2 1B for dataset: {args.dataset_name}, split: {split_type}')
    
    if not dataset_dir.exists() or not tokenizer_path.exists():
        log.error(f"Dataset or Tokenizer not found in {dataset_dir}")
        return

    # 1. Load Tokenizer
    log.info(f"Loading REMI+BPE tokenizer from {tokenizer_path}...")
    remi_tok = REMI(params=str(tokenizer_path))
    base_vocab_size = len(remi_tok)
    log.info(f"MIDI Base Vocab size: {base_vocab_size:,}")

    # 2. Load Datasets
    log.info(f"Loading HuggingFace datasets from {dataset_dir}...")
    train_ds = load_from_disk(str(dataset_dir / "train"))
    val_ds   = load_from_disk(str(dataset_dir / "val"))
    
    # Extract unique genres BEFORE debug slicing
    genres = sorted(list(set(train_ds["genre"])))
    num_genres = len(genres)
    log.info(f"Found {num_genres} genres: {genres}")
    
    if args.debug:
        log.warning("DEBUG MODE: slicing dataset to 256/64 samples.")
        train_ds = train_ds.select(range(min(256, len(train_ds))))
        val_ds   = val_ds.select(range(min(64,  len(val_ds))))
        
    log.info(f"Train size: {len(train_ds):,} | Val size: {len(val_ds):,}")

    # Ensure labels are named 'label' for DataCollator extraction
    def format_dataset(example):
        return {"input_ids": example["input_ids"], "label": example["genre_id"]}
        
    train_ds = train_ds.map(format_dataset, remove_columns=["genre_id", "genre"])
    val_ds = val_ds.map(format_dataset, remove_columns=["genre_id", "genre"])

    # 3. Model Configuration
    log.info("Downloading/Loading pre-trained meta-llama/Llama-3.2-1B...")
    # Note: Requires HuggingFace token configured via `huggingface-cli login` or HF_TOKEN env var!
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=os.environ.get("HF_TOKEN"))
    
    original_vocab_size = model.config.vocab_size
    total_vocab_size = original_vocab_size + base_vocab_size + num_genres
    
    log.info(f"Original Text Vocab Size: {original_vocab_size:,}")
    log.info(f"Extended Vocab Size (Text + MIDI + Genres): {total_vocab_size:,}")
    
    # Expand the embedding layer to make room for MIDI and Genre tokens
    model.resize_token_embeddings(total_vocab_size)
    
    # Determine a pad token (LLaMA 3 lacks a dedicated pad token by default)
    llama_pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    model.config.pad_token_id = llama_pad_token_id
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters to train: {num_params:,}")

    collator = LlamaGenrePrefixCollator(
        original_vocab_size=original_vocab_size,
        base_vocab_size=base_vocab_size,
        pad_token_id=llama_pad_token_id
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps" if args.debug else "epoch",
        eval_steps=3 if args.debug else None,
        save_strategy="steps" if args.debug else "epoch",
        save_steps=3 if args.debug else None,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,  # Effective batch size = batch_size * grad_accum
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs if args.epochs is not None else (2 if args.multi_instrument else 10),
        max_steps=5 if args.debug else -1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=1 if args.debug else 100,
        report_to="none",
        dataloader_num_workers=0 if args.debug else 4,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = True if args.resume.lower() in ("true", "1") else args.resume

    log.info(f"Starting LLaMA fine-tuning (epochs: {training_args.num_train_epochs})...")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    
    log.info(f"Saving best model to {output_dir / 'best_model'}...")
    trainer.save_model(str(output_dir / "best_model"))
    
    send_notification(f'LLaMA 3.2 1B Training finished for dataset: {args.dataset_name}, split: {split_type}')
    log.info("Done!")

if __name__ == "__main__":
    main()
