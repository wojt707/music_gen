import os
import argparse
import logging
from pathlib import Path
import numpy as np
import evaluate
from datasets import load_from_disk
import datasets

datasets.disable_progress_bar()

from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import torch
import torch.nn as nn
from miditok import Octuple
from notify_run import Notify

# Custom Trainer to support class weights
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

# Set up logging
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


def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    return {"accuracy": acc, "f1": f1}


class OctuplePaddingCollator:
    """
    Picklable data collator for Octuple-tokenized datasets.

    Octuple represents each musical event as a tuple of sub-token IDs (pitch,
    duration, velocity, bar, etc.). This collator flattens those 2D lists into
    1D, truncates to MAX_SEQ_LEN, and pads to the longest sequence in the batch.

    Must be a top-level class (not a nested function) so that DataLoader workers
    can pickle it on Windows (spawn multiprocessing).
    """
    MAX_SEQ_LEN = 1024  # hard cap: full self-attention is O(seq_len^2)

    def __init__(self, pad_token_id: int, cls_token_id: int):
        self.pad_id = pad_token_id
        self.cls_token_id = cls_token_id

    def __call__(self, features):
        input_ids_list = [f["input_ids"] for f in features]
        labels_list    = [f["label"]     for f in features]

        # Flatten 2D Octuple events -> 1D token list, then truncate (leaving room for CLS)
        flattened = []
        for ids in input_ids_list:
            flat = [t for note in ids for t in note] if isinstance(ids[0], (list, tuple)) else list(ids)
            flattened.append(flat[:self.MAX_SEQ_LEN - 1])

        max_len = min(max(len(f) for f in flattened), self.MAX_SEQ_LEN - 1)

        padded_ids, attn_mask = [], []
        for flat in flattened:
            pad_len = max_len - len(flat)
            # Prepend CLS token at index 0
            padded_ids.append([self.cls_token_id] + flat + [self.pad_id] * pad_len)
            attn_mask.append([1] + [1] * len(flat) + [0] * pad_len)

        return {
            "input_ids":      torch.tensor(padded_ids, dtype=torch.long),
            "labels":         torch.tensor(labels_list, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask,  dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description="Train a RoBERTa-based Genre Classifier on Octuple MIDI chunks.")
    parser.add_argument("--multi-instrument", action="store_true", help="Train on all-instrument split.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: slice dataset to 256 samples and run only 5 steps. For local smoke testing.")
    parser.add_argument("--dataset-name", type=str, default="gt_1000_with_ties",
                        help="Name of the dataset directory in artist_genres (e.g. gt_1000_no_ties)")
    args = parser.parse_args()

    # Determine base directory (fallback to relative if not running via Slurm)
    project_root = Path(__file__).resolve().parent.parent
    sync_dir = Path(os.getenv("MUSICGEN_SYNC_DIR", project_root))
    
    # Determine paths based on split
    split_type = "multi-instrument" if args.multi_instrument else "piano"
    
    candidate_dirs = [
        sync_dir / "datasets_metadata" / args.dataset_name / "dataset" / split_type,
        sync_dir / "dataset" / args.dataset_name / split_type,
        sync_dir / "artist_genres" / args.dataset_name / "dataset" / split_type,
        project_root / "datasets_metadata" / args.dataset_name / "dataset" / split_type,
    ]
    if os.getenv("SCRATCH"):
        candidate_dirs.insert(0, Path(os.environ["SCRATCH"]) / "dataset" / args.dataset_name / split_type)

    dataset_dir = candidate_dirs[0]
    for cand in candidate_dirs:
        if (cand / "octuple").exists():
            dataset_dir = cand
            break

    octuple_dir = dataset_dir / "octuple"
    tokenizer_path = octuple_dir / "tokenizer" / "tokenizer.json"
    
    send_notification(f'Training classifier for dataset: {args.dataset_name}, split: {split_type}')
    
    output_dir = Path(f"classifier_output/{split_type}")  # split-specific so piano/multi-instrument don't overwrite each other
    
    if not octuple_dir.exists():
        log.error(f"Dataset directory not found: {octuple_dir}")
        return
        
    if not tokenizer_path.exists():
        log.error(f"Tokenizer not found: {tokenizer_path}")
        return

    # Load Tokenizer
    log.info(f"Loading Octuple tokenizer from {tokenizer_path}...")
    tokenizer = Octuple(params=str(tokenizer_path))
    
    # Load Datasets
    log.info(f"Loading HuggingFace datasets from {octuple_dir}...")
    train_ds = load_from_disk(str(octuple_dir / "train"))
    val_ds = load_from_disk(str(octuple_dir / "val"))
    test_ds = load_from_disk(str(octuple_dir / "test"))
    
    log.info(f"Train size: {len(train_ds):,} | Val size: {len(val_ds):,} | Test size: {len(test_ds):,}")

    # Extract unique genres BEFORE debug slicing so num_labels is always the full set
    genres = sorted(list(set(train_ds["genre"])))
    genre2id = {g: i for i, g in enumerate(genres)}
    id2genre = {i: g for g, i in genre2id.items()}
    num_labels = len(genres)
    log.info(f"Found {num_labels} genres: {genres}")

    if args.debug:
        log.warning("DEBUG MODE: slicing dataset to 256/64/64 samples.")
        train_ds = train_ds.select(range(min(256, len(train_ds))))
        val_ds   = val_ds.select(range(min(64,  len(val_ds))))
        test_ds  = test_ds.select(range(min(64,  len(test_ds))))


    # Load class weights
    import json
    weights_candidates = [
        project_root / "datasets_metadata" / args.dataset_name / "split" / split_type / "genre_weights.json",
        sync_dir / "datasets_metadata" / args.dataset_name / "split" / split_type / "genre_weights.json",
        sync_dir / "artist_genres" / args.dataset_name / "split" / split_type / "genre_weights.json",
    ]
    weights_path = None
    for cand in weights_candidates:
        if cand.exists():
            weights_path = cand
            break

    if weights_path and weights_path.exists():
        with open(weights_path, "r", encoding="utf-8") as f:
            genre_weights = json.load(f)
        class_weights = [1.0] * num_labels
        for genre, weight in genre_weights.items():
            if genre in genre2id:
                class_weights[genre2id[genre]] = weight
        log.info(f"Loaded class weights mapping from {weights_path}: {class_weights}")
    else:
        log.warning("No genre_weights.json found, using uniform weights.")
        class_weights = None

    # Ensure labels are named 'label' for HuggingFace Trainer
    def format_dataset(example):
        return {"input_ids": example["input_ids"], "label": example["genre_id"]}
        
    train_ds = train_ds.map(format_dataset, remove_columns=["genre_id", "genre"])
    val_ds = val_ds.map(format_dataset, remove_columns=["genre_id", "genre"])
    test_ds = test_ds.map(format_dataset, remove_columns=["genre_id", "genre"])

    # Configure RoBERTa from scratch
    # We use a lightweight architecture for fast and efficient training
    pad_token_id = tokenizer["PAD_None"]
    
    # Create a dedicated CLS token at the end of the vocabulary
    cls_token_id = len(tokenizer)
    vocab_size = len(tokenizer) + 1
    
    config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=1026,  # 1024 tokens + 2 special tokens
        num_hidden_layers=6,
        num_attention_heads=8,
        hidden_size=512,
        intermediate_size=2048,
        pad_token_id=pad_token_id,
        num_labels=num_labels,
        id2label=id2genre,
        label2id=genre2id,
    )
    
    log.info("Initializing blank RoBERTa model from scratch...")
    model = RobertaForSequenceClassification(config)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {num_params:,}")

    collator = OctuplePaddingCollator(pad_token_id=pad_token_id, cls_token_id=cls_token_id)

    # Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps" if args.debug else "epoch",
        eval_steps=3 if args.debug else None,         # eval mid-run in debug
        save_strategy="steps" if args.debug else "epoch",
        save_steps=3 if args.debug else None,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=5 if args.debug else -1,            # hard cap in debug: 5 steps max
        weight_decay=0.01,
        warmup_ratio=0.1,                             # VERY IMPORTANT for training transformers from scratch
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # only on GPU; disabled automatically for local CPU runs
        logging_steps=1 if args.debug else 100,
        report_to="none", # Disable wandb/tensorboard for simplicity
        dataloader_num_workers=0 if args.debug else 4,  # no subprocesses on debug/Windows
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        class_weights=class_weights
    )

    log.info("Starting training...")
    trainer.train()
    
    log.info("Evaluating on TEST set...")
    test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
    log.info(f"Test Results: {test_results}")

    log.info(f"Saving best model to {output_dir / 'best_model'}...")
    trainer.save_model(str(output_dir / "best_model"))
    
    send_notification(f'Training finished for dataset: {args.dataset_name}, split: {split_type}')
    log.info("Done!")

if __name__ == "__main__":
    main()
