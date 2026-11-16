"""
train_lstm.py
=============
Genre-conditioned LSTM for symbolic music generation.

Conditioning: A learnable genre embedding (genre_embed_dim) is concatenated
to the token embedding at every LSTM step, providing a persistent genre signal.

Usage:
    python train_lstm.py --split piano
    python train_lstm.py --split multi-instrument --epochs 30
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from miditok import REMI
from notify_run import Notify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
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


# ===========================================================================
# Dataset
# ===========================================================================

class MidiChunkDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace arrow dataset of REMI+BPE token chunks."""

    def __init__(self, hf_dataset, pad_token_id: int):
        self.ds = hf_dataset
        self.pad = pad_token_id

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        ids = row["input_ids"]
        genre_id = row["genre_id"]

        # If the track is extremely short, skip/pad it to at least 2 tokens
        if len(ids) < 2:
            ids = ids + [self.pad] * (2 - len(ids))

        return torch.tensor(ids, dtype=torch.long), torch.tensor(genre_id, dtype=torch.long)


class PadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        ids = [item[0] for item in batch]
        genres = [item[1] for item in batch]
        ids_padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id)
        return ids_padded, torch.stack(genres)

# ===========================================================================
# Model
# ===========================================================================

class GenreLSTM(nn.Module):
    """
    3-layer LSTM for autoregressive token generation.

    Genre conditioning: genre embedding concatenated to token embedding at
    every step so the conditioning signal is persistent throughout the sequence.

    LSTM input dim = token_embed_dim + genre_embed_dim
    """

    def __init__(
        self,
        vocab_size: int,
        num_genres: int,
        token_embed_dim: int = 256,
        genre_embed_dim: int = 32,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.token_emb = nn.Embedding(vocab_size, token_embed_dim, padding_idx=pad_token_id)
        self.genre_emb = nn.Embedding(num_genres, genre_embed_dim)
        self.lstm = nn.LSTM(
            input_size=token_embed_dim + genre_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, genre_ids, hidden=None):
        """
        input_ids  : (B, T)
        genre_ids  : (B,)
        Returns logits (B, T, vocab_size) and updated hidden state.
        """
        B, T = input_ids.shape
        tok = self.token_emb(input_ids)                     # (B, T, E)
        gen = self.genre_emb(genre_ids).unsqueeze(1).expand(-1, T, -1)  # (B, T, G)
        x, hidden = self.lstm(torch.cat([tok, gen], dim=-1), hidden)
        return self.output_head(x), hidden                  # (B, T, V)

    @torch.no_grad()
    def generate(
        self,
        genre_id: int,
        prompt_ids: list[int] | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.9,
        device: str = "cpu",
    ) -> list[int]:
        """Autoregressive top-p nucleus sampling."""
        self.eval()
        genre_t = torch.tensor([genre_id], dtype=torch.long, device=device)

        hidden = None
        if prompt_ids:
            prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            _, hidden = self.forward(prompt_t, genre_t, hidden)
            current_id = prompt_ids[-1]
        else:
            current_id = self.pad_token_id

        generated = []
        for _ in range(max_new_tokens):
            x = torch.tensor([[current_id]], dtype=torch.long, device=device)
            logits, hidden = self.forward(x, genre_t, hidden)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumulative - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum()

            current_id = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)].item()
            generated.append(current_id)

        return generated


# ===========================================================================
# Training
# ===========================================================================

def compute_loss(logits, targets, pad_id):
    """Cross-entropy loss with PAD token masking. Predicts tokens [1..T] from [0..T-1]."""
    B, T, V = logits.shape
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    shifted_logits  = logits[:, :-1, :].contiguous().view(-1, V)
    shifted_targets = targets[:, 1:].contiguous().view(-1)
    pad_mask        = (shifted_targets == pad_id)

    loss = loss_fn(shifted_logits, shifted_targets)
    loss = loss.masked_fill(pad_mask, 0.0)
    return loss.sum() / (~pad_mask).sum().clamp(min=1)


def run_epoch(model, loader, optimizer, device, is_train, grad_clip, pad_id, max_batches=None):
    model.train(is_train)
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for ids, genres in loader:
            if max_batches and n >= max_batches:
                break
            ids, genres = ids.to(device), genres.to(device)
            logits, _ = model(ids, genres)
            loss = compute_loss(logits, ids, pad_id)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train genre-conditioned LSTM for MIDI generation.")
    parser.add_argument("--multi-instrument", action="store_true",
                        help="Shorthand for --split multi-instrument (consistent with classifier script).")
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--resume",         action="store_true", help="Resume from best_model.pt if it exists.")
    parser.add_argument("--patience",       type=int,   default=5,   help="Early stopping patience (epochs).")
    parser.add_argument("--grad-clip",      type=float, default=1.0)
    parser.add_argument("--workers",        type=int,   default=4)
    parser.add_argument("--token-embed-dim",type=int,   default=256)
    parser.add_argument("--genre-embed-dim",type=int,   default=32,  help="Learned genre vector size (not # of genres).")
    parser.add_argument("--hidden-size",    type=int,   default=512)
    parser.add_argument("--num-layers",     type=int,   default=3)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--temperature",    type=float, default=1.0, help="Demo generation temperature.")
    parser.add_argument("--top-p",          type=float, default=0.9, help="Demo generation nucleus sampling p.")
    parser.add_argument("--debug",          action="store_true",
                        help="Debug mode: 256 train samples, 1 epoch, 5 batches. For local smoke testing.")
    parser.add_argument("--dataset-name",   type=str, default="gt_1000_with_ties",
                        help="Name of the dataset directory in artist_genres (e.g. gt_1000_no_ties)")
    args = parser.parse_args()
    split = "multi-instrument" if args.multi_instrument else "piano"
    # --debug forces a quick single-epoch run
    if args.debug:
        args.epochs = 5
        args.workers = 0  # no multiprocessing needed for tiny dataset

    # ── Paths ─────────────────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent
    sync_dir    = Path(os.getenv("MUSICGEN_SYNC_DIR", project_root))
    
    candidate_dirs = [
        sync_dir / "datasets_metadata" / args.dataset_name / "dataset" / split / "remi_bpe",
        sync_dir / "dataset" / args.dataset_name / split / "remi_bpe",
        sync_dir / "artist_genres" / args.dataset_name / "dataset" / split / "remi_bpe",
        project_root / "datasets_metadata" / args.dataset_name / "dataset" / split / "remi_bpe",
    ]
    if os.getenv("SCRATCH"):
        candidate_dirs.insert(0, Path(os.environ["SCRATCH"]) / "dataset" / args.dataset_name / split / "remi_bpe")

    dataset_dir = candidate_dirs[0]
    for cand in candidate_dirs:
        if cand.exists():
            dataset_dir = cand
            break

    tok_path    = dataset_dir / "tokenizer" / "tokenizer.json"
    output_dir  = Path(f"lstm_output/{split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    send_notification(f'Training LSTM for dataset: {args.dataset_name}, split: {split}')

    for p, name in [(dataset_dir, "dataset"), (tok_path, "tokenizer")]:
        if not p.exists():
            log.error(f"{name} not found: {p}")
            return

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Load via MidiTok directly — its native JSON format is not HuggingFace tokenizers
    # format, so manual JSON parsing is unreliable. len(tok) and tok["PAD_None"] are
    # the correct MidiTok API calls (same pattern as the classifier uses for Octuple).
    log.info(f"Loading REMI+BPE tokenizer from {tok_path}...")
    remi_tok     = REMI(params=str(tok_path))
    vocab_size   = len(remi_tok)
    pad_token_id = remi_tok["PAD_None"]
    log.info(f"Vocab size: {vocab_size:,}  PAD id: {pad_token_id}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    log.info("Loading datasets...")
    train_raw = load_from_disk(str(dataset_dir / "train"))
    val_raw   = load_from_disk(str(dataset_dir / "val"))

    # Infer genre mapping from dataset features (fast — no full scan needed)
    genre2id = {}
    for row in train_raw.select(range(min(5000, len(train_raw)))):
        genre2id.setdefault(row["genre"], row["genre_id"])
        if len(genre2id) == 12:  # early exit once all 12 genres found
            break
    num_genres = max(genre2id.values()) + 1
    id2genre   = {v: k for k, v in genre2id.items()}
    log.info(f"Genres ({num_genres}): {sorted(genre2id)}")

    train_ds = MidiChunkDataset(train_raw, pad_token_id)
    val_ds   = MidiChunkDataset(val_raw,   pad_token_id)
    collator = PadCollator(pad_token_id)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, collate_fn=collator)
    log.info(f"Train chunks: {len(train_ds):,}  Val chunks: {len(val_ds):,}")

    if args.debug:
        log.warning("DEBUG MODE: slicing to 256/64 samples, 1 epoch, 5 batches max.")
        train_ds = MidiChunkDataset(train_raw.select(range(min(256, len(train_raw)))), pad_token_id)
        val_ds   = MidiChunkDataset(val_raw.select(range(min(64,  len(val_raw)))),   pad_token_id)
        # Rebuild loaders with the sliced datasets and no multiprocessing
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, drop_last=False, collate_fn=collator)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GenreLSTM(
        vocab_size=vocab_size,
        num_genres=num_genres,
        token_embed_dim=args.token_embed_dim,
        genre_embed_dim=args.genre_embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_token_id=pad_token_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Resume Logic ──────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")
    best_path = output_dir / "best_model.pt"

    if args.resume and best_path.exists():
        log.info(f"Resuming from checkpoint: {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            log.info("Loaded optimizer state.")
        else:
            log.warning("No optimizer state in checkpoint. Adam momentum will be reset (expect a temporary loss spike).")
            
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        log.info(f"Resuming at epoch {start_epoch} with best_val_loss={best_val_loss:.4f}")

    # Save run config
    config = {**vars(args), "vocab_size": vocab_size, "num_genres": num_genres, "pad_token_id": pad_token_id}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Training LSTM | split={split} | epochs={args.epochs}")
    log.info("=" * 60)

    patience_counter = 0
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, True,  args.grad_clip, pad_token_id,
                               max_batches=5 if args.debug else None)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, False, args.grad_clip, pad_token_id,
                               max_batches=5 if args.debug else None)
        elapsed    = time.time() - t0

        log.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.4f} ppl={math.exp(min(train_loss,20)):.1f}  "
            f"val={val_loss:.4f} ppl={math.exp(min(val_loss,20)):.1f}  "
            f"time={elapsed:.0f}s"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss, "config": config, "id2genre": id2genre},
                       best_path)
            log.info(f"  ✓ New best saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            log.info(f"  No improvement. Patience {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                log.info("Early stopping.")
                break

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss, "config": config, "id2genre": id2genre},
               output_dir / "final_model.pt")
    log.info(f"Final model saved. Best val_loss: {best_val_loss:.4f}")

    # ── Sanity-check generation ────────────────────────────────────────────────
    log.info("Running demo generation (sanity check)...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    demo_genre = 0
    tokens = model.generate(genre_id=demo_genre, max_new_tokens=128,
                             temperature=args.temperature, top_p=args.top_p, device=str(device))
    log.info(f"Generated {len(tokens)} tokens for genre '{id2genre[demo_genre]}': {tokens[:20]}...")
    with open(output_dir / "demo_generation.json", "w") as f:
        json.dump({"genre": id2genre[demo_genre], "tokens": tokens}, f)

    log.info(f"Done! Best model is at {best_path}")
    send_notification(f'LSTM Training finished for dataset: {args.dataset_name}, split: {split}')


if __name__ == "__main__":
    main()
