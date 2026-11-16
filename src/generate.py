"""
generate.py
===========
Unified MIDI generation module and CLI for all 4 model architectures:
  1. lstm        — GenreLSTM (genre embedding concatenated at every step, REMI+BPE)
  2. transformer — Custom 6-layer LlamaForCausalLM (genre prefix token, REMI+BPE)
  3. llama       — Fine-tuned LLaMA 3.2 1B (extended vocab + genre prefix token, REMI+BPE)
  4. moonbeam    — Fine-tuned Moonbeam 839M (multidimensional FME compound tokens, unconditional)

Designed for two use cases:
  A) Large-scale benchmark generation on PLGrid (e.g. 1,000 MIDI songs per model):
       python generate.py --model transformer --dataset-name gt_1000_with_ties --split piano --num-songs 1000
       python generate.py --model moonbeam --dataset-name top_5_no_ties --multi-instrument --num-songs 1000

  B) Single-song CLI / future Web UI (Gradio/Streamlit) inference:
       python generate.py --model lstm --genre jazz --temperature 0.95 --output sample_jazz.mid

     Or in Python UI code:
       from generate import MusicGenerator
       gen = MusicGenerator(model_type="transformer", dataset_name="gt_1000_with_ties", split="piano")
       midi_path = gen.generate_one(genre="classical", output_path="out.mid", temperature=1.0, top_p=0.9)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. LSTM Architecture Definition (must match train_lstm.py)
# ===========================================================================

class GenreLSTM(nn.Module):
    """3-layer LSTM with persistent genre embedding concatenated at every step."""

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
        self.vocab_size = vocab_size
        self.num_genres = num_genres
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

    def forward(self, input_ids: torch.Tensor, genre_ids: torch.Tensor, hidden=None):
        B, T = input_ids.shape
        tok = self.token_emb(input_ids)
        gen = self.genre_emb(genre_ids).unsqueeze(1).expand(-1, T, -1)
        x, hidden = self.lstm(torch.cat([tok, gen], dim=-1), hidden)
        return self.output_head(x), hidden

    @torch.inference_mode()
    def generate_batch(
        self,
        genre_ids: list[int],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.9,
        device: str | torch.device = "cpu",
    ) -> list[list[int]]:
        """Batched autoregressive top-p nucleus sampling for fast generation."""
        self.eval()
        B = len(genre_ids)
        genre_t = torch.tensor(genre_ids, dtype=torch.long, device=device)
        current_ids = torch.full((B, 1), self.pad_token_id, dtype=torch.long, device=device)
        hidden = None

        generated = torch.empty((B, max_new_tokens), dtype=torch.long, device=device)

        for step in range(max_new_tokens):
            logits, hidden = self.forward(current_ids, genre_t, hidden)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            # Prevent sampling PAD token
            logits[:, self.pad_token_id] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = (cumulative - sorted_probs) > top_p
                sorted_probs[mask] = 0.0
                sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12))
                sampled_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
                next_ids = torch.gather(sorted_idx, -1, sampled_in_sorted)
            else:
                next_ids = torch.multinomial(probs, num_samples=1)

            generated[:, step] = next_ids.squeeze(-1)
            current_ids = next_ids

        return generated.cpu().tolist()


# ===========================================================================
# 2. Path & Checkpoint Discovery Helpers
# ===========================================================================

def get_search_roots(sync_dir: Path) -> list[Path]:
    """
    Return candidate root directories where datasets and models may live:
    1. Explicit environment overrides ($MUSICGEN_MODELS_ROOT, $MUSICGEN_DATASET_BACKUP)
    2. $SCRATCH/models (flattened Ares/Athena structure), $SCRATCH/models_store, $SCRATCH/dataset_backup
    3. Primary $SYNC_DIR
    """
    roots = [sync_dir]
    scratch = os.getenv("SCRATCH")
    if scratch:
        scratch_p = Path(scratch)
        for sub in ["models", "dataset_backup", "musicgen"]:
            cand = scratch_p / sub
            if cand.exists():
                roots.insert(0, cand)
    for env_var in ["MUSICGEN_MODELS_ROOT", "MUSICGEN_DATASET_BACKUP"]:
        val = os.getenv(env_var)
        if val and Path(val).exists():
            roots.insert(0, Path(val))
    return roots


def load_genres_list(sync_dir: Path, dataset_name: str, split: str) -> list[str]:
    """
    Load sorted genre names from split/<split>/genre_weights.json (or train.csv).
    Sorting keys of genre_weights.json matches the exact alphabetical genre2id
    ordering created by tokenize_dataset.py.
    """
    repo_root = Path(__file__).resolve().parent.parent
    gw_candidates = [
        repo_root / "datasets_metadata" / dataset_name / "split" / split / "genre_weights.json",
        sync_dir / "datasets_metadata" / dataset_name / "split" / split / "genre_weights.json",
        sync_dir / "artist_genres" / dataset_name / "split" / split / "genre_weights.json",
    ]
    for gw_path in gw_candidates:
        if gw_path.exists():
            with open(gw_path, "r", encoding="utf-8") as f:
                gw = json.load(f)
            return sorted(gw.keys())

    csv_candidates = [
        repo_root / "datasets_metadata" / dataset_name / "split" / split / "train.csv",
        sync_dir / "datasets_metadata" / dataset_name / "split" / split / "train.csv",
        sync_dir / "artist_genres" / dataset_name / "split" / split / "train.csv",
    ]
    for csv_path in csv_candidates:
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return sorted(set(row["genre"].strip() for row in reader if row.get("genre", "").strip()))

    if "top_5" in dataset_name:
        return ["classical", "country", "metal", "pop", "rock"]
    return [
        "classical", "country", "electronic", "folk", "hip_hop", "jazz",
        "latin", "metal", "pop", "punk", "rock", "soul_rnb",
    ]


def find_remi_tokenizer_path(sync_dir: Path, dataset_name: str, split: str, explicit_path: Optional[str] = None) -> Path:
    """Locate REMI+BPE tokenizer.json across sync_dir, $SCRATCH/models, and $SCRATCH/dataset_backup."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_dir():
            p = p / "tokenizer.json"
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit tokenizer path not found: {explicit_path}")

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "datasets_metadata" / dataset_name / "split" / split / "tokenizer.json",
        repo_root / "datasets_metadata" / dataset_name / "dataset" / split / "remi_bpe" / "tokenizer" / "tokenizer.json",
        sync_dir / "datasets_metadata" / dataset_name / "split" / split / "tokenizer.json",
        sync_dir / "datasets_metadata" / dataset_name / "dataset" / split / "remi_bpe" / "tokenizer" / "tokenizer.json",
        sync_dir / "artist_genres" / dataset_name / "split" / split / "tokenizer.json",
        sync_dir / "artist_genres" / dataset_name / "dataset" / split / "remi_bpe" / "tokenizer" / "tokenizer.json",
    ]
    scratch = os.getenv("SCRATCH")
    if scratch:
        sp = Path(scratch)
        candidates.extend([
            sp / "models" / dataset_name / "tokenizer" / split / "tokenizer.json",
            sp / "models" / dataset_name / split / "tokenizer.json",
            sp / "dataset_backup" / dataset_name / "dataset" / split / "remi_bpe" / "tokenizer" / "tokenizer.json",
            sp / "dataset_backup" / dataset_name / split / "remi_bpe" / "tokenizer" / "tokenizer.json",
        ])
    tscratch_env = os.getenv("TSCRATCH")
    if tscratch_env and Path(tscratch_env).exists():
        tsp = Path(tscratch_env)
        candidates.extend([
            tsp / "dataset_backup" / dataset_name / split / "remi_bpe" / "tokenizer" / "tokenizer.json",
            tsp / "dataset_backup" / dataset_name / "dataset" / split / "remi_bpe" / "tokenizer" / "tokenizer.json",
        ])
    for env_k in ["SCRATCH", "TSCRATCH", "TMPDIR"]:
        env_val = os.getenv(env_k)
        if env_val:
            scratch_path = Path(env_val)
            candidates.extend([
                scratch_path / "dataset_backup" / dataset_name / split / "remi_bpe" / "tokenizer" / "tokenizer.json",
                scratch_path / "dataset_backup" / dataset_name / "dataset" / split / "remi_bpe" / "tokenizer" / "tokenizer.json",
            ])

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not find REMI+BPE tokenizer.json for {dataset_name}/{split}. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def find_latest_checkpoint(
    sync_dir: Path,
    model_type: str,
    dataset_name: str,
    split: str,
    explicit_ckpt: Optional[str] = None,
) -> Path:
    """
    Auto-discover the best/latest checkpoint for (model_type, dataset_name, split)
    supporting both the flattened structure ($SCRATCH/models/<dataset>/<model>/<split>/)
    and the nested run_* structure ($SYNC_DIR/artist_genres/<dataset>/models/<model>/<split>/run_*).
    """
    if explicit_ckpt:
        p = Path(explicit_ckpt)
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit checkpoint path does not exist: {explicit_ckpt}")

    search_bases = []
    for root in get_search_roots(sync_dir):
        search_bases.extend([
            root / dataset_name / model_type / split,
            root / "models" / dataset_name / model_type / split,
            root / "artist_genres" / dataset_name / "models" / model_type / split,
            root / dataset_name / "models" / model_type / split,
            root / model_type / dataset_name / split,
        ])

    # Build candidate directories to inspect:
    # 1. Direct flattened directory (e.g. $SCRATCH/models/top_5_with_ties/transformer/piano)
    # 2. Any run_<JOBID> subdirectories sorted newest first
    candidate_dirs = []
    seen = set()
    for base in search_bases:
        if base.exists() and base not in seen:
            seen.add(base)
            candidate_dirs.append(base)
            run_subdirs = [d for d in base.glob("run_*") if d.is_dir()]
            run_subdirs.sort(
                key=lambda p: int(p.name.replace("run_", "")) if p.name.replace("run_", "").isdigit() else 0,
                reverse=True,
            )
            for rd in run_subdirs:
                if rd not in seen:
                    seen.add(rd)
                    candidate_dirs.append(rd)

    for cand_dir in candidate_dirs:
        if model_type == "lstm":
            for fname in ["best_model.pt", "final_model.pt"]:
                cand = cand_dir / fname
                if cand.exists() and cand.stat().st_size > 0:
                    return cand
        elif model_type in ("transformer", "llama"):
            def _is_valid_hf_dir(d: Path) -> bool:
                if not d.exists() or not (d / "config.json").exists():
                    return False
                if (d / "config.json").stat().st_size == 0:
                    return False
                idx_file = d / "model.safetensors.index.json"
                if idx_file.exists() and idx_file.stat().st_size == 0:
                    return False
                weights = list(d.glob("*.safetensors")) + list(d.glob("pytorch_model*.bin"))
                return bool(weights) and all(w.stat().st_size > 0 for w in weights)

            best_dir = cand_dir / "best_model"
            if _is_valid_hf_dir(best_dir):
                return best_dir
            if _is_valid_hf_dir(cand_dir):
                return cand_dir
            ckpts = sorted(
                [d for d in cand_dir.glob("checkpoint-*") if d.is_dir()],
                key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0,
                reverse=True,
            )
            for ck in ckpts:
                if _is_valid_hf_dir(ck):
                    return ck
        elif model_type == "moonbeam":
            for fname in ["best_model.pt", "moonbeam_839M-latest.pt"]:
                cand = cand_dir / fname
                if cand.exists() and cand.stat().st_size > 0:
                    return cand
                cand_sub = cand_dir / "ddp-moonbeam_839M" / fname
                if cand_sub.exists() and cand_sub.stat().st_size > 0:
                    return cand_sub
            pts = sorted(cand_dir.rglob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            for pt in pts:
                if "optimizer" not in pt.name and pt.stat().st_size > 0:
                    return pt

    raise FileNotFoundError(
        f"No valid checkpoint found for model={model_type}, dataset={dataset_name}, split={split}. "
        f"Searched in: {[str(b) for b in search_bases]}"
    )


# ===========================================================================
# 3. Moonbeam Standalone / Prompted Generation Helpers
# ===========================================================================

def _moonbeam_compound_to_midi_safe(tokens, time_resolution: int = 100):
    """
    Convert Moonbeam (N, 6) compound tokens [onset, dur, oct, pitch_class, instr, vel]
    to a mido.MidiFile, safely wrapping MIDI channels to 0..15 (skipping drum channel 9
    for pitched instruments) so mido never fails on dense multi-instrument tracks.
    """
    import mido

    mid = mido.MidiFile()
    mid.ticks_per_beat = time_resolution // 2  # 2 beats/sec at 120 BPM

    time_index = defaultdict(list)
    for row in tokens:
        time_in_ticks, duration, octave, pitch_class, instrument, velocity = [int(x) for x in row]
        if time_in_ticks < 0 or duration <= 0:
            continue
        note = int(octave * 12 + pitch_class)
        note = max(0, min(127, note))
        velocity = max(1, min(127, velocity))
        instrument = max(0, min(128, instrument))
        time_index[(time_in_ticks, 0)].append((note, instrument, velocity))
        time_index[(time_in_ticks + duration, 1)].append((note, instrument, velocity))

    # Available pitched channels (0..15 excluding 9 which is reserved for drums)
    pitched_channels = [c for c in range(16) if c != 9]
    track_idx = {}
    pitched_counter = 0

    for time_in_ticks, event_type in sorted(time_index.keys()):
        for note, instrument, velocity in time_index[(time_in_ticks, event_type)]:
            if event_type == 0:  # note_on
                if instrument not in track_idx:
                    track = mido.MidiTrack()
                    mid.tracks.append(track)
                    if instrument == 128:
                        ch = 9
                        track.append(mido.Message("program_change", channel=ch, program=0, time=0))
                    else:
                        ch = pitched_channels[pitched_counter % len(pitched_channels)]
                        pitched_counter += 1
                        track.append(mido.Message("program_change", channel=ch, program=min(127, instrument), time=0))
                    track_idx[instrument] = [track, 0, ch]

                track, prev_time, ch = track_idx[instrument]
                delta = max(0, time_in_ticks - prev_time)
                track.append(mido.Message("note_on", note=note, channel=ch, velocity=velocity, time=delta))
                track_idx[instrument][1] = time_in_ticks
            else:  # note_off
                if instrument not in track_idx:
                    continue
                track, prev_time, ch = track_idx[instrument]
                delta = max(0, time_in_ticks - prev_time)
                track.append(mido.Message("note_off", note=note, channel=ch, velocity=0, time=delta))
                track_idx[instrument][1] = time_in_ticks

    return mid


# ===========================================================================
# 4. Unified MusicGenerator Class (Reusable for CLI & Future Web UI)
# ===========================================================================

class MusicGenerator:
    """
    Unified generator for LSTM, Custom Transformer, LLaMA 3.2 1B, and Moonbeam 839M.
    Can be instantiated once in a Web UI or CLI script and called repeatedly.
    """

    def __init__(
        self,
        model_type: str,
        dataset_name: str = "gt_1000_with_ties",
        split: str = "piano",
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        sync_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        self.model_type = model_type.lower().strip()
        if self.model_type not in ("lstm", "transformer", "llama", "moonbeam"):
            raise ValueError(f"Unsupported model_type '{model_type}'. Choose from: lstm, transformer, llama, moonbeam.")

        self.dataset_name = dataset_name
        self.split = split
        self.sync_dir = Path(sync_dir or os.getenv("MUSICGEN_SYNC_DIR", Path(__file__).resolve().parent))
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.genres = load_genres_list(self.sync_dir, self.dataset_name, self.split)
        self.genre2id = {g: i for i, g in enumerate(self.genres)}
        self.id2genre = {i: g for i, g in enumerate(self.genres)}
        self.num_genres = len(self.genres)

        self.checkpoint_path = find_latest_checkpoint(
            self.sync_dir, self.model_type, self.dataset_name, self.split, checkpoint_path
        )
        log.info(f"[{self.model_type.upper()}] Using checkpoint: {self.checkpoint_path}")
        log.info(f"[{self.model_type.upper()}] Device: {self.device} | Genres ({self.num_genres}): {self.genres}")

        self.remi_tok = None
        self.base_vocab_size = 0
        self.pad_token_id = 0
        self.original_vocab_size = 0
        self.model = None
        self.mb_tokenizer = None
        self.mb_config = None
        self.mb_test_files: list[Path] = []

        if self.model_type in ("lstm", "transformer", "llama"):
            self._init_remi_tokenizer(tokenizer_path)

        if self.model_type == "lstm":
            self._load_lstm()
        elif self.model_type == "transformer":
            self._load_transformer()
        elif self.model_type == "llama":
            self._load_llama()
        elif self.model_type == "moonbeam":
            self._load_moonbeam()

    def _init_remi_tokenizer(self, explicit_tok_path: Optional[str]):
        import tempfile
        from miditok import REMI

        tok_file = find_remi_tokenizer_path(self.sync_dir, self.dataset_name, self.split, explicit_tok_path)
        log.info(f"Loading REMI+BPE tokenizer from {tok_file}...")
        try:
            self.remi_tok = REMI(params=str(tok_file))
        except Exception as e:
            if "ModelWrapper" not in str(e):
                raise
            log.info("Detected older HuggingFace `tokenizers` (<0.20); converting BPE merges from list-of-pairs to strings...")
            with open(tok_file, "r", encoding="utf-8") as f:
                tok_data = json.load(f)
            model_json = json.loads(tok_data["_model"])
            merges = model_json.get("model", {}).get("merges", [])
            if merges and isinstance(merges[0], list):
                model_json["model"]["merges"] = [f"{a} {b}" for a, b in merges]
            tok_data["_model"] = json.dumps(model_json)
            with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as tf:
                json.dump(tok_data, tf)
                tmp_path = tf.name
            try:
                self.remi_tok = REMI(params=tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        self.base_vocab_size = len(self.remi_tok)
        self.pad_token_id = self.remi_tok["PAD_None"]

    def _load_lstm(self):
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        except Exception as e:
            alt = self.checkpoint_path.parent / "final_model.pt"
            if alt.exists() and alt != self.checkpoint_path and alt.stat().st_size > 0:
                log.warning(f"Failed loading {self.checkpoint_path} ({e}); falling back to {alt}")
                self.checkpoint_path = alt
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            else:
                raise RuntimeError(
                    f"Corrupted LSTM checkpoint at {self.checkpoint_path} ({e}). "
                    f"Please re-copy best_model.pt/final_model.pt from Athena $SCRATCH."
                ) from e
        cfg = ckpt.get("config", {})
        if "id2genre" in ckpt and ckpt["id2genre"]:
            self.id2genre = {int(k): v for k, v in ckpt["id2genre"].items()}
            self.genre2id = {v: k for k, v in self.id2genre.items()}
            self.genres = [self.id2genre[i] for i in sorted(self.id2genre.keys())]
            self.num_genres = len(self.genres)

        self.model = GenreLSTM(
            vocab_size=cfg.get("vocab_size", self.base_vocab_size),
            num_genres=cfg.get("num_genres", self.num_genres),
            token_embed_dim=cfg.get("token_embed_dim", 256),
            genre_embed_dim=cfg.get("genre_embed_dim", 32),
            hidden_size=cfg.get("hidden_size", 512),
            num_layers=cfg.get("num_layers", 3),
            dropout=cfg.get("dropout", 0.1),
            pad_token_id=cfg.get("pad_token_id", self.pad_token_id),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _load_transformer(self):
        from transformers import LlamaForCausalLM

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = LlamaForCausalLM.from_pretrained(
            str(self.checkpoint_path),
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def _load_llama(self):
        from transformers import LlamaForCausalLM

        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            # bfloat16 on modern Xeon CPUs (AVX512/oneDNN) halves memory bandwidth and speeds up GEMM ~2.5x
            dtype = torch.bfloat16

        self.model = LlamaForCausalLM.from_pretrained(
            str(self.checkpoint_path),
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        total_vocab = self.model.config.vocab_size
        self.original_vocab_size = total_vocab - self.base_vocab_size - self.num_genres
        active_vocab_size = self.base_vocab_size + self.num_genres

        # Prune the 128,256 unused English text tokens from embed_tokens and lm_head.
        # This removes ~525M dead parameters, shrinks lm_head and Top-P sorting by 13.8x,
        # and keeps exact mathematical equivalence for all MIDI + genre tokens.
        if self.original_vocab_size > 0:
            with torch.no_grad():
                old_emb = self.model.model.embed_tokens
                new_emb = nn.Embedding(
                    active_vocab_size,
                    old_emb.embedding_dim,
                    dtype=old_emb.weight.dtype,
                    device=old_emb.weight.device,
                )
                new_emb.weight.copy_(old_emb.weight[self.original_vocab_size:])
                self.model.model.embed_tokens = new_emb

                old_head = self.model.lm_head
                new_head = nn.Linear(
                    old_head.in_features,
                    active_vocab_size,
                    bias=old_head.bias is not None,
                    dtype=old_head.weight.dtype,
                    device=old_head.weight.device,
                )
                new_head.weight.copy_(old_head.weight[self.original_vocab_size:])
                if old_head.bias is not None:
                    new_head.bias.copy_(old_head.bias[self.original_vocab_size:])
                self.model.lm_head = new_head
                self.model.config.vocab_size = active_vocab_size
                self.model.config.pad_token_id = self.pad_token_id
                self.model.config.bos_token_id = None
                self.model.config.eos_token_id = None

        log.info(
            f"[LLaMA] Pruned dead text vocab ({self.original_vocab_size:,} tokens removed) -> "
            f"active_vocab={active_vocab_size:,} (midi={self.base_vocab_size:,}, genres={self.num_genres}), dtype={dtype}"
        )

    def _load_moonbeam(self):
        mb_root = self.sync_dir / "Moonbeam-MIDI-Foundation-Model"
        if not mb_root.exists():
            raise FileNotFoundError(f"Moonbeam repo not found at {mb_root}")

        # Prepend Moonbeam's custom transformers_minimal so its custom LlamaForCausalLM (with FME/MRA) is loaded
        tf_min = str(mb_root / "src" / "llama_recipes" / "transformers_minimal" / "src")
        mb_src = str(mb_root / "src")
        for p in [ mb_src, tf_min ]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # Clear any already-imported standard transformers modules if needed
        for mod_name in list(sys.modules.keys()):
            if mod_name == "transformers" or mod_name.startswith("transformers."):
                del sys.modules[mod_name]

        from transformers import LlamaConfig, LlamaForCausalLM
        from llama_recipes.datasets.music_tokenizer import MusicTokenizer

        cfg_path = mb_root / "src" / "llama_recipes" / "configs" / "model_config.json"
        self.mb_config = LlamaConfig.from_pretrained(str(cfg_path))
        self.model = LlamaForCausalLM(self.mb_config)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        clean_sd = { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }
        self.model.load_state_dict(clean_sd, strict=False)

        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model = self.model.to(device=self.device, dtype=dtype)
        else:
            self.model = self.model.to(device=self.device)
        self.model.eval()

        self.mb_tokenizer = MusicTokenizer(
            timeshift_vocab_size=self.mb_config.onset_vocab_size,
            dur_vocab_size=self.mb_config.dur_vocab_size,
            octave_vocab_size=self.mb_config.octave_vocab_size,
            pitch_class_vocab_size=self.mb_config.pitch_class_vocab_size,
            instrument_vocab_size=self.mb_config.instrument_vocab_size,
            velocity_vocab_size=self.mb_config.velocity_vocab_size,
            sos_token=self.mb_config.sos_token,
            eos_token=self.mb_config.eos_token,
            pad_token=self.mb_config.pad_token,
        )

        # Discover optional test .npy files for short 5-note prompts if available
        candidates = [
            self.sync_dir / "artist_genres" / self.dataset_name / "dataset" / self.split / "moonbeam_preprocessed" / "test",
        ]
        scratch = os.getenv("SCRATCH")
        if scratch:
            sp = Path(scratch)
            candidates.extend([
                sp / "dataset_backup" / self.dataset_name / "dataset" / self.split / "moonbeam_preprocessed" / "test",
                sp / "dataset_backup" / self.dataset_name / self.split / "moonbeam_preprocessed" / "test",
            ])
        for c in candidates:
            if c.exists():
                self.mb_test_files = sorted(c.glob("*.npy"))
                if self.mb_test_files:
                    log.info(f"[Moonbeam] Found {len(self.mb_test_files):,} test .npy files for prompt seeding in {c}")
                    break

    # -----------------------------------------------------------------------
    # Token Decoding to MIDI
    # -----------------------------------------------------------------------

    def decode_remi_to_midi(self, token_ids: list[int], output_path: str | Path) -> int:
        """
        Filter valid REMI+BPE token IDs, decode to symusic.Score, and dump as .mid.
        Handles both 1D ('T',) token streams (multi-instrument, one_token_stream=True)
        and 2D ('I', 'T') token streams (piano, one_token_stream=False).
        Returns the number of valid MIDI tokens decoded.
        """
        from miditok import TokSequence

        valid_ids = [
            int(t) for t in token_ids
            if 0 <= int(t) < self.base_vocab_size and int(t) != self.pad_token_id
        ]
        if not valid_ids:
            return 0

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        tok_seq = TokSequence(ids=valid_ids, are_ids_encoded=True)
        if getattr(self.remi_tok, "one_token_stream", True):
            decode_input = tok_seq
        else:
            decode_input = [tok_seq]

        score = self.remi_tok.decode(decode_input)
        if hasattr(score, "dump_midi"):
            score.dump_midi(str(out_p))
        else:
            score.dump(str(out_p))
        return len(valid_ids)

    # -----------------------------------------------------------------------
    # Batch Generation Methods
    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def generate_batch_sequences(
        self,
        genre: Optional[str | int | list[str | int]] = None,
        batch_size: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.9,
        prompt_len: int = 5,
    ) -> list:
        """
        Generate a batch of token sequences for the given genre(s) (or unconditional for Moonbeam).
        `genre` can be a single genre name/ID (replicated across `batch_size`) or a list of
        genre names/IDs of length `batch_size` (allowing mixed-genre batching with zero tail waste).
        """
        if self.model_type == "moonbeam":
            return self._generate_moonbeam_batch(
                batch_size=batch_size,
                max_gen_len=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                prompt_len=prompt_len,
            )

        def _resolve_one_genre(g) -> int:
            if g is None:
                return 0
            if isinstance(g, int):
                return g
            g_clean = g.strip().lower()
            if g_clean not in self.genre2id:
                raise ValueError(f"Unknown genre '{g}'. Available genres: {self.genres}")
            return self.genre2id[g_clean]

        if isinstance(genre, list):
            genre_ids = [_resolve_one_genre(g) for g in genre]
            batch_size = len(genre_ids)
        else:
            gid = _resolve_one_genre(genre)
            genre_ids = [gid] * batch_size

        if self.model_type == "lstm":
            return self.model.generate_batch(
                genre_ids=genre_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=self.device,
            )

        elif self.model_type in ("transformer", "llama"):
            from transformers import LogitsProcessor, LogitsProcessorList

            # Both Custom Transformer and vocab-pruned LLaMA have active vocab [0, base_vocab_size + num_genres)
            genre_tokens = [self.base_vocab_size + gid for gid in genre_ids]
            input_ids = torch.tensor(genre_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
            attention_mask = torch.ones_like(input_ids)

            class ValidRemiLogitsProcessor(LogitsProcessor):
                def __init__(self, base_vocab: int, pad_id: int):
                    self.base_vocab = base_vocab
                    self.pad_id = pad_id

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                    scores[:, self.base_vocab:] = float("-inf")
                    scores[:, self.pad_id] = float("-inf")
                    return scores

            processors = LogitsProcessorList([ValidRemiLogitsProcessor(self.base_vocab_size, self.pad_token_id)])
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                logits_processor=processors,
                pad_token_id=self.pad_token_id,
                bos_token_id=None,
                eos_token_id=None,
                use_cache=True,
            )
            # Strip prepended genre prefix token
            return outputs[:, 1:].cpu().tolist()

    @torch.inference_mode()
    def _generate_moonbeam_batch(
        self,
        batch_size: int = 4,
        max_gen_len: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        prompt_len: int = 5,
    ) -> list[list[list[int]]]:
        """
        Batched generation for Moonbeam 839M using its dual-stage (Transformer + GRU attribute decoder)
        compound token architecture. Works both with test .npy prompt seeds and standalone from scratch.
        """
        scales = [
            [0, 2, 4, 5, 7, 9, 11],  # Major
            [0, 2, 3, 5, 7, 8, 10],  # Natural Minor
            [0, 2, 4, 7, 9],         # Major Pentatonic
            [0, 3, 5, 7, 10],        # Minor Pentatonic
            [0, 2, 3, 5, 7, 9, 10],  # Dorian
        ]
        multi_instr_pool = [0, 0, 24, 25, 32, 33, 40, 48, 73]

        prompts = []
        for _ in range(batch_size):
            if self.mb_test_files and prompt_len > 1:
                npy_path = random.choice(self.mb_test_files)
                try:
                    arr = np.load(str(npy_path))
                    encoded = self.mb_tokenizer.encode_series(arr, if_add_sos=True, if_add_eos=False)
                    # Ensure the test prompt actually advances in time (not all at onset=0)
                    if len(encoded) >= prompt_len and encoded[prompt_len - 1][0] > encoded[1][0]:
                        prompts.append(encoded[:prompt_len])
                        continue
                except Exception:
                    pass

            # Fallback / scratch prompt: SOS + 4 progressive notes with advancing onsets,
            # varied durations, velocities, and pitches so MRA establishes forward tempo immediately.
            scale = random.choice(scales)
            root_shift = random.randint(0, 11)
            cur_onset = 0
            synth_notes = [self.mb_tokenizer.sos_token_compound]
            n_seed_notes = max(4, prompt_len - 1)
            for step_i in range(n_seed_notes):
                if step_i > 0:
                    cur_onset += random.randint(20, 55)  # 0.20s - 0.55s inter-onset interval
                dur = random.randint(20, 65)
                octv = random.choice([3, 4, 5])
                pc = (random.choice(scale) + root_shift) % 12
                instr = 0 if self.split == "piano" else random.choice(multi_instr_pool)
                vel = random.randint(58, 96)
                synth_notes.append([cur_onset, dur, octv, pc, instr, vel])
            prompts.append(synth_notes)

        bsz = len(prompts)
        min_prompt_len = min(len(t) for t in prompts)
        max_prompt_len = max(len(t) for t in prompts)
        max_ctx = getattr(self.mb_config, "max_len", 1024)
        total_len = min(max_ctx, max_gen_len + max_prompt_len)

        pad_id = self.mb_tokenizer.pad_token_compound
        pad_tensor = torch.tensor(pad_id, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(0)
        tokens = pad_tensor.expand(bsz, total_len, -1).clone()

        for k, t in enumerate(prompts):
            t_tensor = torch.tensor(t, dtype=torch.long, device=self.device)
            tokens[k, :len(t)] = t_tensor

        prev_pos = 0
        eos_reached = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        input_mask = torch.all(tokens != pad_tensor, dim=-1).unsqueeze(-1)
        past_key_values = None

        attr_names = [
            "timeshift_dict_decode",
            "duration_dict_decode",
            "octave_dict_decode",
            "pitch_dict_decode",
            "instrument_dict_decode",
            "velocity_dict_decode",
        ]

        # Build clean valid vocabulary index tensors:
        # - Exclude SOS attribute tokens (never valid as generated outputs)
        # - Exclude duration=0 and velocity=0 (degenerate silent/zero-length notes)
        # - Separate non-EOS vs with-EOS sets so EOS is blocked for the first min_gen_notes=64 steps
        sos_raw = {
            "timeshift_dict_decode": self.mb_tokenizer.sos_timeshift,
            "duration_dict_decode": self.mb_tokenizer.sos_dur,
            "octave_dict_decode": self.mb_tokenizer.sos_octave,
            "pitch_dict_decode": self.mb_tokenizer.sos_pitch_class,
            "instrument_dict_decode": self.mb_tokenizer.sos_instrument,
            "velocity_dict_decode": self.mb_tokenizer.sos_velocity,
        }
        eos_raw = {
            "timeshift_dict_decode": self.mb_tokenizer.eos_timeshift,
            "duration_dict_decode": self.mb_tokenizer.eos_dur,
            "octave_dict_decode": self.mb_tokenizer.eos_octave,
            "pitch_dict_decode": self.mb_tokenizer.eos_pitch_class,
            "instrument_dict_decode": self.mb_tokenizer.eos_instrument,
            "velocity_dict_decode": self.mb_tokenizer.eos_velocity,
        }

        allowed_no_eos = {}
        allowed_with_eos = {}
        for attr in attr_names:
            decode_map = getattr(self.mb_tokenizer, attr)
            no_eos_list = []
            with_eos_list = []
            for lang_tok, raw_val in decode_map.items():
                if raw_val == sos_raw[attr]:
                    continue
                if attr == "duration_dict_decode" and raw_val <= 0:
                    continue
                if attr == "velocity_dict_decode" and raw_val <= 0:
                    continue
                if attr == "octave_dict_decode" and (raw_val < 1 or (raw_val > 9 and raw_val != eos_raw[attr])):
                    continue
                if attr == "pitch_dict_decode" and (raw_val > 11 and raw_val != eos_raw[attr]):
                    continue
                with_eos_list.append(lang_tok)
                if raw_val != eos_raw[attr]:
                    no_eos_list.append(lang_tok)
            allowed_no_eos[attr] = torch.tensor(no_eos_list, dtype=torch.long, device=self.device)
            allowed_with_eos[attr] = torch.tensor(with_eos_list, dtype=torch.long, device=self.device)

        zero_timeshift_lang_tok = self.mb_tokenizer.timeshift_dict[0]
        consecutive_zero_ts = torch.zeros(bsz, dtype=torch.long, device=self.device)
        min_gen_notes = 64

        for cur_pos in range(min_prompt_len, total_len):
            output = self.model.forward(
                input_ids=tokens[:, prev_pos:cur_pos],
                past_key_values=past_key_values,
                use_cache=True,
                attention_mask=None,
            )
            seq_chunk = cur_pos - prev_pos
            next_decoder_token = torch.tensor(self.mb_tokenizer.sos_out, device=self.device, dtype=torch.long).expand(
                bsz * seq_chunk, 1
            )
            next_decoder_token_out = next_decoder_token
            hidden_state = output.logits
            hidden_state = (
                hidden_state.view(bsz * seq_chunk, hidden_state.shape[2])
                .unsqueeze(0)
                .expand(self.model.decoder.num_hidden_layers, -1, -1)
                .contiguous()
            )

            allow_eos_now = (cur_pos - min_prompt_len) >= min_gen_notes

            for attr in attr_names:
                output_decoder = self.model.forward(
                    decoded_hidden_state=hidden_state,
                    decoded_language_tokens=next_decoder_token,
                    attention_mask=None,
                )
                generation_logits = output_decoder.generation_logits[:, -1, :]  # (bsz*seq_chunk, vocab)
                hidden_state = output_decoder.generation_hidden_state

                valid_idx = allowed_with_eos[attr] if allow_eos_now else allowed_no_eos[attr]
                masked_logits = torch.full_like(generation_logits, float("-inf"))
                masked_logits[:, valid_idx] = generation_logits[:, valid_idx]

                # Anti-collapse chord guard: if a sequence already has >= 4 consecutive notes
                # at the exact same onset (a 5-note chord), force timeshift > 0 on the next note
                if attr == "timeshift_dict_decode":
                    reshaped_logits = masked_logits.view(bsz, seq_chunk, -1)
                    stuck_mask = consecutive_zero_ts >= 4
                    if torch.any(stuck_mask):
                        reshaped_logits[stuck_mask, -1, zero_timeshift_lang_tok] = float("-inf")
                    masked_logits = reshaped_logits.view(bsz * seq_chunk, -1)

                if temperature > 0:
                    probs = torch.softmax(masked_logits / temperature, dim=-1)
                    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                    probs_sum = torch.cumsum(probs_sort, dim=-1)
                    probs_sort[probs_sum - probs_sort > top_p] = 0.0
                    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True).clamp(min=1e-12))
                    sampled = torch.multinomial(probs_sort, num_samples=1)
                    next_decoder_token = torch.gather(probs_idx, -1, sampled)
                else:
                    next_decoder_token = masked_logits.argmax(dim=-1, keepdim=True)

                next_decoder_token_out = torch.cat([next_decoder_token_out, next_decoder_token], dim=-1)

            next_decoder_token_out_reshaped = next_decoder_token_out[:, 1:].view(bsz, -1, 6)
            next_decoder_token_lang = self.mb_tokenizer.convert_from_language_tokens(
                next_decoder_token_out_reshaped.cpu()
            ).to(self.device)

            sampled_ts = next_decoder_token_lang[:, -1, 0]
            consecutive_zero_ts = torch.where(
                sampled_ts == 0,
                consecutive_zero_ts + 1,
                torch.zeros_like(consecutive_zero_ts),
            )

            previous_onset = tokens[:, cur_pos - 1, 0].clamp(min=0)
            new_onset = previous_onset + sampled_ts
            next_decoder_token_onset = torch.cat(
                [new_onset.unsqueeze(-1), next_decoder_token_lang[:, -1, 1:]], dim=-1
            ).to(tokens.dtype)

            next_token = torch.where(input_mask[:, cur_pos], tokens[:, cur_pos], next_decoder_token_onset)
            tokens[:, cur_pos] = next_token

            last_lang = next_decoder_token_lang[:, -1, :]
            eos_cond = (
                (last_lang[:, 0] == self.mb_tokenizer.eos_timeshift)
                | (last_lang[:, 1] == self.mb_tokenizer.eos_dur)
                | (last_lang[:, 2] == self.mb_tokenizer.eos_octave)
                | (last_lang[:, 3] == self.mb_tokenizer.eos_pitch_class)
                | (last_lang[:, 4] == self.mb_tokenizer.eos_instrument)
                | (last_lang[:, 5] == self.mb_tokenizer.eos_velocity)
            )
            eos_reached |= (~input_mask[:, cur_pos].squeeze(-1)) & eos_cond
            prev_pos = cur_pos
            past_key_values = output.past_key_values
            if torch.all(eos_reached):
                break

        # Strip SOS token at index 0 and truncate at first EOS attribute
        tokens_list = tokens[:, 1:, :].cpu().tolist()
        stop_tokens = [
            self.mb_tokenizer.eos_timeshift,
            self.mb_tokenizer.eos_dur,
            self.mb_tokenizer.eos_octave,
            self.mb_tokenizer.eos_pitch_class,
            self.mb_tokenizer.eos_instrument,
            self.mb_tokenizer.eos_velocity,
        ]
        out_sequences = []
        for i, toks in enumerate(tokens_list):
            toks = toks[: (len(prompts[i]) - 1) + max_gen_len]
            for j, stop_tok in enumerate(stop_tokens):
                if j == 0:
                    continue
                for row_idx, row in enumerate(toks):
                    if row[j] == stop_tok or row[j] < 0:
                        toks = toks[:row_idx]
                        break
            out_sequences.append(toks)
        return out_sequences

    def save_sequence_as_midi(self, seq: list, output_path: str | Path) -> int:
        """Save a single generated sequence (REMI or Moonbeam compound) to a .mid file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if self.model_type == "moonbeam":
            if not seq:
                return 0
            midi_obj = _moonbeam_compound_to_midi_safe(seq)
            midi_obj.save(str(out_p))
            return len(seq)
        else:
            return self.decode_remi_to_midi(seq, out_p)

    def generate_one(
        self,
        output_path: str | Path,
        genre: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Convenience method for single-song CLI or Web UI (Gradio/Streamlit) generation.
        Generates 1 MIDI file and returns a metadata dictionary.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        t0 = time.time()
        seqs = self.generate_batch_sequences(
            genre=genre,
            batch_size=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        n_tokens = self.save_sequence_as_midi(seqs[0], output_path)
        elapsed = time.time() - t0
        return {
            "output_path": str(output_path),
            "model": self.model_type,
            "dataset": self.dataset_name,
            "split": self.split,
            "genre": genre if self.model_type != "moonbeam" else "unconditional",
            "num_tokens": n_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "elapsed_sec": round(elapsed, 2),
        }


import struct


def is_valid_midi_file(
    fpath: Path,
    min_duration_sec: float = 5.0,
    min_notes: int = 16,
    min_unique_onsets: int = 8,
) -> bool:
    """
    Fast binary MIDI quality check (<0.05 ms per file, zero external library overhead).
    Rejects empty, corrupted, or collapsed tracks (e.g., 1-2s tracks, single-note tracks,
    or Moonbeam chord-collapse tracks where all notes occur at onset=0).
    """
    try:
        if not fpath.exists():
            return False
        data = fpath.read_bytes()
        if len(data) < 40 or data[:4] != b"MThd":
            return False
        tpb = struct.unpack(">H", data[12:14])[0]
        if tpb <= 0:
            tpb = 480

        pos = 14
        notes = 0
        max_ticks = 0
        unique_onsets = set()
        tempos = []  # list of (tick, tempo_us_per_beat)

        while pos + 8 <= len(data):
            if data[pos : pos + 4] != b"MTrk":
                break
            tlen = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
            trk = data[pos + 8 : pos + 8 + tlen]
            pos += 8 + tlen

            i = 0
            tt = 0
            status = 0
            n_trk = len(trk)
            while i < n_trk:
                v = 0
                while i < n_trk:
                    b = trk[i]
                    i += 1
                    v = (v << 7) | (b & 0x7F)
                    if not (b & 0x80):
                        break
                tt += v
                if i >= n_trk:
                    break
                b = trk[i]
                if b & 0x80:
                    status = b
                    i += 1
                if status == 0xFF:
                    if i >= n_trk:
                        break
                    mtype = trk[i]
                    i += 1
                    mlen = 0
                    while i < n_trk:
                        b2 = trk[i]
                        i += 1
                        mlen = (mlen << 7) | (b2 & 0x7F)
                        if not (b2 & 0x80):
                            break
                    if mtype == 0x51 and mlen == 3 and i + 3 <= n_trk:
                        # set_tempo meta event: 3 bytes microsec per beat
                        tempo_val = (trk[i] << 16) | (trk[i + 1] << 8) | trk[i + 2]
                        tempos.append((tt, tempo_val))
                    i += mlen
                elif (status & 0xF0) in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    if (status & 0xF0) == 0x90 and i + 1 < n_trk and trk[i + 1] > 0:
                        notes += 1
                        unique_onsets.add(tt)
                    i += 2
                elif (status & 0xF0) in (0xC0, 0xD0):
                    i += 1
                else:
                    i += 1
            if tt > max_ticks:
                max_ticks = tt

        # Compute accurate duration in seconds honoring all tempo changes
        if not tempos:
            duration_sec = (max_ticks / tpb) * 0.5  # default 120 BPM
        else:
            tempos.sort(key=lambda x: x[0])
            duration_sec = 0.0
            cur_tick = 0
            cur_tempo = 500000  # default 120 BPM
            for t_tick, t_tempo in tempos:
                if t_tick > cur_tick:
                    chunk_ticks = min(max_ticks, t_tick) - cur_tick
                    duration_sec += (chunk_ticks / tpb) * (cur_tempo / 1_000_000.0)
                    cur_tick = min(max_ticks, t_tick)
                    if cur_tick >= max_ticks:
                        break
                cur_tempo = t_tempo
            if max_ticks > cur_tick:
                duration_sec += ((max_ticks - cur_tick) / tpb) * (cur_tempo / 1_000_000.0)

        return (
            duration_sec >= min_duration_sec
            and notes >= min_notes
            and len(unique_onsets) >= min_unique_onsets
        )
    except Exception:
        return False


# ===========================================================================
# 5. CLI Benchmark & Single-File Entrypoint
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate MIDI songs using trained LSTM, Transformer, LLaMA, or Moonbeam models."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "transformer", "llama", "moonbeam"],
        help="Model architecture to use for generation.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="gt_1000_with_ties",
        help="Dataset taxonomy variant (e.g. gt_1000_with_ties, gt_1000_no_ties, top_5_with_ties, top_5_no_ties).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="piano",
        choices=["piano", "multi-instrument"],
        help="Instrument split (piano or multi-instrument).",
    )
    parser.add_argument(
        "--multi-instrument",
        action="store_true",
        help="Shorthand flag for --split multi-instrument.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit path to model checkpoint file or folder (auto-discovered if omitted).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Optional explicit path to REMI+BPE tokenizer.json.",
    )

    # Single-file mode vs. Batch benchmark mode
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If set, generates a SINGLE MIDI file to this path (demo / UI mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for batch benchmark generation. Defaults to $SYNC_DIR/generated_midis/<dataset>/<split>/<model>.",
    )
    parser.add_argument(
        "--genre",
        type=str,
        default=None,
        help="Specific genre to generate (for single-file mode or single-genre batch run).",
    )
    parser.add_argument(
        "--num-songs",
        type=int,
        default=1000,
        help="Total number of MIDI songs to generate for this model (distributed evenly across genres). Default: 1000.",
    )
    parser.add_argument(
        "--songs-per-genre",
        type=int,
        default=None,
        help="If set, overrides --num-songs for conditioned models and generates exactly N songs per genre.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for parallel generation (auto-tuned per model if omitted).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max generated tokens per song (defaults to 1024 for REMI models, 512 notes for Moonbeam).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (defaults to 1.0 for REMI models, 0.8 for Moonbeam).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p threshold (default: 0.9).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even if target .mid files already exist.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="0-based shard index when splitting generation across multiple parallel jobs.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parallel shards.",
    )

    args = parser.parse_args()
    split = "multi-instrument" if args.multi_instrument else args.split
    sync_dir = Path(os.getenv("MUSICGEN_SYNC_DIR", Path(__file__).resolve().parent))

    # Sensible defaults per model family
    if args.max_tokens is None:
        args.max_tokens = 512 if args.model == "moonbeam" else 1024
    if args.temperature is None:
        args.temperature = 0.8 if args.model == "moonbeam" else 1.0
    if args.batch_size is None:
        if torch.cuda.is_available():
            default_bs = {"lstm": 64, "transformer": 32, "llama": 32, "moonbeam": 16}
        else:
            default_bs = {"lstm": 16, "transformer": 16, "llama": 16, "moonbeam": 8}
        args.batch_size = default_bs[args.model]

    if not torch.cuda.is_available():
        num_threads = int(os.environ.get("OMP_NUM_THREADS") or os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 16))
        torch.set_num_threads(num_threads)

    effective_seed = args.seed + args.shard_id * 10007
    torch.manual_seed(effective_seed)
    np.random.seed(effective_seed)
    random.seed(effective_seed)

    gen = MusicGenerator(
        model_type=args.model,
        dataset_name=args.dataset_name,
        split=split,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        sync_dir=sync_dir,
    )

    # ── Mode A: Single MIDI file generation (CLI / Demo / UI) ─────────────────
    if args.output:
        chosen_genre = args.genre or (gen.genres[0] if gen.genres else None)
        info = gen.generate_one(
            output_path=args.output,
            genre=chosen_genre,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        log.info(f"Generated single MIDI: {json.dumps(info, indent=2)}")
        return

    # ── Mode B: Batch Benchmark Generation (e.g. 1,000 songs per model) ───────
    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = sync_dir / "generated_midis" / args.dataset_name / split / args.model
    out_root.mkdir(parents=True, exist_ok=True)

    # Build generation plan: list of (genre_name, count_to_generate)
    if args.model == "moonbeam":
        if args.songs_per_genre:
            n_genres = 5 if "top_5" in args.dataset_name else 12
            plan = [("unconditional", args.songs_per_genre * n_genres)]
        else:
            plan = [("unconditional", args.num_songs)]
    elif args.genre:
        if args.genre not in gen.genre2id:
            raise ValueError(f"Unknown genre '{args.genre}'. Valid: {gen.genres}")
        count = args.songs_per_genre if args.songs_per_genre else args.num_songs
        plan = [(args.genre, count)]
    else:
        if args.songs_per_genre:
            plan = [(g, args.songs_per_genre) for g in gen.genres]
        else:
            base_count = args.num_songs // gen.num_genres
            remainder = args.num_songs % gen.num_genres
            plan = [
                (g, base_count + (1 if idx < remainder else 0))
                for idx, g in enumerate(gen.genres)
            ]

    total_planned = sum(c for _, c in plan)
    log.info("=" * 70)
    log.info(
        f"Batch Generation | model={args.model} | dataset={args.dataset_name} | "
        f"split={split} | total_songs={total_planned} | batch_size={args.batch_size}"
    )
    log.info(f"Output directory: {out_root}")
    log.info("=" * 70)

    manifest_path = out_root / "manifest.csv"
    manifest_rows = []
    start_all = time.time()
    generated_count = 0

    generation_log_path = out_root / "generation_log.csv"
    gen_log_rows = []

    # 1. Build the complete deterministic task list FIRST (before checking disk),
    #    so shard modulo slicing is 100% static and never shifts when other shards write files.
    all_tasks: list[tuple[str, int, Path]] = []
    for genre_name, target_count in plan:
        genre_dir = out_root / genre_name
        genre_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(target_count):
            fname = f"{args.model}_{genre_name}_{idx:04d}.mid"
            fpath = genre_dir / fname
            all_tasks.append((genre_name, idx, fpath))

    if args.num_shards > 1:
        shard_tasks = all_tasks[args.shard_id :: args.num_shards]
    else:
        shard_tasks = all_tasks

    total_planned = len(shard_tasks)

    # 2. Filter out already-generated VALID files within this shard's static assignment,
    #    and automatically delete + queue any defective/short (<5s or chord-collapsed) files.
    pending_tasks: list[tuple[str, int, Path]] = []
    removed_defective = 0
    for genre_name, idx, fpath in shard_tasks:
        genre_id = gen.genre2id.get(genre_name, -1)
        if not args.overwrite and is_valid_midi_file(fpath):
            manifest_rows.append({
                "song_id": f"{args.model}_{genre_name}_{idx:04d}",
                "file_path": str(fpath.relative_to(out_root)),
                "model": args.model,
                "dataset": args.dataset_name,
                "split": split,
                "genre": genre_name,
                "genre_id": genre_id,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_tokens": -1,
            })
            generated_count += 1
        else:
            if not args.overwrite and fpath.exists():
                try:
                    fpath.unlink()
                    removed_defective += 1
                except OSError:
                    pass
            pending_tasks.append((genre_name, idx, fpath))

    if removed_defective > 0:
        log.info(
            f"Filtered and removed {removed_defective} defective/short (<5s or collapsed) MIDI files; "
            f"queued them for clean regeneration."
        )

    if args.num_shards > 1:
        log.info(
            f"[Shard {args.shard_id}/{args.num_shards}] Static shard size={total_planned} | "
            f"already_done={generated_count} | pending={len(pending_tasks)}"
        )

    if not pending_tasks:
        log.info(f"All {total_planned} songs in this run/shard are valid and complete — skipping generation.")
    else:
        log.info(
            f"Generating {len(pending_tasks)}/{total_planned} pending songs across genres "
            f"(batch_size={args.batch_size})..."
        )
        for b_start in range(0, len(pending_tasks), args.batch_size):
            batch_items = pending_tasks[b_start : b_start + args.batch_size]
            cur_bs = len(batch_items)
            batch_genres = (
                [g for g, _, _ in batch_items] if args.model != "moonbeam" else None
            )

            t_batch = time.time()
            seqs = gen.generate_batch_sequences(
                genre=batch_genres,
                batch_size=cur_bs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            for seq, (genre_name, song_idx, fpath) in zip(seqs, batch_items):
                genre_id = gen.genre2id.get(genre_name, -1)
                song_id = f"{args.model}_{genre_name}_{song_idx:04d}"
                n_tok = 0
                attempts_used = 0
                first_attempt_valid = False

                for attempt in range(5):
                    attempts_used += 1
                    try:
                        if attempt > 0:
                            retry_genre = [genre_name] if args.model != "moonbeam" else None
                            seq = gen.generate_batch_sequences(
                                genre=retry_genre,
                                batch_size=1,
                                max_new_tokens=args.max_tokens,
                                temperature=min(1.15, max(args.temperature, 0.9) + 0.05 * attempt),
                                top_p=max(args.top_p, 0.92),
                            )[0]
                        n_tok = gen.save_sequence_as_midi(seq, fpath)
                        if n_tok > 0 and is_valid_midi_file(fpath):
                            if attempt == 0:
                                first_attempt_valid = True
                            break
                    except Exception as e:
                        log.warning(f"Attempt {attempt + 1} failed saving {fpath}: {e}")
                        n_tok = 0

                gen_log_rows.append({
                    "song_id": song_id,
                    "model": args.model,
                    "dataset": args.dataset_name,
                    "split": split,
                    "genre": genre_name,
                    "attempts": attempts_used,
                    "first_attempt_valid": int(first_attempt_valid),
                    "success": int(n_tok > 0 and fpath.exists() and fpath.stat().st_size > 0),
                })

                if n_tok > 0 and fpath.exists() and fpath.stat().st_size > 0:
                    manifest_rows.append({
                        "song_id": song_id,
                        "file_path": str(fpath.relative_to(out_root)),
                        "model": args.model,
                        "dataset": args.dataset_name,
                        "split": split,
                        "genre": genre_name,
                        "genre_id": genre_id,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "num_tokens": n_tok,
                    })
                    generated_count += 1

            elapsed_so_far = time.time() - start_all
            rate = generated_count / max(elapsed_so_far, 1e-6)
            log.info(
                f"  Batch ({cur_bs} songs) finished in {time.time() - t_batch:.1f}s | "
                f"Progress: {generated_count}/{total_planned} songs "
                f"({generated_count / total_planned * 100:.1f}%) | {rate:.2f} songs/s"
            )

    if generated_count == 0:
        raise RuntimeError(f"0/{total_planned} MIDI files were saved to {out_root}. Check errors above.")

    # Rebuild full manifest from all completed .mid files on disk (safe for multi-shard runs)
    token_counts = {row["song_id"]: row["num_tokens"] for row in manifest_rows}
    full_manifest_rows = []
    for genre_name, target_count in plan:
        genre_dir = out_root / genre_name
        genre_id = gen.genre2id.get(genre_name, -1)
        for idx in range(target_count):
            song_id = f"{args.model}_{genre_name}_{idx:04d}"
            fpath = genre_dir / f"{song_id}.mid"
            if fpath.exists() and fpath.stat().st_size > 0:
                full_manifest_rows.append({
                    "song_id": song_id,
                    "file_path": str(fpath.relative_to(out_root)),
                    "model": args.model,
                    "dataset": args.dataset_name,
                    "split": split,
                    "genre": genre_name,
                    "genre_id": genre_id,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "num_tokens": token_counts.get(song_id, -1),
                })

    fieldnames = [
        "song_id", "file_path", "model", "dataset", "split",
        "genre", "genre_id", "temperature", "top_p", "num_tokens",
    ]
    tmp_manifest = out_root / f".manifest.csv.tmp.{os.getpid()}"
    with open(tmp_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(full_manifest_rows)
    os.replace(tmp_manifest, manifest_path)

    # Save / append generation audit log (attempts & first-attempt validity)
    if gen_log_rows:
        log_fieldnames = ["song_id", "model", "dataset", "split", "genre", "attempts", "first_attempt_valid", "success"]
        write_header = not generation_log_path.exists()
        with open(generation_log_path, "a", newline="", encoding="utf-8") as f:
            log_writer = csv.DictWriter(f, fieldnames=log_fieldnames)
            if write_header:
                log_writer.writeheader()
            log_writer.writerows(gen_log_rows)
        log.info(f"Generation attempt log updated: {generation_log_path}")

    total_elapsed = time.time() - start_all
    log.info("=" * 70)
    log.info(
        f"DONE! Generated {generated_count} MIDI files in {total_elapsed / 60:.1f} min. "
        f"Manifest saved to: {manifest_path}"
    )
    log.info("=" * 70)


if __name__ == "__main__":
    main()

