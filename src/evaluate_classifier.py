"""
evaluate_classifier.py — Stage 4.2 Reverse Genre Classifier Evaluation

Evaluates generated MIDI files (or real dataset MIDI files) using the trained
6-layer RoBERTa genre classifier (Octuple tokenization).

Pipeline (matching implementation_plan.md & train_classifier.py):
  1. Discover generated MIDI files in:
       - $SCRATCH/generated_midis/<dataset>/<split>/<model>/<genre>/*.mid
       - or extract automatically from $SYNC_DIR/generated_midis_archives/<dataset>_<split>_<model>.tar(.gz)
  2. Preprocess each MIDI file identically to Stage 1 (tokenize_dataset.py):
       - Extract piano or multi-instrument tracks via symusic
       - Normalize key to C major / A minor via music21 (optional --no-transpose)
       - Tokenize into Octuple events (miditok.Octuple)
       - Flatten 2D Octuple tuples into 1D sequence, cap at 1023 tokens, prepend CLS token
  3. Run inference through the trained RobertaForSequenceClassification model for (<dataset>, <split>).
  4. Compute & save:
       - Overall Top-1 Accuracy, Top-3 Accuracy, Macro F1, Weighted F1, Mean Target Confidence
       - Mode-collapse diagnostics: Predicted Genre Distribution & KL/JS divergence
       - Per-genre accuracy/F1 breakdown (per_genre_metrics.csv)
       - Confusion matrix (confusion_matrix.csv)
       - Per-song predictions (predictions.csv)
       - Summary JSON (summary.json) & aggregated row in master_classifier_results.csv

Usage examples:
  # Evaluate a single model on Ares/Athena:
  python evaluate_classifier.py --dataset-name gt_1000_with_ties --split piano --model lstm

  # Evaluate all available models for a dataset & split:
  python evaluate_classifier.py --dataset-name gt_1000_with_ties --split piano --model all

  # Evaluate all datasets, splits, and models found on disk:
  python evaluate_classifier.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATASETS = [
    "gt_1000_with_ties",
    "gt_1000_no_ties",
    "top_5_with_ties",
    "top_5_no_ties",
]
SPLITS = ["piano", "multi-instrument"]
MODELS = ["real", "lstm", "transformer", "llama", "moonbeam"]
MAX_SEQ_LEN = 1024  # Matches OctuplePaddingCollator in train_classifier.py


# ============================================================================
# 1. Checkpoint & Tokenizer Resolution
# ============================================================================

def _is_valid_hf_dir(d: Path) -> bool:
    """Return True only if directory has non-empty config.json and weight files."""
    cfg = d / "config.json"
    if not (cfg.is_file() and cfg.stat().st_size > 0):
        return False
    weights = list(d.glob("*.safetensors")) + list(d.glob("*.bin"))
    if not weights:
        return False
    for w in weights:
        if w.stat().st_size == 0:
            return False
    return True


def find_classifier_checkpoint(
    dataset_name: str,
    split: str,
    sync_dir: Path,
    explicit_ckpt: str | Path | None = None,
) -> Path:
    """Locate the trained RoBERTa genre classifier checkpoint for (dataset_name, split)."""
    if explicit_ckpt is not None:
        p = Path(explicit_ckpt)
        if not p.exists():
            raise FileNotFoundError(f"Explicit classifier checkpoint not found: {p}")
        return p

    # 1. Check $MUSICGEN_MODELS_ROOT or $SCRATCH/models/<dataset>/classifier/<split>
    scratch_roots: list[Path] = []
    if os.getenv("MUSICGEN_MODELS_ROOT"):
        scratch_roots.append(Path(os.environ["MUSICGEN_MODELS_ROOT"]))
    if os.getenv("SCRATCH"):
        scratch_roots.append(Path(os.environ["SCRATCH"]) / "models")

    for s_root in scratch_roots:
        flat_dir = s_root / dataset_name / "classifier" / split
        if flat_dir.exists():
            if (flat_dir / "best_model").is_dir() and _is_valid_hf_dir(flat_dir / "best_model"):
                return flat_dir / "best_model"
            if _is_valid_hf_dir(flat_dir):
                return flat_dir
            ckpts = sorted(
                [d for d in flat_dir.glob("checkpoint-*") if d.is_dir() and _is_valid_hf_dir(d)],
                key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else -1,
            )
            if ckpts:
                return ckpts[-1]

    # 2. Fallback to $SYNC_DIR/artist_genres/<dataset>/models/classifier/<split>/run_*
    models_root = sync_dir / "artist_genres" / dataset_name / "models" / "classifier" / split
    if not models_root.exists():
        raise FileNotFoundError(
            f"Classifier directory not found in $SCRATCH/models/{dataset_name}/classifier/{split} "
            f"or {models_root}"
        )

    runs = sorted(
        [d for d in models_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
    )
    for run_dir in reversed(runs):
        if (run_dir / "best_model").is_dir() and _is_valid_hf_dir(run_dir / "best_model"):
            return run_dir / "best_model"
        if _is_valid_hf_dir(run_dir):
            return run_dir

    raise FileNotFoundError(f"No valid classifier checkpoint found for {dataset_name}/{split}")


def build_octuple_tokenizer(dataset_name: str, split: str, sync_dir: Path):
    """
    Load or construct the deterministic MidiTok Octuple tokenizer matching Stage 1
    (tokenize_dataset.py).
    """
    from miditok import Octuple, TokenizerConfig

    multi_instrument = split == "multi-instrument"

    # Check if saved tokenizer.json exists on disk or in dataset_backup
    candidates = [
        sync_dir / "artist_genres" / dataset_name / "dataset" / split / "octuple" / "tokenizer" / "tokenizer.json",
        sync_dir / "artist_genres" / dataset_name / "split" / split / "octuple_tokenizer.json",
    ]
    if os.getenv("SCRATCH"):
        candidates.append(
            Path(os.environ["SCRATCH"])
            / "dataset_backup"
            / dataset_name
            / "dataset"
            / split
            / "octuple"
            / "tokenizer"
            / "tokenizer.json"
        )

    for cand in candidates:
        if cand.is_file() and cand.stat().st_size > 0:
            log.info(f"Loading Octuple tokenizer from {cand}...")
            return Octuple(params=str(cand))

    # Deterministic construction identical to tokenize_dataset.py (Octuple has no BPE)
    log.info(f"Constructing deterministic Octuple tokenizer (multi_instrument={multi_instrument})...")
    octuple_config = TokenizerConfig(
        num_velocities=16,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=multi_instrument,
        one_token_stream_for_programs=True,
        beat_res={(0, 4): 8, (4, 12): 4},
        max_bar_embedding=1024,
    )
    return Octuple(octuple_config)


# ============================================================================
# 2. MIDI Preprocessing & Octuple Tokenization
# ============================================================================

def extract_midi_tracks(midi_path: Path, multi_instrument: bool = False):
    """
    Identical track extraction to tokenize_dataset.py:
      - piano: merge tracks with program 0..7, non-drum
      - multi-instrument: group non-drum tracks by GM family (program // 8)
    """
    import symusic

    score = symusic.Score(str(midi_path))
    if multi_instrument:
        valid_tracks = [t for t in score.tracks if not t.is_drum and len(t.notes) > 0]
        if not valid_tracks:
            return None

        family_notes: dict[int, list] = {}
        for t in valid_tracks:
            family = t.program // 8
            family_notes.setdefault(family, []).extend(t.notes)

        new_tracks = []
        for family, notes in family_notes.items():
            notes.sort(key=lambda n: (n.time, n.pitch))
            merged_track = symusic.Track()
            merged_track.program = family * 8
            merged_track.is_drum = False
            merged_track.notes.extend(notes)
            new_tracks.append(merged_track)

        new_tracks.sort(key=lambda t: t.program)
        score.tracks = new_tracks
        return score
    else:
        piano_tracks = [
            t for t in score.tracks if not t.is_drum and 0 <= t.program <= 7 and len(t.notes) > 0
        ]
        # Fallback: if generated MIDI didn't set program 0..7 explicitly, take all non-drum tracks
        if not piano_tracks:
            piano_tracks = [t for t in score.tracks if not t.is_drum and len(t.notes) > 0]
        if not piano_tracks:
            return None

        merged_notes = []
        for t in piano_tracks:
            merged_notes.extend(t.notes)
        merged_notes.sort(key=lambda n: (n.time, n.pitch))

        merged_track = symusic.Track()
        merged_track.program = 0
        merged_track.is_drum = False
        merged_track.notes.extend(merged_notes)
        score.tracks = [merged_track]
        return score


def transpose_to_c(score, midi_path: Path):
    """Normalize key to C major / A minor using music21 (matching tokenize_dataset.py)."""
    try:
        import music21
        import symusic

        analysis_score = symusic.Score(score.ticks_per_quarter)
        if score.tracks:
            best_track = max(score.tracks, key=lambda t: len(t.notes))
            new_t = symusic.Track()
            new_t.program = best_track.program
            new_t.is_drum = best_track.is_drum
            new_t.notes.extend(best_track.notes[:400])
            analysis_score.tracks.append(new_t)

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            tmp = Path(f.name)
        try:
            analysis_score.dump_midi(str(tmp))
            stream = music21.converter.parse(str(tmp))
            key = stream.analyze("key")
        finally:
            tmp.unlink(missing_ok=True)

        tonic = key.tonic.midi % 12
        semitones = (0 - tonic) % 12 if key.mode == "major" else (9 - tonic) % 12
        if semitones != 0:
            for track in score.tracks:
                for note in track.notes:
                    note.pitch = max(0, min(127, note.pitch + semitones))
        return score
    except Exception:
        return score


def midi_to_octuple_flat(
    midi_path: Path,
    octuple_tok,
    multi_instrument: bool,
    normalize_key: bool = True,
) -> list[int] | None:
    """
    Convert a single MIDI file into a flattened 1D Octuple token list
    (up to MAX_SEQ_LEN - 1 = 1023 tokens), ready for OctuplePaddingCollator.
    """
    try:
        score = extract_midi_tracks(midi_path, multi_instrument=multi_instrument)
        if score is None:
            return None
        total_notes = sum(len(t.notes) for t in score.tracks)
        if total_notes == 0:
            return None
        if normalize_key:
            score = transpose_to_c(score, midi_path)

        octuple_res = octuple_tok(score)
        if isinstance(octuple_res, list):
            if len(octuple_res) == 1:
                octuple_tokens = octuple_res[0].ids
            else:
                octuple_tokens = [note for track_seq in octuple_res for note in track_seq.ids]
        else:
            octuple_tokens = octuple_res.ids
        if not octuple_tokens:
            return None

        flat = (
            [t for note in octuple_tokens for t in note]
            if isinstance(octuple_tokens[0], (list, tuple))
            else list(octuple_tokens)
        )
        if not flat:
            return None
        return flat[: MAX_SEQ_LEN - 1]
    except Exception as e:
        log.warning(f"Failed to tokenize {midi_path}: {e}")
        return None


# ============================================================================
# 3. Discovering Generated Songs (Scratch Dir or Archive Fallback)
# ============================================================================

def resolve_generated_model_dir(
    dataset_name: str,
    split: str,
    model_name: str,
    sync_dir: Path,
    generated_root: Path | None = None,
) -> tuple[Path | None, tempfile.TemporaryDirectory | None]:
    """
    Find the directory containing generated .mid files for (dataset_name, split, model_name).
    Checks:
      1. Explicit generated_root/<dataset>/<split>/<model>
      2. $SCRATCH/generated_midis/<dataset>/<split>/<model>
      3. $SYNC_DIR/generated_midis/<dataset>/<split>/<model>
      4. Unpacked folder or .tar/.tar.gz archive in $SYNC_DIR/generated_midis_archives/
    Returns (model_dir, temp_dir_handle_if_extracted).
    """
    candidates: list[Path] = []
    if generated_root is not None:
        candidates.append(generated_root / dataset_name / split / model_name)
    if os.getenv("SCRATCH"):
        candidates.append(Path(os.environ["SCRATCH"]) / "generated_midis" / dataset_name / split / model_name)
    candidates.append(sync_dir / "generated_midis" / dataset_name / split / model_name)

    # Also check already-unpacked folders in sample_generated_midis or generated_midis_archives
    repo_root = Path(__file__).resolve().parent.parent
    archive_bases = [
        sync_dir / "sample_generated_midis",
        repo_root / "sample_generated_midis",
        sync_dir / "generated_midis_archives",
    ]
    stem = f"{dataset_name}_{split}_{model_name}"
    for ab in archive_bases:
        candidates.append(ab / stem / model_name)
        candidates.append(ab / stem)

    for cand in candidates:
        if cand.is_dir() and any(cand.rglob("*.mid")):
            return cand, None

    # Check if a .tar.gz or .tar archive exists and extract to a temporary directory
    for ab in archive_bases:
        for ext in (".tar.gz", ".tar"):
            arch = ab / f"{stem}{ext}"
            if arch.is_file() and arch.stat().st_size > 0:
                log.info(f"Extracting archive {arch} to temporary directory for evaluation...")
                tmp_dir = tempfile.TemporaryDirectory(prefix=f"eval_cls_{stem}_")
                tmp_path = Path(tmp_dir.name)
                with tarfile.open(arch, "r:*") as tar:
                    tar.extractall(path=tmp_path)
                sub = tmp_path / model_name
                if sub.is_dir() and any(sub.rglob("*.mid")):
                    return sub, tmp_dir
                if any(tmp_path.rglob("*.mid")):
                    return tmp_path, tmp_dir
                tmp_dir.cleanup()

    return None, None


def collect_midi_items(model_dir: Path) -> list[dict[str, Any]]:
    """
    Collect all .mid files under model_dir.
    Uses manifest.csv if present; otherwise infers target genre from parent directory name.
    """
    items: list[dict[str, Any]] = []
    manifest_path = model_dir / "manifest.csv"

    if manifest_path.is_file() and manifest_path.stat().st_size > 0:
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row.get("file_path", "")
                fpath = model_dir / rel_path
                if fpath.is_file() and fpath.stat().st_size > 0:
                    items.append({
                        "song_id": row.get("song_id") or fpath.stem,
                        "file_path": fpath,
                        "rel_path": rel_path,
                        "target_genre": row.get("genre", fpath.parent.name),
                    })
        if items:
            return items

    # Fallback: glob all *.mid files
    for fpath in sorted(model_dir.rglob("*.mid")):
        if fpath.stat().st_size > 0:
            target_genre = fpath.parent.name if fpath.parent != model_dir else "unconditional"
            items.append({
                "song_id": fpath.stem,
                "file_path": fpath,
                "rel_path": str(fpath.relative_to(model_dir)),
                "target_genre": target_genre,
            })
    return items


# ============================================================================
# 4. Classifier Evaluation Core
# ============================================================================

def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    probs: np.ndarray,
    genres: list[str],
) -> dict[str, Any]:
    """
    Compute Top-1, Top-3, Macro/Weighted Precision/Recall/F1, Mean Target Probability,
    Predicted Genre Distribution, KL/JS divergence, and Confusion Matrix.
    """
    num_classes = len(genres)
    n = len(y_pred)
    if n == 0:
        return {}

    # Predicted distribution across all generated samples (even unconditional)
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    pred_dist = pred_counts / max(n, 1)
    uniform_dist = np.full(num_classes, 1.0 / num_classes)

    # KL(pred || uniform) and Jensen-Shannon divergence vs uniform
    eps = 1e-12
    kl_to_uniform = float(np.sum(pred_dist * np.log((pred_dist + eps) / (uniform_dist + eps))))
    m_dist = 0.5 * (pred_dist + uniform_dist)
    js_to_uniform = float(
        0.5 * np.sum(pred_dist * np.log((pred_dist + eps) / (m_dist + eps)))
        + 0.5 * np.sum(uniform_dist * np.log((uniform_dist + eps) / (m_dist + eps)))
    )

    # Prediction confidence & entropy per sample
    sample_entropies = -np.sum(probs * np.log(probs + eps), axis=-1)
    mean_pred_confidence = float(np.mean(np.max(probs, axis=-1)))
    mean_pred_entropy = float(np.mean(sample_entropies))

    # Check if this is a genre-conditioned run (valid y_true >= 0)
    valid_mask = [t >= 0 for t in y_true]
    has_ground_truth = any(valid_mask)

    if not has_ground_truth:
        return {
            "conditioned": False,
            "num_evaluated": n,
            "top1_accuracy": None,
            "top3_accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "mean_target_prob": None,
            "mean_pred_confidence": mean_pred_confidence,
            "mean_pred_entropy": mean_pred_entropy,
            "kl_div_to_uniform": kl_to_uniform,
            "js_div_to_uniform": js_to_uniform,
            "predicted_distribution": {genres[i]: float(pred_dist[i]) for i in range(num_classes)},
            "predicted_counts": {genres[i]: int(pred_counts[i]) for i in range(num_classes)},
            "per_genre": {},
            "confusion_matrix": None,
        }

    yt = np.array([y_true[i] for i in range(n) if valid_mask[i]], dtype=np.int64)
    yp = np.array([y_pred[i] for i in range(n) if valid_mask[i]], dtype=np.int64)
    pr = probs[valid_mask]
    n_cond = len(yt)

    # Top-1 and Top-3 accuracy
    top1_acc = float(np.mean(yt == yp))
    k = min(3, num_classes)
    topk_preds = np.argsort(-pr, axis=-1)[:, :k]
    top3_acc = float(np.mean([yt[i] in topk_preds[i] for i in range(n_cond)]))

    # Mean probability assigned to the requested target genre
    target_probs = pr[np.arange(n_cond), yt]
    mean_target_prob = float(np.mean(target_probs))

    # Confusion matrix: rows = true genre, cols = predicted genre
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t_idx, p_idx in zip(yt, yp):
        cm[t_idx, p_idx] += 1

    # Per-genre precision, recall, F1, accuracy, mean_target_prob
    per_genre: dict[str, dict[str, Any]] = {}
    f1s, precs, recs, supports = [], [], [], []

    for c in range(num_classes):
        tp = int(cm[c, c])
        support = int(cm[c, :].sum())
        pred_pos = int(cm[:, c].sum())

        prec = tp / pred_pos if pred_pos > 0 else 0.0
        rec = tp / support if support > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        c_mask = yt == c
        c_top3 = (
            float(np.mean([yt[i] in topk_preds[i] for i in np.where(c_mask)[0]]))
            if support > 0
            else 0.0
        )
        c_target_prob = float(np.mean(pr[c_mask, c])) if support > 0 else 0.0

        per_genre[genres[c]] = {
            "genre": genres[c],
            "genre_id": c,
            "support": support,
            "predicted_count": pred_pos,
            "top1_accuracy": rec,
            "top3_accuracy": c_top3,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mean_target_prob": c_target_prob,
        }

        if support > 0:
            f1s.append(f1)
            precs.append(prec)
            recs.append(rec)
            supports.append(support)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    macro_prec = float(np.mean(precs)) if precs else 0.0
    macro_rec = float(np.mean(recs)) if recs else 0.0
    weighted_f1 = (
        float(np.average(f1s, weights=supports)) if supports and sum(supports) > 0 else 0.0
    )

    return {
        "conditioned": True,
        "num_evaluated": n_cond,
        "num_classes": num_classes,
        "random_baseline_top1": 1.0 / num_classes,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "mean_target_prob": mean_target_prob,
        "mean_pred_confidence": mean_pred_confidence,
        "mean_pred_entropy": mean_pred_entropy,
        "kl_div_to_uniform": kl_to_uniform,
        "js_div_to_uniform": js_to_uniform,
        "predicted_distribution": {genres[i]: float(pred_dist[i]) for i in range(num_classes)},
        "predicted_counts": {genres[i]: int(pred_counts[i]) for i in range(num_classes)},
        "per_genre": per_genre,
        "confusion_matrix": cm.tolist(),
    }


def resolve_octuple_test_dir(
    dataset_name: str,
    split: str,
    sync_dir: Path,
    explicit_dir: Path | str | None = None,
) -> Path | None:
    """Locate pre-tokenized Octuple/test Arrow dataset on disk or cluster scratch."""
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    candidates.append(Path.cwd() / "dataset_backup" / dataset_name / split / "octuple" / "test")
    candidates.append(sync_dir / "dataset_backup" / dataset_name / split / "octuple" / "test")
    candidates.append(sync_dir.parent / "dataset_backup" / dataset_name / split / "octuple" / "test")
    if os.getenv("SCRATCH"):
        sc = Path(os.environ["SCRATCH"])
        candidates.extend([
            sc / "dataset_backup" / dataset_name / split / "octuple" / "test",
            sc / "dataset_backup" / dataset_name / "dataset" / split / "octuple" / "test",
            sc / "octuple" / dataset_name / split / "test",
        ])
    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend([
        repo_root / "datasets_metadata" / dataset_name / "dataset" / split / "octuple" / "test",
        repo_root / "datasets_metadata" / dataset_name / split / "octuple" / "test",
        sync_dir / "datasets_metadata" / dataset_name / "dataset" / split / "octuple" / "test",
        sync_dir / "datasets_metadata" / dataset_name / split / "octuple" / "test",
        sync_dir / "artist_genres" / dataset_name / "dataset" / split / "octuple" / "test",
        sync_dir / "artist_genres" / dataset_name / split / "octuple" / "test",
    ])
    if os.getenv("PLG_GROUPS_STORAGE"):
        pg = Path(os.environ["PLG_GROUPS_STORAGE"])
        plg_proj = os.getenv("PLG_PROJECT_NAME", "music_gen")
        candidates.extend([
            pg / plg_proj / "datasets_metadata" / dataset_name / "dataset" / split / "octuple" / "test",
            pg / plg_proj / "datasets_metadata" / dataset_name / split / "octuple" / "test",
        ])
    for c in candidates:
        if c.is_dir() and ((c / "dataset_info.json").is_file() or (c / "state.json").is_file() or any(c.glob("*.arrow"))):
            return c
    return None


def evaluate_real_from_tokenized_dataset(
    dataset_name: str,
    split: str,
    octuple_test_dir: Path,
    classifier_model: RobertaForSequenceClassification,
    device: torch.device,
    out_base_dir: Path,
    batch_size: int = 32,
    max_samples_per_genre: int = 100,
) -> dict[str, Any]:
    """Directly evaluate the classifier on the pre-tokenized Arrow test split (zero re-tokenization)."""
    from datasets import load_from_disk

    log.info("=" * 70)
    log.info(f"Evaluating Classifier on Pre-Tokenized Test Split | dataset={dataset_name} | split={split}")
    log.info(f"Octuple Test Path: {octuple_test_dir}")
    log.info("=" * 70)

    ds = load_from_disk(str(octuple_test_dir))
    log.info(f"Loaded {len(ds):,} test samples directly from Arrow dataset.")

    pad_token_id = classifier_model.config.pad_token_id
    cls_token_id = classifier_model.config.vocab_size - 1

    id2genre = {int(k): v for k, v in classifier_model.config.id2label.items()}
    genres = [id2genre[i] for i in range(len(id2genre))]
    genre2id = {g: i for i, g in id2genre.items()}

    # Group by genre and select up to max_samples_per_genre (or evaluate all if <= 0)
    indices_to_eval = list(range(len(ds)))
    if max_samples_per_genre and max_samples_per_genre > 0:
        by_gid = defaultdict(list)
        for i, ex in enumerate(ds):
            gid = ex.get("label", ex.get("genre_id"))
            if gid is None and "genre" in ex:
                gid = genre2id.get(ex["genre"], -1)
            by_gid[int(gid) if gid is not None else -1].append(i)

        rng = random.Random(42)
        selected_set = set()
        for gid, idxs in sorted(by_gid.items()):
            n_sample = min(max_samples_per_genre, len(idxs))
            for idx in rng.sample(idxs, n_sample):
                selected_set.add(idx)
        indices_to_eval = sorted(selected_set)
        log.info(f"Subsampled {len(indices_to_eval)} test tracks ({max_samples_per_genre}/genre) for direct comparison with generative models.")

    flat_seqs: list[list[int]] = []
    y_true: list[int] = []
    valid_items: list[dict[str, Any]] = []

    for idx in indices_to_eval:
        ex = ds[idx]
        ids = ex["input_ids"]
        flat = [t for note in ids for t in note] if (ids and isinstance(ids[0], (list, tuple))) else list(ids)
        if not flat:
            continue
        flat_trunc = flat[: MAX_SEQ_LEN - 1]
        gid = ex.get("label", ex.get("genre_id"))
        if gid is None and "genre" in ex:
            gid = genre2id.get(ex["genre"], -1)
        gid = int(gid) if gid is not None else -1
        gname = id2genre.get(gid, ex.get("genre", f"genre_{gid}"))

        flat_seqs.append(flat_trunc)
        y_true.append(gid)
        valid_items.append({
            "song_id": ex.get("song_id") or f"real_test_{idx:05d}",
            "rel_path": ex.get("file_path") or f"{gname}/track_{idx:05d}",
            "target_genre": gname,
            "target_genre_id": gid,
        })

    log.info(f"Running batched RoBERTa inference over {len(flat_seqs)} pre-tokenized test samples...")
    classifier_model.eval()
    all_probs_list = []

    with torch.no_grad():
        for b_start in range(0, len(flat_seqs), batch_size):
            batch_flats = flat_seqs[b_start : b_start + batch_size]
            max_len = min(max(len(f) for f in batch_flats), MAX_SEQ_LEN - 1)

            padded_ids, attn_mask = [], []
            for flat in batch_flats:
                pad_len = max_len - len(flat)
                padded_ids.append([cls_token_id] + flat + [pad_token_id] * pad_len)
                attn_mask.append([1] + [1] * len(flat) + [0] * pad_len)

            input_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
            attn_mask_t = torch.tensor(attn_mask, dtype=torch.long, device=device)

            outputs = classifier_model(input_ids=input_ids_t, attention_mask=attn_mask_t)
            probs_b = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs_list.append(probs_b)

    probs = np.vstack(all_probs_list)
    y_pred = np.argmax(probs, axis=-1).tolist()

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        genres=genres,
    )
    metrics.update({
        "dataset": dataset_name,
        "split": split,
        "model": "real",
        "total_files": len(indices_to_eval),
        "valid_files": len(valid_items),
        "failed_files": len(indices_to_eval) - len(valid_items),
    })

    out_dir = out_base_dir / dataset_name / split / "real"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. summary.json
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 2. per_genre_metrics.csv
    if metrics.get("per_genre"):
        pg_path = out_dir / "per_genre_metrics.csv"
        pg_fields = [
            "genre", "genre_id", "support", "predicted_count",
            "top1_accuracy", "top3_accuracy", "precision", "recall", "f1", "mean_target_prob",
        ]
        with open(pg_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pg_fields)
            writer.writeheader()
            for g in genres:
                writer.writerow(metrics["per_genre"][g])

    # 3. confusion_matrix.csv
    if metrics.get("confusion_matrix") is not None:
        cm_path = out_dir / "confusion_matrix.csv"
        cm = metrics["confusion_matrix"]
        with open(cm_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true_genre \\ pred_genre"] + genres)
            for g_idx, g_name in enumerate(genres):
                writer.writerow([g_name] + cm[g_idx])

    # 4. predictions.csv
    pred_path = out_dir / "predictions.csv"
    k_top = min(3, len(genres))
    topk_indices = np.argsort(-probs, axis=-1)[:, :k_top]

    pred_fields = [
        "song_id", "rel_path", "target_genre", "target_genre_id",
        "predicted_genre", "predicted_genre_id",
        "correct_top1", "correct_top3",
        "target_prob", "predicted_prob",
    ] + [f"prob_{g}" for g in genres]

    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        for i, it in enumerate(valid_items):
            t_id = y_true[i]
            p_id = y_pred[i]
            row_dict = {
                "song_id": it["song_id"],
                "rel_path": it["rel_path"],
                "target_genre": it["target_genre"],
                "target_genre_id": t_id,
                "predicted_genre": id2genre[p_id],
                "predicted_genre_id": p_id,
                "correct_top1": int(t_id == p_id) if t_id >= 0 else "",
                "correct_top3": int(t_id in topk_indices[i]) if t_id >= 0 else "",
                "target_prob": round(float(probs[i, t_id]), 6) if t_id >= 0 else "",
                "predicted_prob": round(float(probs[i, p_id]), 6),
            }
            for g_idx, g_name in enumerate(genres):
                row_dict[f"prob_{g_name}"] = round(float(probs[i, g_idx]), 6)
            writer.writerow(row_dict)

    log.info(
        f"RESULT [{dataset_name} | {split} | real] "
        f"Top-1 Acc: {metrics['top1_accuracy']*100:.2f}% | "
        f"Top-3 Acc: {metrics['top3_accuracy']*100:.2f}% | "
        f"Macro F1: {metrics['macro_f1']:.4f} | "
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
    )
    return metrics


def evaluate_single_model(
    dataset_name: str,
    split: str,
    model_name: str,
    classifier_model: RobertaForSequenceClassification,
    octuple_tok,
    device: torch.device,
    sync_dir: Path,
    out_base_dir: Path,
    generated_root: Path | None = None,
    octuple_test_dir: Path | None = None,
    batch_size: int = 32,
    normalize_key: bool = True,
    songs_per_genre: int = 100,
) -> dict[str, Any] | None:
    """Classify generated MIDIs (or evaluate pre-tokenized real test split) for a single (dataset, split, model)."""
    if model_name == "real":
        resolved_oct_dir = resolve_octuple_test_dir(dataset_name, split, sync_dir, explicit_dir=octuple_test_dir)
        if resolved_oct_dir is None:
            log.error(
                f"[SKIP] Pre-tokenized Octuple test set not found for {dataset_name} | {split}. "
                f"Ensure it exists at $SCRATCH/dataset_backup/{dataset_name}/{split}/octuple/test"
            )
            return None
        return evaluate_real_from_tokenized_dataset(
            dataset_name=dataset_name,
            split=split,
            octuple_test_dir=resolved_oct_dir,
            classifier_model=classifier_model,
            device=device,
            out_base_dir=out_base_dir,
            batch_size=batch_size,
            max_samples_per_genre=songs_per_genre,
        )

    model_dir, tmp_handle = resolve_generated_model_dir(
        dataset_name=dataset_name,
        split=split,
        model_name=model_name,
        sync_dir=sync_dir,
        generated_root=generated_root,
    )
    if model_dir is None:
        log.warning(f"[SKIP] No generated MIDIs found for {dataset_name} | {split} | {model_name}")
        return None
    items = collect_midi_items(model_dir)
    if not items:
        log.warning(f"[SKIP] 0 MIDI files in {model_dir}")
        return None

    try:
        log.info("=" * 70)
        log.info(
            f"Evaluating Classifier | dataset={dataset_name} | split={split} | "
            f"model={model_name} | files={len(items)}"
        )
        log.info(f"Source directory: {model_dir}")
        log.info("=" * 70)

        multi_instrument = split == "multi-instrument"
        pad_token_id = classifier_model.config.pad_token_id
        cls_token_id = classifier_model.config.vocab_size - 1

        # Build genre mappings from classifier config
        id2genre = {int(k): v for k, v in classifier_model.config.id2label.items()}
        genres = [id2genre[i] for i in range(len(id2genre))]
        genre2id = {g: i for i, g in id2genre.items()}

        valid_items = []
        flat_seqs: list[list[int]] = []
        failed_count = 0

        for idx, item in enumerate(items):
            flat = midi_to_octuple_flat(
                item["file_path"],
                octuple_tok=octuple_tok,
                multi_instrument=multi_instrument,
                normalize_key=normalize_key,
            )
            if flat is None or len(flat) == 0:
                failed_count += 1
                continue
            valid_items.append(item)
            flat_seqs.append(flat)

            if (idx + 1) % 100 == 0 or (idx + 1) == len(items):
                log.info(f"  Tokenized {idx + 1}/{len(items)} MIDI files (valid={len(valid_items)}, failed={failed_count})")

        if not valid_items:
            log.error(f"All {len(items)} MIDI files failed Octuple tokenization for {model_name}!")
            return None

        # Batched forward pass through RoBERTa classifier
        all_probs_list = []
        classifier_model.eval()

        with torch.no_grad():
            for b_start in range(0, len(flat_seqs), batch_size):
                batch_flats = flat_seqs[b_start : b_start + batch_size]
                max_len = min(max(len(f) for f in batch_flats), MAX_SEQ_LEN - 1)

                padded_ids, attn_mask = [], []
                for flat in batch_flats:
                    flat_trunc = flat[:max_len]
                    pad_len = max_len - len(flat_trunc)
                    padded_ids.append([cls_token_id] + flat_trunc + [pad_token_id] * pad_len)
                    attn_mask.append([1] + [1] * len(flat_trunc) + [0] * pad_len)

                input_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
                attn_mask_t = torch.tensor(attn_mask, dtype=torch.long, device=device)

                outputs = classifier_model(input_ids=input_ids_t, attention_mask=attn_mask_t)
                probs_b = F.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs_list.append(probs_b)

        probs = np.vstack(all_probs_list)
        y_pred = np.argmax(probs, axis=-1).tolist()
        y_true = [genre2id.get(it["target_genre"], -1) for it in valid_items]

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probs=probs,
            genres=genres,
        )
        metrics.update({
            "dataset": dataset_name,
            "split": split,
            "model": model_name,
            "total_files": len(items),
            "valid_files": len(valid_items),
            "failed_files": failed_count,
        })

        # Save outputs to out_base_dir/<dataset>/<split>/<model>/
        out_dir = out_base_dir / dataset_name / split / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. summary.json
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # 2. per_genre_metrics.csv
        if metrics.get("per_genre"):
            pg_path = out_dir / "per_genre_metrics.csv"
            pg_fields = [
                "genre", "genre_id", "support", "predicted_count",
                "top1_accuracy", "top3_accuracy", "precision", "recall", "f1", "mean_target_prob",
            ]
            with open(pg_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=pg_fields)
                writer.writeheader()
                for g in genres:
                    writer.writerow(metrics["per_genre"][g])

        # 3. confusion_matrix.csv
        if metrics.get("confusion_matrix") is not None:
            cm_path = out_dir / "confusion_matrix.csv"
            cm = metrics["confusion_matrix"]
            with open(cm_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["true_genre \\ pred_genre"] + genres)
                for g_idx, g_name in enumerate(genres):
                    writer.writerow([g_name] + cm[g_idx])

        # 4. predictions.csv (per-song details)
        pred_path = out_dir / "predictions.csv"
        k_top = min(3, len(genres))
        topk_indices = np.argsort(-probs, axis=-1)[:, :k_top]

        pred_fields = [
            "song_id", "rel_path", "target_genre", "target_genre_id",
            "predicted_genre", "predicted_genre_id",
            "correct_top1", "correct_top3",
            "target_prob", "predicted_prob",
        ] + [f"prob_{g}" for g in genres]

        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pred_fields)
            writer.writeheader()
            for i, it in enumerate(valid_items):
                t_id = y_true[i]
                p_id = y_pred[i]
                row_dict = {
                    "song_id": it["song_id"],
                    "rel_path": it["rel_path"],
                    "target_genre": it["target_genre"],
                    "target_genre_id": t_id,
                    "predicted_genre": id2genre[p_id],
                    "predicted_genre_id": p_id,
                    "correct_top1": int(t_id == p_id) if t_id >= 0 else "",
                    "correct_top3": int(t_id in topk_indices[i]) if t_id >= 0 else "",
                    "target_prob": round(float(probs[i, t_id]), 6) if t_id >= 0 else "",
                    "predicted_prob": round(float(probs[i, p_id]), 6),
                }
                for g_idx, g_name in enumerate(genres):
                    row_dict[f"prob_{g_name}"] = round(float(probs[i, g_idx]), 6)
                writer.writerow(row_dict)

        if metrics["conditioned"]:
            log.info(
                f"RESULT [{dataset_name} | {split} | {model_name}] "
                f"Top-1 Acc: {metrics['top1_accuracy']*100:.2f}% | "
                f"Top-3 Acc: {metrics['top3_accuracy']*100:.2f}% | "
                f"Macro F1: {metrics['macro_f1']:.4f} | "
                f"Weighted F1: {metrics['weighted_f1']:.4f} | "
                f"Target Prob: {metrics['mean_target_prob']:.4f}"
            )
        else:
            log.info(
                f"RESULT [{dataset_name} | {split} | {model_name} (unconditional)] "
                f"Mean Confidence: {metrics['mean_pred_confidence']:.4f} | "
                f"Predicted Dist: {metrics['predicted_counts']}"
            )
        return metrics

    finally:
        if tmp_handle is not None:
            tmp_handle.cleanup()


# ============================================================================
# 5. CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated MIDI files with the trained RoBERTa (Octuple) genre classifier."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="gt_1000_with_ties",
        choices=DATASETS + ["all"],
        help="Dataset variant to evaluate (or 'all').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="piano",
        choices=SPLITS + ["all"],
        help="Instrument split ('piano', 'multi-instrument', or 'all').",
    )
    parser.add_argument(
        "--multi-instrument",
        action="store_true",
        help="Shorthand for --split multi-instrument.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=MODELS + ["all"],
        help="Generative model to evaluate ('lstm', 'transformer', 'llama', 'moonbeam', or 'all').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all datasets, all splits, and all models found on disk.",
    )
    parser.add_argument(
        "--generated-root",
        type=str,
        default=None,
        help="Root directory of generated MIDIs (defaults to $SCRATCH/generated_midis or $SYNC_DIR/generated_midis_archives).",
    )
    parser.add_argument(
        "--classifier-ckpt",
        type=str,
        default=None,
        help="Explicit path to a classifier checkpoint (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (defaults to $SYNC_DIR/evaluation_results/classifier).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for RoBERTa classifier inference.",
    )
    parser.add_argument(
        "--no-transpose",
        action="store_true",
        help="Skip music21 C-major/A-minor key normalization before Octuple tokenization (recommended if generated pieces were already produced in normalized key by trained models).",
    )
    parser.add_argument(
        "--include-real",
        action="store_true",
        help="Evaluate the real test.csv dataset baseline first when running with --all (default: False).",
    )
    parser.add_argument(
        "--no-real",
        action="store_true",
        help="Skip evaluating the real dataset baseline.",
    )
    parser.add_argument(
        "--real-songs-per-genre",
        type=int,
        default=100,
        help="Number of real test songs per genre to sample for baseline evaluation (default: 100).",
    )
    parser.add_argument(
        "--octuple-test-dir",
        type=str,
        default=None,
        help="Explicit directory of pre-tokenized Octuple test Arrow dataset (auto-detected in $SCRATCH/dataset_backup/<dataset>/<split>/octuple/test if omitted).",
    )

    args = parser.parse_args()
    if args.multi_instrument:
        args.split = "multi-instrument"

    repo_root = Path(__file__).resolve().parent.parent
    sync_dir = Path(os.getenv("MUSICGEN_SYNC_DIR", repo_root))
    out_base_dir = Path(args.output_dir) if args.output_dir else (repo_root / "evaluation_results" / "classifier")
    out_base_dir.mkdir(parents=True, exist_ok=True)
    generated_root = Path(args.generated_root) if args.generated_root else None
    oct_test_dir = Path(args.octuple_test_dir) if args.octuple_test_dir else None

    if not torch.cuda.is_available():
        num_threads = int(
            os.environ.get("OMP_NUM_THREADS")
            or os.environ.get("SLURM_CPUS_PER_TASK")
            or (os.cpu_count() or 8)
        )
        torch.set_num_threads(num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Running Classifier Evaluation on device: {device}")

    target_datasets = DATASETS if (args.all or args.dataset_name == "all") else [args.dataset_name]
    target_splits = SPLITS if (args.all or args.split == "all") else [args.split]
    if args.model != "all":
        target_models = [args.model]
    else:
        target_models = (
            ["real", "lstm", "transformer", "llama", "moonbeam"]
            if (args.include_real and not args.no_real)
            else ["lstm", "transformer", "llama", "moonbeam"]
        )

    master_rows: list[dict[str, Any]] = []

    for ds in target_datasets:
        for sp in target_splits:
            try:
                ckpt_path = find_classifier_checkpoint(
                    dataset_name=ds,
                    split=sp,
                    sync_dir=sync_dir,
                    explicit_ckpt=args.classifier_ckpt,
                )
            except FileNotFoundError as e:
                log.warning(f"[SKIP] {e}")
                continue

            log.info(f"Loading Classifier [{ds} | {sp}] from {ckpt_path}...")
            classifier_model = RobertaForSequenceClassification.from_pretrained(str(ckpt_path))
            classifier_model.to(device)
            classifier_model.eval()

            octuple_tok = build_octuple_tokenizer(ds, sp, sync_dir)
            if len(octuple_tok) + 1 != classifier_model.config.vocab_size:
                log.warning(
                    f"Octuple tokenizer len+1 ({len(octuple_tok) + 1}) != classifier vocab_size "
                    f"({classifier_model.config.vocab_size}). Using classifier config dimensions."
                )

            for m_name in target_models:
                res = evaluate_single_model(
                    dataset_name=ds,
                    split=sp,
                    model_name=m_name,
                    classifier_model=classifier_model,
                    octuple_tok=octuple_tok,
                    device=device,
                    sync_dir=sync_dir,
                    out_base_dir=out_base_dir,
                    generated_root=generated_root,
                    octuple_test_dir=oct_test_dir,
                    batch_size=args.batch_size,
                    normalize_key=not args.no_transpose,
                    songs_per_genre=args.real_songs_per_genre,
                )
                if res is not None:
                    master_rows.append({
                        "dataset": ds,
                        "split": sp,
                        "model": m_name,
                        "conditioned": res["conditioned"],
                        "total_files": res["total_files"],
                        "valid_files": res["valid_files"],
                        "failed_files": res["failed_files"],
                        "top1_accuracy": round(res["top1_accuracy"], 6) if res["top1_accuracy"] is not None else "",
                        "top3_accuracy": round(res["top3_accuracy"], 6) if res["top3_accuracy"] is not None else "",
                        "macro_f1": round(res["macro_f1"], 6) if res["macro_f1"] is not None else "",
                        "weighted_f1": round(res["weighted_f1"], 6) if res["weighted_f1"] is not None else "",
                        "macro_precision": round(res["macro_precision"], 6) if res["macro_precision"] is not None else "",
                        "macro_recall": round(res["macro_recall"], 6) if res["macro_recall"] is not None else "",
                        "mean_target_prob": round(res["mean_target_prob"], 6) if res["mean_target_prob"] is not None else "",
                        "mean_pred_confidence": round(res["mean_pred_confidence"], 6),
                        "mean_pred_entropy": round(res["mean_pred_entropy"], 6),
                        "kl_div_to_uniform": round(res["kl_div_to_uniform"], 6),
                        "js_div_to_uniform": round(res["js_div_to_uniform"], 6),
                    })

    if master_rows:
        master_csv = out_base_dir / "master_classifier_results.csv"
        # Merge with existing master_classifier_results.csv if present so incremental runs update cleanly
        existing_map: dict[tuple[str, str, str], dict[str, Any]] = {}
        if master_csv.is_file() and master_csv.stat().st_size > 0:
            with open(master_csv, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    key = (row["dataset"], row["split"], row["model"])
                    existing_map[key] = row

        for row in master_rows:
            key = (row["dataset"], row["split"], row["model"])
            existing_map[key] = row

        merged_rows = sorted(
            existing_map.values(),
            key=lambda r: (r["dataset"], r["split"], r["model"]),
        )
        fieldnames = list(master_rows[0].keys())
        with open(master_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)

        log.info(f"Saved aggregated classifier summary ({len(merged_rows)} rows) to {master_csv}")
    else:
        log.warning("No models were evaluated.")


if __name__ == "__main__":
    main()

