"""
evaluate_metrics.py — Stage 4.2 Objective Musical Metrics & Real Dataset Baseline

Computes all objective symbolic music evaluation metrics specified in
implementation_plan.md (Stage 4.2) for both:
  - Generated MIDI files (lstm, transformer, llama, moonbeam)
  - Real Dataset Baseline ('real' — sampled from test.csv / test split)

Metrics computed per song, per genre, and per model:
  1. Pitch Class Entropy (PCE) [bits] — tonal diversity across the 12 pitch classes
  2. Scale Consistency (SC) [0..1] — best-fitting major/minor scale in-scale note ratio
  3. C-Major / A-Minor Scale Consistency [0..1] — adherence to normalized key
  4. Polyphony Rate (PR) — average simultaneous notes across active time steps
  5. Polyphonic Ratio [0..1] — fraction of active steps with >= 2 simultaneous notes
  6. Groove Consistency (GC) [0..1] — bar-to-bar rhythmic onset pattern Cosine Similarity (Thesis Eq. 3.12)
  7. Inter-Onset Interval (IOI) Statistics — mean, std, coefficient of variation (CV),
     entropy, and 13-bin musical IOI histogram (with JS divergence vs. Real Baseline)
  8. Structural Statistics — note count, duration (s), note density (notes/s),
     pitch range (semitones), unique pitches, mean note duration (s), active instruments
  9. Data Memorization & Repetition:
     - Self-repetition rate (4-gram & 8-gram internal loop ratio)
     - Training Set Memorization (Max LCS ratio & 8-gram overlap vs. train.csv reference)
 10. Statistical Comparison vs. Real Baseline (Mann-Whitney U / Wilcoxon p-value &
     relative difference when 'real' baseline is available for the same dataset & split).

Usage examples:
  # Evaluate real dataset baseline + all generated models for a dataset & split:
  python evaluate_metrics.py --dataset-name gt_1000_with_ties --split piano --model all --include-real

  # Evaluate everything across all datasets and splits:
  python evaluate_metrics.py --all --include-real
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

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

# Scale pitch-class templates for scale consistency analysis
MAJOR_TEMPLATE = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR_TEMPLATE = (0, 2, 3, 5, 7, 8, 10)
HARMONIC_MINOR_TEMPLATE = (0, 2, 3, 5, 7, 8, 11)  # Leading tone (e.g. G# in A minor)
ALL_24_SCALES: list[tuple[int, str, set[int]]] = []
for root_pc in range(12):
    ALL_24_SCALES.append((root_pc, "major", {(root_pc + iv) % 12 for iv in MAJOR_TEMPLATE}))
    ALL_24_SCALES.append((root_pc, "minor", {(root_pc + iv) % 12 for iv in NATURAL_MINOR_TEMPLATE}))
    ALL_24_SCALES.append((root_pc, "harmonic_minor", {(root_pc + iv) % 12 for iv in HARMONIC_MINOR_TEMPLATE}))
# C Major / A Minor reference pitch class set (includes diatonic C-maj/A-nat-min and G# from A-harm-min)
C_AM_SCALE = {0, 2, 4, 5, 7, 8, 9, 11}


def _normalize_pitches_to_c_am(pitches: list[int]) -> list[int]:
    """
    Fast O(N) key normalization to C major / A minor (matches Stage 1 key normalization
    in tokenize_dataset.py without music21 subprocess overhead). Finds the best-fitting scale
    among the major/minor scales and shifts pitches so the tonic is C (major) or A (minor).
    Semitone transposition is strictly in [0, 11] to preserve exact octave alignment with tokenize_dataset.py.
    """
    if not pitches:
        return pitches
    pcs = [p % 12 for p in pitches]
    best_hits = -1
    best_root = 0
    best_mode = "major"
    for root_pc, mode, scale_set in ALL_24_SCALES:
        hits = sum(1 for pc in pcs if pc in scale_set)
        if hits > best_hits:
            best_hits = hits
            best_root = root_pc
            best_mode = mode

    semitones = (0 - best_root) % 12 if best_mode == "major" else (9 - best_root) % 12
    if semitones == 0:
        return pitches
    return [max(0, min(127, p + semitones)) for p in pitches]

# Fixed IOI histogram bin edges in seconds (from 10ms to 2.0s)
IOI_BIN_EDGES = np.array(
    [0.0, 0.03, 0.06, 0.09, 0.125, 0.18, 0.25, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 4.0],
    dtype=np.float64,
)


# ============================================================================
# 1. Core Symbolic Music Metric Definitions
# ============================================================================

def _lcs_length(a: list[int], b: list[int]) -> int:
    """Exact Longest Common Subsequence length between two integer sequences."""
    if not a or not b:
        return 0
    la, lb = len(a), len(b)
    dp = [0] * (lb + 1)
    for x in a:
        prev = 0
        for j in range(lb):
            temp = dp[j + 1]
            if x == b[j]:
                dp[j + 1] = prev + 1
            elif dp[j] > dp[j + 1]:
                dp[j + 1] = dp[j]
            prev = temp
    return dp[lb]


class TrainingMemorizationIndex:
    """
    Fast inverted 6-gram index over training set pitch sequences per genre,
    allowing exact LCS memorization scoring in milliseconds per generated piece.
    """

    def __init__(self, seq_cap: int = 256, ngram_k: int = 6):
        self.seq_cap = seq_cap
        self.ngram_k = ngram_k
        # genre -> list of pitch sequences (each capped at seq_cap notes)
        self.seqs_by_genre: dict[str, list[list[int]]] = defaultdict(list)
        # genre -> {6-gram tuple: list of seq indices}
        self.index_by_genre: dict[str, dict[tuple[int, ...], list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Global fallback across all genres (for unconditional models like Moonbeam)
        self.all_seqs: list[list[int]] = []
        self.all_index: dict[tuple[int, ...], list[int]] = defaultdict(list)

    def add_sequence(self, genre: str, pitches: list[int]):
        if len(pitches) < self.ngram_k:
            return
        seq = pitches[: self.seq_cap]
        g_idx = len(self.seqs_by_genre[genre])
        self.seqs_by_genre[genre].append(seq)

        all_idx = len(self.all_seqs)
        self.all_seqs.append(seq)

        seen_ngrams = set()
        for i in range(len(seq) - self.ngram_k + 1):
            ng = tuple(seq[i : i + self.ngram_k])
            if ng not in seen_ngrams:
                seen_ngrams.add(ng)
                self.index_by_genre[genre][ng].append(g_idx)
                self.all_index[ng].append(all_idx)

    def score_memorization(self, genre: str, pitches: list[int], top_candidates: int = 20) -> dict[str, float]:
        """
        Compute:
          - max_lcs_ratio: max(LCS(gen, train)) / len(gen) over candidate training songs
          - train_6gram_overlap: fraction of 6-grams in gen that exist in the training set
        """
        seq = pitches[: self.seq_cap]
        if len(seq) < self.ngram_k:
            return {"max_lcs_ratio": 0.0, "train_6gram_overlap": 0.0}

        use_genre = genre in self.seqs_by_genre and len(self.seqs_by_genre[genre]) > 0
        pool_seqs = self.seqs_by_genre[genre] if use_genre else self.all_seqs
        pool_idx = self.index_by_genre[genre] if use_genre else self.all_index

        if not pool_seqs:
            return {"max_lcs_ratio": np.nan, "train_6gram_overlap": np.nan}

        ngrams = [tuple(seq[i : i + self.ngram_k]) for i in range(len(seq) - self.ngram_k + 1)]
        hit_count = 0
        candidate_hits: Counter[int] = Counter()

        for ng in ngrams:
            posting = pool_idx.get(ng)
            if posting:
                hit_count += 1
                for cand_id in posting:
                    candidate_hits[cand_id] += 1

        overlap_ratio = hit_count / len(ngrams)

        # Pick top candidate training tracks by shared n-grams + a few random baseline tracks
        cand_ids = [cid for cid, _ in candidate_hits.most_common(top_candidates)]
        if len(cand_ids) < 3:
            extra = random.sample(range(len(pool_seqs)), min(3, len(pool_seqs)))
            for eid in extra:
                if eid not in cand_ids:
                    cand_ids.append(eid)

        best_lcs = 0
        for cid in cand_ids:
            lcs_val = _lcs_length(seq, pool_seqs[cid])
            if lcs_val > best_lcs:
                best_lcs = lcs_val

        return {
            "max_lcs_ratio": best_lcs / len(seq),
            "train_6gram_overlap": overlap_ratio,
        }


def compute_single_midi_metrics(
    midi_path: Path,
    multi_instrument: bool = False,
    max_notes_cap: int | None = None,
    mem_index: TrainingMemorizationIndex | None = None,
    genre: str = "unconditional",
) -> dict[str, Any] | None:
    """
    Parse a MIDI file with symusic and compute all Stage 4.2 objective metrics.
    If max_notes_cap is set (e.g., for full-length real MIDI tracks), crops to the
    first max_notes_cap notes so length-dependent metrics are comparable to 1024-token chunks.
    """
    import symusic

    try:
        score = symusic.Score(str(midi_path))
    except Exception as e:
        log.debug(f"Failed parsing MIDI {midi_path}: {e}")
        return None

    tpq = max(int(score.ticks_per_quarter), 1)

    # Filter tracks identically to Stage 1
    if multi_instrument:
        valid_tracks = [t for t in score.tracks if not t.is_drum and len(t.notes) > 0]
    else:
        valid_tracks = [
            t for t in score.tracks if not t.is_drum and 0 <= t.program <= 7 and len(t.notes) > 0
        ]
        if not valid_tracks:
            valid_tracks = [t for t in score.tracks if not t.is_drum and len(t.notes) > 0]

    if not valid_tracks:
        return None

    # Also convert a copy to seconds for real-time IOI & duration metrics
    try:
        score_sec = score.to("second")
        if multi_instrument:
            sec_tracks = [t for t in score_sec.tracks if not t.is_drum and len(t.notes) > 0]
        else:
            sec_tracks = [
                t for t in score_sec.tracks
                if not t.is_drum and 0 <= t.program <= 7 and len(t.notes) > 0
            ]
            if not sec_tracks:
                sec_tracks = [t for t in score_sec.tracks if not t.is_drum and len(t.notes) > 0]
    except Exception:
        sec_tracks = []

    # Collect (start_tick, end_tick, pitch, velocity, program)
    notes_tick = []
    programs_used = set()
    for t in valid_tracks:
        prog_family = t.program // 8 if multi_instrument else 0
        programs_used.add(prog_family)
        for n in t.notes:
            notes_tick.append((int(n.time), int(n.time + n.duration), int(n.pitch), int(n.velocity)))

    if not notes_tick:
        return None

    notes_tick.sort(key=lambda x: (x[0], x[2]))
    if max_notes_cap is not None and len(notes_tick) > max_notes_cap:
        notes_tick = notes_tick[:max_notes_cap]

    num_notes = len(notes_tick)
    if num_notes < 4:
        return None

    pitches = [n[2] for n in notes_tick]
    pitches_arr = np.array(pitches, dtype=np.int32)
    pitch_classes = pitches_arr % 12

    # ── 1. Pitch Class Entropy (PCE) [bits] ──────────────────────────────────
    pc_counts = np.bincount(pitch_classes, minlength=12).astype(np.float64)
    pc_probs = pc_counts / num_notes
    nz = pc_probs > 0
    pitch_class_entropy = float(-np.sum(pc_probs[nz] * np.log2(pc_probs[nz])))

    # ── 2. Scale Consistency (SC) & C-maj/A-min Consistency ──────────────────
    best_sc = 0.0
    for _, _, scale_set in ALL_24_SCALES:
        in_scale = sum(1 for pc in pitch_classes if pc in scale_set)
        ratio = in_scale / num_notes
        if ratio > best_sc:
            best_sc = ratio
    ca_sc = sum(1 for pc in pitch_classes if pc in C_AM_SCALE) / num_notes

    # ── 3. Polyphony Rate (PR) & Polyphonic Steps Ratio ──────────────────────
    # Discretize onto a 16th-note grid (step = tpq / 4 ticks)
    grid_step = max(tpq // 4, 1)
    min_tick = notes_tick[0][0]
    max_tick = max(n[1] for n in notes_tick)
    total_grid_steps = max((max_tick - min_tick) // grid_step + 1, 1)
    # Cap grid length to avoid huge allocations on corrupt MIDI timestamps
    total_grid_steps = min(total_grid_steps, 200_000)

    active_counts = np.zeros(total_grid_steps, dtype=np.int32)
    for st, et, _, _ in notes_tick:
        s_idx = max(0, (st - min_tick) // grid_step)
        e_idx = max(s_idx + 1, (et - min_tick) // grid_step)
        if s_idx < total_grid_steps:
            active_counts[s_idx : min(e_idx, total_grid_steps)] += 1

    active_mask = active_counts > 0
    if np.any(active_mask):
        active_vals = active_counts[active_mask]
        polyphony_rate = float(np.mean(active_vals))
        polyphonic_ratio = float(np.mean(active_vals >= 2))
    else:
        polyphony_rate = 1.0
        polyphonic_ratio = 0.0

    # ── 4. Groove Consistency (GC) [bar-to-bar rhythmic onset similarity] ──
    # Measures rhythmic stability across measures via Cosine Similarity (Thesis Equation 3.12)
    # Default metric grid: 4/4 meter (16 sixteenth-note slots per bar)
    beats_per_bar = 4
    if hasattr(score, "time_signatures") and score.time_signatures:
        ts = score.time_signatures[0]
        if ts.numerator in (2, 3, 4, 6, 12):
            beats_per_bar = int(ts.numerator)

    num_slots = 16 if beats_per_bar == 4 else (12 if beats_per_bar in (3, 6) else 16)
    bar_ticks = tpq * beats_per_bar
    slot_ticks = max(bar_ticks // num_slots, 1)

    min_tick = notes_tick[0][0]
    max_tick = max(n[1] for n in notes_tick)
    bar_offset = min_tick // bar_ticks
    num_bars = min(max((max_tick // bar_ticks) - bar_offset + 1, 1), 512)

    if num_bars >= 2:
        bar_patterns = np.zeros((num_bars, num_slots), dtype=np.float64)
        for st, _, _, _ in notes_tick:
            b_idx = (st // bar_ticks) - bar_offset
            if 0 <= b_idx < num_bars:
                slot = (st % bar_ticks) // slot_ticks
                if 0 <= slot < num_slots:
                    bar_patterns[b_idx, slot] = 1.0

        # Keep non-empty bars (bars with at least one onset)
        non_empty_bars = bar_patterns[np.sum(bar_patterns, axis=1) > 0]
        if len(non_empty_bars) >= 2:
            # Pairwise cosine similarity between consecutive measures (Equation 3.12 in thesis)
            dots = np.sum(non_empty_bars[:-1] * non_empty_bars[1:], axis=1)
            norms = np.linalg.norm(non_empty_bars[:-1], axis=1) * np.linalg.norm(non_empty_bars[1:], axis=1)
            valid = norms > 0
            if np.any(valid):
                groove_consistency = float(np.mean(dots[valid] / norms[valid]))
            else:
                groove_consistency = 1.0
        else:
            groove_consistency = 1.0
    else:
        groove_consistency = 1.0

    # ── 5. Inter-Onset Interval (IOI) & Real-Time Duration ───────────────────
    notes_sec = []
    for t in sec_tracks:
        for n in t.notes:
            notes_sec.append((float(n.time), float(n.duration), int(n.pitch)))
    notes_sec.sort(key=lambda x: (x[0], x[2]))
    if max_notes_cap is not None and len(notes_sec) > max_notes_cap:
        notes_sec = notes_sec[:max_notes_cap]

    if len(notes_sec) >= 4:
        onsets_sec = np.array([n[0] for n in notes_sec], dtype=np.float64)
        durs_sec = np.array([max(n[1], 1e-4) for n in notes_sec], dtype=np.float64)
    else:
        # Fallback: 120 BPM (0.5s per quarter note)
        sec_per_tick = 0.5 / tpq
        onsets_sec = np.array([n[0] * sec_per_tick for n in notes_tick], dtype=np.float64)
        durs_sec = np.array([max((n[1] - n[0]) * sec_per_tick, 1e-4) for n in notes_tick], dtype=np.float64)

    piece_duration_sec = float(max(np.max(onsets_sec + durs_sec) - np.min(onsets_sec), 0.1))
    note_density = float(num_notes / piece_duration_sec)
    duration_mean = float(np.mean(durs_sec))

    # Unique chord/event onset times (collapse simultaneous chord notes within 10ms)
    unique_onsets = [onsets_sec[0]]
    for t_val in onsets_sec[1:]:
        if t_val - unique_onsets[-1] >= 0.010:
            unique_onsets.append(t_val)
    unique_onsets_arr = np.array(unique_onsets, dtype=np.float64)

    if len(unique_onsets_arr) >= 2:
        raw_iois = np.diff(unique_onsets_arr)
        valid_iois = raw_iois[(raw_iois >= 0.010) & (raw_iois <= 4.0)]
        if len(valid_iois) == 0:
            valid_iois = raw_iois
        ioi_mean = float(np.mean(valid_iois))
        ioi_std = float(np.std(valid_iois))
        ioi_cv = float(ioi_std / max(ioi_mean, 1e-6))

        hist_counts, _ = np.histogram(np.clip(valid_iois, 0.0, 3.999), bins=IOI_BIN_EDGES)
        hist_probs = hist_counts / max(hist_counts.sum(), 1)
        nz_ioi = hist_probs > 0
        ioi_entropy = float(-np.sum(hist_probs[nz_ioi] * np.log2(hist_probs[nz_ioi])))
        ioi_hist_list = hist_counts.astype(int).tolist()
    else:
        ioi_mean = 0.0
        ioi_std = 0.0
        ioi_cv = 0.0
        ioi_entropy = 0.0
        ioi_hist_list = [0] * (len(IOI_BIN_EDGES) - 1)

    # ── 6. Pitch Range & Self-Repetition / Memorization ──────────────────────
    pitch_range = int(np.max(pitches_arr) - np.min(pitches_arr))
    unique_pitches = int(len(np.unique(pitches_arr)))

    # Internal 4-gram repetition ratio (1 - unique_4grams / total_4grams)
    if num_notes >= 8:
        ng4 = [tuple(pitches[i : i + 4]) for i in range(num_notes - 3)]
        self_rep_4gram = float(1.0 - (len(set(ng4)) / len(ng4)))
    else:
        self_rep_4gram = 0.0

    mem_scores = {"max_lcs_ratio": np.nan, "train_6gram_overlap": np.nan}
    if mem_index is not None:
        norm_pitches = _normalize_pitches_to_c_am(pitches[:256])
        mem_scores = mem_index.score_memorization(genre, norm_pitches)

    return {
        "num_notes": num_notes,
        "duration_sec": piece_duration_sec,
        "note_density": note_density,
        "duration_mean": duration_mean,
        "num_instruments": len(programs_used),
        "pitch_range": pitch_range,
        "unique_pitches": unique_pitches,
        "pitch_class_entropy": pitch_class_entropy,
        "scale_consistency": best_sc,
        "ca_scale_consistency": ca_sc,
        "polyphony_rate": polyphony_rate,
        "polyphonic_ratio": polyphonic_ratio,
        "groove_consistency": groove_consistency,
        "ioi_mean": ioi_mean,
        "ioi_std": ioi_std,
        "ioi_cv": ioi_cv,
        "ioi_entropy": ioi_entropy,
        "self_repetition_4gram": self_rep_4gram,
        "max_lcs_ratio": mem_scores["max_lcs_ratio"],
        "train_6gram_overlap": mem_scores["train_6gram_overlap"],
        "_ioi_hist_counts": ioi_hist_list,
        "_pitches_head": pitches[:256],
    }


# ============================================================================
# 2. Building Training Set Memorization Index & Real Test Set Baseline
# ============================================================================

def resolve_raw_midi_root(sync_dir: Path, explicit_root: Path | str | None = None) -> Path | None:
    """Find raw_dataset folder if available on disk or cluster storage."""
    candidates = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    if os.getenv("MUSICGEN_MIDI_ROOT"):
        candidates.append(Path(os.environ["MUSICGEN_MIDI_ROOT"]))
    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "raw_dataset")
    candidates.append(sync_dir / "raw_dataset")
    candidates.append(sync_dir.parent / "raw_dataset")
    if os.getenv("SCRATCH"):
        sc = Path(os.environ["SCRATCH"])
        candidates.extend([
            sc / "raw_dataset",
            sc / "musicgen" / "raw_dataset",
            sc / "dataset_backup" / "raw_dataset",
        ])
    if os.getenv("PLG_GROUPS_STORAGE"):
        pg = Path(os.environ["PLG_GROUPS_STORAGE"])
        plg_proj = os.getenv("PLG_PROJECT_NAME", "music_gen")
        candidates.extend([
            pg / plg_proj / "raw_dataset",
        ])
    for c in candidates:
        if c.is_dir():
            return c
    return None


def resolve_hf_remi_split(dataset_name: str, split: str, hf_split: str, sync_dir: Path) -> Path | None:
    """Find pre-tokenized HuggingFace remi_bpe/<hf_split> dataset if raw_dataset isn't available."""
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "datasets_metadata" / dataset_name / "dataset" / split / "remi_bpe" / hf_split,
        sync_dir / "datasets_metadata" / dataset_name / "dataset" / split / "remi_bpe" / hf_split,
        sync_dir / "artist_genres" / dataset_name / "dataset" / split / "remi_bpe" / hf_split,
    ]
    if os.getenv("SCRATCH"):
        sp = Path(os.environ["SCRATCH"])
        candidates.extend([
            sp / "dataset_backup" / dataset_name / "dataset" / split / "remi_bpe" / hf_split,
            sp / "dataset_backup" / dataset_name / split / "remi_bpe" / hf_split,
        ])
    for c in candidates:
        if c.is_dir() and (c / "dataset_info.json").is_file():
            return c
    return None


def _load_remi_decoder(dataset_name: str, split: str, sync_dir: Path):
    """Load REMI+BPE tokenizer and id2genre map for decoding pre-tokenized real tracks when raw_dataset is absent."""
    from generate import find_remi_tokenizer_path, load_genres_list
    from miditok import REMI

    tok_file = find_remi_tokenizer_path(sync_dir, dataset_name, split)
    try:
        remi_tok = REMI(params=str(tok_file))
    except Exception as e:
        if "ModelWrapper" not in str(e):
            raise
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
            remi_tok = REMI(params=tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    genres = load_genres_list(sync_dir, dataset_name, split)
    id2genre = {i: g for i, g in enumerate(genres)}
    return remi_tok, id2genre


def _decode_remi_ids_to_score(remi_tok, token_ids: list[int]):
    """Decode REMI+BPE token IDs directly to a symusic.Score object in memory."""
    from miditok import TokSequence

    base_vocab = len(remi_tok)
    pad_id = remi_tok["PAD_None"]
    valid_ids = [int(t) for t in token_ids if 0 <= int(t) < base_vocab and int(t) != pad_id]
    if not valid_ids:
        return None
    tok_seq = TokSequence(ids=valid_ids, are_ids_encoded=True)
    decode_input = tok_seq if getattr(remi_tok, "one_token_stream", True) else [tok_seq]
    return remi_tok.decode(decode_input)


def build_memorization_index(
    dataset_name: str,
    split: str,
    sync_dir: Path,
    explicit_raw_root: Path | str | None = None,
    max_per_genre: int = 150,
    seed: int = 42,
) -> TrainingMemorizationIndex | None:
    """
    Build a fast 6-gram + LCS reference index from a stratified sample of train.csv
    (or from pre-tokenized remi_bpe/train if raw_dataset is not present on this cluster).
    """
    import symusic

    raw_root = resolve_raw_midi_root(sync_dir, explicit_root=explicit_raw_root)
    repo_root = Path(__file__).resolve().parent.parent
    train_candidates = [
        repo_root / "datasets_metadata" / dataset_name / "split" / split / "train.csv",
        sync_dir / "datasets_metadata" / dataset_name / "split" / split / "train.csv",
        sync_dir / "artist_genres" / dataset_name / "split" / split / "train.csv",
    ]
    train_csv = next((p for p in train_candidates if p.is_file()), train_candidates[0])
    multi_instrument = split == "multi-instrument"
    rng = random.Random(seed)
    mem_index = TrainingMemorizationIndex(seq_cap=256, ngram_k=6)
    loaded = 0

    if raw_root is not None and train_csv.is_file():
        by_genre: dict[str, list[str]] = defaultdict(list)
        with open(train_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_genre[row["genre"]].append(row["file_path"])

        for genre, rel_paths in sorted(by_genre.items()):
            sampled = rng.sample(rel_paths, min(max_per_genre, len(rel_paths)))
            for rp in sampled:
                clean_rp = rp[2:] if rp.startswith("./") else rp
                fpath = raw_root / clean_rp
                if not fpath.is_file():
                    fpath = raw_root / Path(rp)
                if not fpath.is_file():
                    continue
                try:
                    score = symusic.Score(str(fpath))
                    if multi_instrument:
                        tracks = [t for t in score.tracks if not t.is_drum and len(t.notes) > 0]
                    else:
                        tracks = [
                            t for t in score.tracks
                            if not t.is_drum and 0 <= t.program <= 7 and len(t.notes) > 0
                        ]
                    if not tracks:
                        continue
                    notes = []
                    for t in tracks:
                        for n in t.notes:
                            notes.append((int(n.time), int(n.pitch)))
                    notes.sort(key=lambda x: (x[0], x[1]))
                    pitches = _normalize_pitches_to_c_am([p for _, p in notes[:256]])
                    mem_index.add_sequence(genre, pitches)
                    loaded += 1
                except Exception:
                    continue
        if loaded > 0:
            log.info(f"Built Training Memorization Index ({loaded} reference training tracks from {raw_root}).")
            return mem_index

    # Fallback: build memorization index from pre-tokenized remi_bpe/train Arrow dataset
    hf_train_dir = resolve_hf_remi_split(dataset_name, split, "train", sync_dir)
    if hf_train_dir is not None:
        try:
            from datasets import load_from_disk

            remi_tok, id2genre = _load_remi_decoder(dataset_name, split, sync_dir)
            ds = load_from_disk(str(hf_train_dir))
            by_gid: dict[int, list[int]] = defaultdict(list)
            gids = ds["genre_id"]
            for i, gid in enumerate(gids):
                by_gid[int(gid)].append(i)

            for gid, indices in sorted(by_gid.items()):
                genre = id2genre.get(gid, f"genre_{gid}")
                sampled_idx = rng.sample(indices, min(max_per_genre, len(indices)))
                for idx in sampled_idx:
                    try:
                        score = _decode_remi_ids_to_score(remi_tok, ds[idx]["input_ids"])
                        if score is None:
                            continue
                        notes = []
                        for t in score.tracks:
                            for n in t.notes:
                                notes.append((int(n.time), int(n.pitch)))
                        notes.sort(key=lambda x: (x[0], x[1]))
                        pitches = _normalize_pitches_to_c_am([p for _, p in notes[:256]])
                        mem_index.add_sequence(genre, pitches)
                        loaded += 1
                    except Exception:
                        continue
            if loaded > 0:
                log.info(f"Built Training Memorization Index ({loaded} reference training tracks from {hf_train_dir}).")
                return mem_index
        except Exception as e:
            log.warning(f"Could not build memorization index from {hf_train_dir}: {e}")

    log.info("Training MIDIs not found on this node — skipping train LCS index. Pass --raw-midi-root to enable.")
    return None


def collect_real_baseline_items(
    dataset_name: str,
    split: str,
    sync_dir: Path,
    explicit_raw_root: Path | str | None = None,
    songs_per_genre: int = 100,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], tempfile.TemporaryDirectory | None]:
    """
    Sample up to `songs_per_genre` real tracks per genre from test.csv (if raw_dataset is present)
    or decode them from pre-tokenized remi_bpe/test (if dataset_backup is present).
    """
    raw_root = resolve_raw_midi_root(sync_dir, explicit_root=explicit_raw_root)
    repo_root = Path(__file__).resolve().parent.parent
    test_candidates = [
        repo_root / "datasets_metadata" / dataset_name / "split" / split / "test.csv",
        sync_dir / "datasets_metadata" / dataset_name / "split" / split / "test.csv",
        sync_dir / "artist_genres" / dataset_name / "split" / split / "test.csv",
    ]
    test_csv = next((p for p in test_candidates if p.is_file()), test_candidates[0])
    rng = random.Random(seed)

    if raw_root is not None and test_csv.is_file():
        by_genre: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with open(test_csv, "r", encoding="utf-8") as f:
            for idx, row in enumerate(csv.DictReader(f)):
                raw_fp = row["file_path"]
                clean_rp = raw_fp[2:] if raw_fp.startswith("./") else raw_fp
                fpath = raw_root / clean_rp
                if not fpath.is_file():
                    fpath = raw_root / Path(raw_fp)
                if fpath.is_file():
                    by_genre[row["genre"]].append({
                        "song_id": f"real_{row['genre']}_{idx:04d}",
                        "file_path": fpath,
                        "rel_path": raw_fp,
                        "target_genre": row["genre"],
                    })
        if by_genre:
            selected: list[dict[str, Any]] = []
            for genre, rows in sorted(by_genre.items()):
                sampled = rng.sample(rows, min(songs_per_genre, len(rows)))
                selected.extend(sampled)
            return selected, None

    # Fallback: decode real test tracks from pre-tokenized remi_bpe/test Arrow dataset
    hf_test_dir = resolve_hf_remi_split(dataset_name, split, "test", sync_dir)
    if hf_test_dir is not None:
        try:
            from datasets import load_from_disk

            log.info(f"Decoding real test baseline tracks from {hf_test_dir}...")
            remi_tok, id2genre = _load_remi_decoder(dataset_name, split, sync_dir)
            ds = load_from_disk(str(hf_test_dir))
            by_gid: dict[int, list[int]] = defaultdict(list)
            for i, gid in enumerate(ds["genre_id"]):
                by_gid[int(gid)].append(i)

            tmp_handle = tempfile.TemporaryDirectory(prefix=f"real_{dataset_name}_{split}_")
            tmp_root = Path(tmp_handle.name)
            selected = []
            for gid, indices in sorted(by_gid.items()):
                genre = id2genre.get(gid, f"genre_{gid}")
                g_dir = tmp_root / genre
                g_dir.mkdir(parents=True, exist_ok=True)
                sampled_idx = rng.sample(indices, min(songs_per_genre, len(indices)))
                for k, idx in enumerate(sampled_idx):
                    try:
                        score = _decode_remi_ids_to_score(remi_tok, ds[idx]["input_ids"])
                        if score is None:
                            continue
                        song_id = f"real_{genre}_{k:04d}"
                        fpath = g_dir / f"{song_id}.mid"
                        if hasattr(score, "dump_midi"):
                            score.dump_midi(str(fpath))
                        else:
                            score.dump(str(fpath))
                        if fpath.is_file() and fpath.stat().st_size > 0:
                            selected.append({
                                "song_id": song_id,
                                "file_path": fpath,
                                "rel_path": f"{genre}/{song_id}.mid",
                                "target_genre": genre,
                            })
                    except Exception:
                        continue
            if selected:
                return selected, tmp_handle
            tmp_handle.cleanup()
        except Exception as e:
            log.warning(f"Failed decoding real test baseline from {hf_test_dir}: {e}")

    return [], None


# ============================================================================
# 3. Aggregating Per-Song Metrics into Model & Genre Summaries
# ============================================================================

METRIC_KEYS = [
    "pitch_class_entropy",
    "scale_consistency",
    "ca_scale_consistency",
    "polyphony_rate",
    "polyphonic_ratio",
    "groove_consistency",
    "ioi_mean",
    "ioi_std",
    "ioi_cv",
    "ioi_entropy",
    "note_density",
    "duration_mean",
    "duration_sec",
    "num_notes",
    "pitch_range",
    "unique_pitches",
    "num_instruments",
    "self_repetition_4gram",
    "max_lcs_ratio",
    "train_6gram_overlap",
]


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log((p + eps) / (m + eps)))
        + 0.5 * np.sum(q * np.log((q + eps) / (m + eps)))
    )


def summarize_song_rows(
    song_rows: list[dict[str, Any]],
    real_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute mean, std, median, IQR for all metrics, plus IOI histogram JS divergence."""
    summary: dict[str, Any] = {"num_valid_songs": len(song_rows)}
    if not song_rows:
        return summary

    # Aggregate IOI histogram
    total_ioi_hist = np.sum([np.array(r["_ioi_hist_counts"], dtype=np.float64) for r in song_rows], axis=0)
    ioi_hist_dist = (total_ioi_hist / max(total_ioi_hist.sum(), 1.0)).tolist()
    summary["ioi_histogram_probs"] = ioi_hist_dist
    summary["ioi_histogram_bin_edges"] = IOI_BIN_EDGES.tolist()

    if real_reference and "ioi_histogram_probs" in real_reference:
        real_ioi = np.array(real_reference["ioi_histogram_probs"], dtype=np.float64)
        summary["ioi_js_div_to_real"] = _js_divergence(total_ioi_hist, real_ioi)
    else:
        summary["ioi_js_div_to_real"] = None

    metrics_stats: dict[str, dict[str, float | None]] = {}
    for mk in METRIC_KEYS:
        vals = np.array(
            [r[mk] for r in song_rows if r.get(mk) is not None and not np.isnan(r[mk])],
            dtype=np.float64,
        )
        if len(vals) == 0:
            metrics_stats[mk] = {"mean": None, "std": None, "median": None, "iqr": None}
            continue

        q75, q25 = np.percentile(vals, [75, 25])
        stat_entry: dict[str, float | None] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "iqr": float(q75 - q25),
        }

        # Statistical significance (Mann-Whitney U test) vs Real Baseline if available
        if real_reference and "_raw_values" in real_reference and mk in real_reference["_raw_values"]:
            real_vals = np.array(real_reference["_raw_values"][mk], dtype=np.float64)
            if len(real_vals) >= 5 and len(vals) >= 5:
                try:
                    from scipy.stats import mannwhitneyu
                    _, p_val = mannwhitneyu(vals, real_vals, alternative="two-sided")
                    stat_entry["p_value_vs_real"] = float(p_val)
                except Exception:
                    stat_entry["p_value_vs_real"] = None
                stat_entry["diff_vs_real"] = float(np.mean(vals) - np.mean(real_vals))

        metrics_stats[mk] = stat_entry

    summary["metrics"] = metrics_stats
    return summary


# ============================================================================
# 4. Evaluating a Model (or Real Baseline)
# ============================================================================

def evaluate_model_metrics(
    dataset_name: str,
    split: str,
    model_name: str,
    sync_dir: Path,
    out_base_dir: Path,
    generated_root: Path | None = None,
    mem_index: TrainingMemorizationIndex | None = None,
    real_reference: dict[str, Any] | None = None,
    real_songs_per_genre: int = 100,
    real_max_notes: int = 512,
    explicit_raw_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Evaluate all MIDI files for (dataset_name, split, model_name)."""
    from evaluate_classifier import collect_midi_items, resolve_generated_model_dir

    tmp_handle = None
    max_notes_cap = None

    if model_name == "real":
        items, tmp_handle = collect_real_baseline_items(
            dataset_name=dataset_name,
            split=split,
            sync_dir=sync_dir,
            explicit_raw_root=explicit_raw_root,
            songs_per_genre=real_songs_per_genre,
        )
        if not items:
            # Check if real baseline was already evaluated (e.g. on Athena) and saved in out_base_dir
            cached_dir = out_base_dir / dataset_name / split / "real"
            cached_summary = cached_dir / "summary.json"
            cached_songs = cached_dir / "per_song_metrics.csv"
            if cached_summary.is_file() and cached_songs.is_file():
                log.info(f"Loading cached Real Baseline reference from {cached_dir}...")
                with open(cached_summary, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                raw_vals: dict[str, list[float]] = defaultdict(list)
                with open(cached_songs, "r", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        for mk in METRIC_KEYS:
                            v = r.get(mk, "")
                            if v != "":
                                try:
                                    raw_vals[mk].append(float(v))
                                except ValueError:
                                    pass
                summary_data["_raw_values"] = dict(raw_vals)
                return summary_data
            log.warning(f"[SKIP] Could not collect real baseline MIDIs for {dataset_name} | {split}")
            return None
        # Cap full-length real tracks at real_max_notes (~512 notes) for apples-to-apples comparison with generative chunks
        max_notes_cap = real_max_notes
        source_desc = str(test_csv)
    else:
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
        source_desc = str(model_dir)

    try:
        if not items:
            log.warning(f"[SKIP] 0 MIDI items for {dataset_name} | {split} | {model_name}")
            return None

        log.info("=" * 70)
        log.info(
            f"Evaluating Objective Metrics | dataset={dataset_name} | split={split} | "
            f"model={model_name} | files={len(items)}"
        )
        log.info(f"Source: {source_desc}")
        log.info("=" * 70)

        multi_instrument = split == "multi-instrument"
        song_rows: list[dict[str, Any]] = []
        by_genre_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        failed_count = 0

        for idx, it in enumerate(items):
            m = compute_single_midi_metrics(
                midi_path=it["file_path"],
                multi_instrument=multi_instrument,
                max_notes_cap=max_notes_cap,
                mem_index=mem_index,
                genre=it["target_genre"],
            )
            if m is None:
                failed_count += 1
                continue

            row = {
                "song_id": it["song_id"],
                "rel_path": it["rel_path"],
                "genre": it["target_genre"],
                **m,
            }
            song_rows.append(row)
            by_genre_rows[it["target_genre"]].append(row)

            if (idx + 1) % 200 == 0 or (idx + 1) == len(items):
                log.info(
                    f"  Processed {idx + 1}/{len(items)} files "
                    f"(valid={len(song_rows)}, failed/empty={failed_count})"
                )

        if not song_rows:
            log.error(f"0 valid MIDI files for {dataset_name} | {split} | {model_name}")
            return None

        overall_summary = summarize_song_rows(song_rows, real_reference=real_reference)
        overall_summary.update({
            "dataset": dataset_name,
            "split": split,
            "model": model_name,
            "total_files": len(items),
            "valid_files": len(song_rows),
            "failed_files": failed_count,
            "valid_rate": len(song_rows) / max(len(items), 1),
        })

        # Keep raw arrays in memory if this is the 'real' baseline so subsequent models can run Mann-Whitney U
        if model_name == "real":
            overall_summary["_raw_values"] = {
                mk: [
                    r[mk] for r in song_rows
                    if r.get(mk) is not None and not np.isnan(r[mk])
                ]
                for mk in METRIC_KEYS
            }

        # Per-genre breakdown
        per_genre_summary: dict[str, Any] = {}
        for g_name, g_rows in sorted(by_genre_rows.items()):
            per_genre_summary[g_name] = summarize_song_rows(g_rows)
        overall_summary["per_genre"] = per_genre_summary

        # Save outputs to out_base_dir/<dataset>/<split>/<model>/
        out_dir = out_base_dir / dataset_name / split / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. summary.json (strip internal _raw_values before JSON serialization)
        json_copy = {k: v for k, v in overall_summary.items() if not k.startswith("_")}
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(json_copy, f, indent=2)

        # 2. per_song_metrics.csv
        song_fields = ["song_id", "rel_path", "genre"] + METRIC_KEYS
        with open(out_dir / "per_song_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=song_fields)
            writer.writeheader()
            for r in song_rows:
                clean_r = {
                    k: (round(r[k], 6) if isinstance(r[k], float) and not np.isnan(r[k]) else ("" if isinstance(r[k], float) else r[k]))
                    for k in song_fields
                }
                writer.writerow(clean_r)

        # 3. per_genre_metrics.csv
        pg_fields = ["genre", "num_valid_songs"] + [f"{mk}_mean" for mk in METRIC_KEYS] + [f"{mk}_std" for mk in METRIC_KEYS]
        with open(out_dir / "per_genre_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pg_fields)
            writer.writeheader()
            for g_name, g_sum in sorted(per_genre_summary.items()):
                g_row: dict[str, Any] = {
                    "genre": g_name,
                    "num_valid_songs": g_sum["num_valid_songs"],
                }
                for mk in METRIC_KEYS:
                    m_stat = g_sum["metrics"].get(mk, {})
                    g_row[f"{mk}_mean"] = round(m_stat["mean"], 6) if m_stat.get("mean") is not None else ""
                    g_row[f"{mk}_std"] = round(m_stat["std"], 6) if m_stat.get("std") is not None else ""
                writer.writerow(g_row)

        m_stats = overall_summary["metrics"]
        log.info(
            f"RESULT [{dataset_name} | {split} | {model_name}] "
            f"PCE={m_stats['pitch_class_entropy']['mean']:.3f}±{m_stats['pitch_class_entropy']['std']:.3f} | "
            f"SC={m_stats['scale_consistency']['mean']*100:.1f}% | "
            f"PR={m_stats['polyphony_rate']['mean']:.2f} | "
            f"GC={m_stats['groove_consistency']['mean']*100:.1f}% | "
            f"LCS={m_stats['max_lcs_ratio']['mean'] or 0.0:.3f}"
        )
        return overall_summary

    finally:
        if tmp_handle is not None:
            tmp_handle.cleanup()


# ============================================================================
# 5. CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute Stage 4.2 Objective Musical Metrics for generated and real MIDI datasets."
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
        help="Model to evaluate ('real', 'lstm', 'transformer', 'llama', 'moonbeam', or 'all').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all datasets, splits, and models.",
    )
    parser.add_argument(
        "--include-real",
        action="store_true",
        default=True,
        help="Evaluate the real test.csv dataset baseline first for comparison (default: True).",
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
        help="Number of real test songs per genre to sample for the baseline (default: 100).",
    )
    parser.add_argument(
        "--real-max-notes",
        type=int,
        default=512,
        help="Note cap for real full-length MIDI tracks to match generative chunk note length (default: 512).",
    )
    parser.add_argument(
        "--raw-midi-root",
        type=str,
        default=None,
        help="Root directory of raw dataset MIDIs (defaults to $MUSICGEN_MIDI_ROOT or $SYNC_DIR/raw_dataset).",
    )
    parser.add_argument(
        "--generated-root",
        type=str,
        default=None,
        help="Root directory of generated MIDIs (defaults to $SCRATCH/generated_midis).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for objective metrics (defaults to $SYNC_DIR/evaluation_results/metrics).",
    )

    args = parser.parse_args()
    if args.multi_instrument:
        args.split = "multi-instrument"

    repo_root = Path(__file__).resolve().parent.parent
    sync_dir = Path(os.getenv("MUSICGEN_SYNC_DIR", repo_root))
    out_base_dir = Path(args.output_dir) if args.output_dir else (repo_root / "evaluation_results" / "metrics")
    out_base_dir.mkdir(parents=True, exist_ok=True)
    generated_root = Path(args.generated_root) if args.generated_root else None

    target_datasets = DATASETS if (args.all or args.dataset_name == "all") else [args.dataset_name]
    target_splits = SPLITS if (args.all or args.split == "all") else [args.split]

    if args.all or args.model == "all":
        target_models = ["real", "lstm", "transformer", "llama", "moonbeam"] if not args.no_real else ["lstm", "transformer", "llama", "moonbeam"]
    else:
        target_models = [args.model]
        if args.include_real and not args.no_real and "real" not in target_models:
            target_models = ["real"] + target_models

    master_rows: list[dict[str, Any]] = []

    for ds in target_datasets:
        for sp in target_splits:
            mem_index = build_memorization_index(
                dataset_name=ds,
                split=sp,
                sync_dir=sync_dir,
                explicit_raw_root=args.raw_midi_root,
            )
            real_ref: dict[str, Any] | None = None

            for m_name in target_models:
                res = evaluate_model_metrics(
                    dataset_name=ds,
                    split=sp,
                    model_name=m_name,
                    sync_dir=sync_dir,
                    out_base_dir=out_base_dir,
                    generated_root=generated_root,
                    mem_index=mem_index,
                    real_reference=real_ref,
                    real_songs_per_genre=args.real_songs_per_genre,
                    real_max_notes=args.real_max_notes,
                    explicit_raw_root=args.raw_midi_root,
                )
                if res is not None:
                    if m_name == "real":
                        real_ref = res

                    row: dict[str, Any] = {
                        "dataset": ds,
                        "split": sp,
                        "model": m_name,
                        "total_files": res["total_files"],
                        "valid_files": res["valid_files"],
                        "valid_rate": round(res["valid_rate"], 6),
                        "ioi_js_div_to_real": (
                            round(res["ioi_js_div_to_real"], 6)
                            if res.get("ioi_js_div_to_real") is not None
                            else ""
                        ),
                    }
                    for mk in METRIC_KEYS:
                        m_stat = res["metrics"].get(mk, {})
                        row[f"{mk}_mean"] = round(m_stat["mean"], 6) if m_stat.get("mean") is not None else ""
                        row[f"{mk}_std"] = round(m_stat["std"], 6) if m_stat.get("std") is not None else ""
                        row[f"{mk}_median"] = round(m_stat["median"], 6) if m_stat.get("median") is not None else ""
                        if "p_value_vs_real" in m_stat and m_stat["p_value_vs_real"] is not None:
                            row[f"{mk}_pval"] = round(m_stat["p_value_vs_real"], 6)
                        else:
                            row[f"{mk}_pval"] = ""
                    master_rows.append(row)

    if master_rows:
        master_csv = out_base_dir / "master_objective_metrics.csv"
        existing_map: dict[tuple[str, str, str], dict[str, Any]] = {}
        if master_csv.is_file() and master_csv.stat().st_size > 0:
            with open(master_csv, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    existing_map[(r["dataset"], r["split"], r["model"])] = r

        for r in master_rows:
            existing_map[(r["dataset"], r["split"], r["model"])] = r

        merged_rows = sorted(
            existing_map.values(),
            key=lambda x: (x["dataset"], x["split"], x["model"]),
        )
        fieldnames = list(master_rows[0].keys())
        with open(master_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)

        log.info(f"Saved aggregated objective metrics ({len(merged_rows)} rows) to {master_csv}")
    else:
        log.warning("No models were evaluated.")


if __name__ == "__main__":
    main()
