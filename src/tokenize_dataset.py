"""
tokenize_dataset.py

Stage 1 Preprocessing:
  1. Load train/val/test CSV splits (each row has a file_path + genre label).
  2. For each MIDI file:
     a. Open with symusic and extract piano-only tracks (program 0-7, channel != 9/10).
     b. Detect key with music21, transpose to C major or A minor.
     c. Tokenize using MidiTok REMI (for the 4 generative models).
     d. Tokenize using MidiTok Octuple (for the MusicBERT genre classifier).
  3. Train BPE on REMI tokens from a random 5k-track subset of train split.
  4. Apply BPE to all REMI tokens.
  5. Chunk all token sequences into 1024-token sliding windows (50% stride).
  6. Save two HuggingFace datasets:
       - giga_midi_genres/dataset/remi_bpe/   (for generative models)
       - giga_midi_genres/dataset/octuple/    (for MusicBERT classifier)

Usage:
    # Local test on small sample (fast, ~100 songs):
    python tokenize_dataset.py --sample 100 --workers 2

    # Full run on Ares (60k songs, all CPU cores):
    python tokenize_dataset.py --workers 48
"""

import argparse
import logging
import os
import random
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
import signal
from contextlib import contextmanager

import pandas as pd
from notify_run import Notify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ---------------------------------------------------------------------------
# Paths
# Can be overridden by env vars for SLURM scratch disk
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = Path(os.getenv("MUSICGEN_SPLITS_DIR", PROJECT_ROOT / "datasets_metadata" / "gt_1000_with_ties" / "split" / "piano"))
RAW_MIDI_ROOT = Path(os.getenv("MUSICGEN_MIDI_ROOT", PROJECT_ROOT / "raw_dataset"))
OUTPUT_DIR = Path(os.getenv("MUSICGEN_OUTPUT_DIR", PROJECT_ROOT / "datasets_metadata" / "gt_1000_with_ties" / "dataset" / "piano"))
REMI_BPE_DIR = OUTPUT_DIR / "remi_bpe"
OCTUPLE_DIR = OUTPUT_DIR / "octuple"

# ---------------------------------------------------------------------------
# Tokenization settings (matching the thesis implementation plan)
# ---------------------------------------------------------------------------
REMI_CHUNK_LEN = 1024         # tokens per sequence chunk for generative models
REMI_CHUNK_STRIDE = 512       # 50% overlap

OCTUPLE_CHUNK_LEN = 2048      # Double the context window for the classifier
OCTUPLE_CHUNK_STRIDE = 1024   # 50% overlap
BPE_VOCAB_SIZE = 10_000  # final vocab size after BPE merges
BPE_TRAIN_SAMPLE = 5_000 # number of files used to train BPE
RANDOM_SEED = 42

# GENRE2ID will be passed to workers dynamically based on the dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_midi_path(raw_path: str, root: Path) -> Path | None:
    """
    Convert the CSV file_path (starts with './' relative to gigamidi root)
    to an absolute path on disk.
    """
    p = Path(raw_path)
    full = root / p
    if full.exists():
        return full
    return None


def extract_midi_tracks(midi_path: Path, multi_instrument: bool = False):
    """
    Load MIDI with symusic.
    If multi_instrument is False: keep only piano tracks (program 0-7, non-drum) and merge them.
    If multi_instrument is True: keep all non-drum tracks, grouping them by General MIDI instrument family.
    Returns a symusic.Score or None on failure.
    """
    try:
        import symusic
        score = symusic.Score(str(midi_path))
        
        if multi_instrument:
            valid_tracks = [t for t in score.tracks if not t.is_drum and len(t.notes) > 0]
            if not valid_tracks:
                log.warning(f"No valid pitched tracks in {midi_path}.")
                return "no_piano"
            
            # Group by GM family (each family has 8 instruments)
            family_notes = {}
            for t in valid_tracks:
                family = t.program // 8
                if family not in family_notes:
                    family_notes[family] = []
                family_notes[family].extend(t.notes)
                
            new_tracks = []
            for family, notes in family_notes.items():
                notes.sort(key=lambda n: (n.time, n.pitch))
                merged_track = symusic.Track()
                merged_track.program = family * 8  # Use the first program in the family
                merged_track.is_drum = False
                merged_track.notes.extend(notes)
                new_tracks.append(merged_track)
                
            # Sort tracks by program number for determinism
            new_tracks.sort(key=lambda t: t.program)
            score.tracks = new_tracks
            return score
        else:
            piano_tracks = [
                t for t in score.tracks
                if not t.is_drum and 0 <= t.program <= 7 and len(t.notes) > 0
            ]
            if not piano_tracks:
                found = [(t.program, t.is_drum, len(t.notes)) for t in score.tracks]
                log.warning(f"No valid piano tracks in {midi_path}. Found tracks (prog, is_drum, notes): {found}")
                return "no_piano"
            
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
    except Exception as e:
        log.warning(f"symusic parse error on {midi_path}: {e}")
        return "parse_error"


def transpose_to_c(score, midi_path: Path):
    """
    Detect key using music21, return the semitone transposition needed
    to put the piece in C major or A minor.
    Returns the transposed symusic.Score.
    """
    import tempfile
    try:
        import symusic
        import music21

        # Create a lightweight copy of the score for fast key detection.
        # music21 parsing (chordify) scales catastrophically with polyphony.
        # We only need a few hundred notes of the main track to detect the key.
        analysis_score = symusic.Score(score.ticks_per_quarter)
        
        if score.tracks:
            # Find the track with the most notes (likely the melody or piano accompaniment)
            best_track = max(score.tracks, key=lambda t: len(t.notes))
            new_t = symusic.Track()
            new_t.program = best_track.program
            new_t.is_drum = best_track.is_drum
            # Truncate to first 400 notes for speed
            new_t.notes.extend(best_track.notes[:400])
            analysis_score.tracks.append(new_t)

        # Write truncated score to a temp file for music21 key analysis.
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            tmp = Path(f.name)
        analysis_score.dump_midi(str(tmp))
        
        with time_limit(10):
            stream = music21.converter.parse(str(tmp))
            key = stream.analyze("key")
            
        tmp.unlink(missing_ok=True)

        tonic = key.tonic.midi % 12  # 0=C, 2=D, ...

        if key.mode == "major":
            semitones = (0 - tonic) % 12  # target C (0)
        else:
            semitones = (9 - tonic) % 12  # target A (9) → A minor

        if semitones == 0:
            return score, False  # False = no transposition needed

        # Shift every note by semitones
        for track in score.tracks:
            for note in track.notes:
                note.pitch = max(0, min(127, note.pitch + semitones))
        return score, True  # True = transposed successfully

    except Exception as e:
        log.debug(f"music21 key detection failed for {midi_path}: {e}")
        return score, False  # return untransposed rather than crash


def chunk_sequence(ids: list[int], chunk_len: int, stride: int) -> list[list[int]]:
    """Sliding-window chunking. Returns a list of chunks."""
    chunks = []
    start = 0
    while start < len(ids):
        end = start + chunk_len
        chunk = ids[start:end]
        if len(chunk) >= chunk_len // 4:   # discard very short tails
            chunks.append(chunk)
        if end >= len(ids):
            break
        start += stride
    return chunks


# ---------------------------------------------------------------------------
# Worker function (called in parallel)
# ---------------------------------------------------------------------------

# Failure reason codes for diagnostics
FAIL_NOT_FOUND  = "not_found"
FAIL_NO_PIANO   = "no_piano"
FAIL_PARSE      = "parse_error"
FAIL_BAD_GENRE  = "bad_genre"
FAIL_TOKENIZE   = "tokenize_error"
FAIL_EMPTY      = "empty_chunks"
FAIL_TOO_LONG   = "too_long"

def process_one(row_dict):
    """
    Process a single MIDI file.
    Returns: dict with keys 'remi_chunks', 'octuple_chunks', 'genre_id'
             or a string failure reason code on failure.
    """
    global _midi_root, _remi_tok, _octuple_tok, _multi_instrument, _genre2id
    
    midi_path = resolve_midi_path(row_dict["file_path"], _midi_root)
    if midi_path is None:
        return FAIL_NOT_FOUND

    score_result = extract_midi_tracks(midi_path, _multi_instrument)
    if isinstance(score_result, str):
        return score_result
    score = score_result

    total_notes = sum(len(t.notes) for t in score.tracks)
    if total_notes > 50000:
        return FAIL_TOO_LONG

    score, _transposed = transpose_to_c(score, midi_path)

    genre_id = _genre2id.get(row_dict["genre"], -1)
    if genre_id < 0:
        return FAIL_BAD_GENRE

    try:
        with time_limit(40):
            remi_res = _remi_tok(score)
            remi_tokens = remi_res[0].ids if isinstance(remi_res, list) else remi_res.ids
            
            octuple_res = _octuple_tok(score)
            octuple_tokens = octuple_res[0].ids if isinstance(octuple_res, list) else octuple_res.ids
    except TimeoutException:
        log.warning(f"Tokenization timed out (took >40s) for {midi_path}")
        return FAIL_TOKENIZE
    except Exception as e:
        log.warning(f"Tokenization failed for {midi_path}: {e}")
        return FAIL_TOKENIZE

    remi_chunks = chunk_sequence(remi_tokens, REMI_CHUNK_LEN, REMI_CHUNK_STRIDE)
    octuple_chunks = chunk_sequence(octuple_tokens, OCTUPLE_CHUNK_LEN, OCTUPLE_CHUNK_STRIDE)

    if not remi_chunks and not octuple_chunks:
        return FAIL_EMPTY

    return {
        "remi_chunks": remi_chunks,
        "octuple_chunks": octuple_chunks,
        "genre_id": genre_id,
        "genre": row_dict["genre"],
        "file_path": row_dict["file_path"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_tokenizers(train_df: pd.DataFrame, midi_root: Path, sample_size: int, multi_instrument: bool, out_dir: Path):
    """
    Train the REMI + BPE and Octuple tokenizers.
    Returns (remi_tokenizer, octuple_tokenizer).
    """
    from miditok import REMI, Octuple, TokenizerConfig

    log.info("Building base tokenizer configs...")
    config = TokenizerConfig(
        num_velocities=16,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=multi_instrument,
        one_token_stream_for_programs=True,
        beat_res={(0, 4): 8, (4, 12): 4},
    )

    # Octuple requires max_bar_embedding in additional_params.
    # The default is 60 bars which is far too small for classical pieces.
    # We set it to 1024 to cover even the longest pieces in GigaMIDI.
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

    remi_tok = REMI(config)
    octuple_tok = Octuple(octuple_config)

    # ----- BPE training -----
    bpe_path = out_dir / "remi_bpe" / "tokenizer"
    bpe_path.mkdir(parents=True, exist_ok=True)

    if (bpe_path / "tokenizer.json").exists():
        log.info("Loading existing BPE tokenizer from disk...")
        remi_tok = REMI(params=bpe_path / "tokenizer.json")
    else:
        log.info(f"Training BPE on a {sample_size}-track subset of train split...")
        sample_df = train_df.sample(min(sample_size, len(train_df)), random_state=RANDOM_SEED)
        midi_paths = []
        for _, row in sample_df.iterrows():
            p = resolve_midi_path(row["file_path"], midi_root)
            if p is not None:
                midi_paths.append(p)
        log.info(f"  Found {len(midi_paths)} valid paths for BPE training subset.")
        remi_tok.train(vocab_size=BPE_VOCAB_SIZE, files_paths=midi_paths)
        remi_tok.save_pretrained(str(bpe_path))
        log.info(f"  BPE tokenizer saved to {bpe_path}.")

    # Save Octuple tokenizer config too
    oct_path = out_dir / "octuple" / "tokenizer"
    oct_path.mkdir(parents=True, exist_ok=True)
    octuple_tok.save_pretrained(str(oct_path))

    log.info(f"  REMI+BPE vocab size: {len(remi_tok)}")
    log.info(f"  Octuple vocab size : {len(octuple_tok)}")

    return remi_tok, octuple_tok


def process_split(
    df: pd.DataFrame,
    split_name: str,
    midi_root: Path,
    remi_tok,
    octuple_tok,
    workers: int,
    multi_instrument: bool,
    out_dir: Path,
    genre2id: dict,
):
    """
    Tokenize all rows in df, chunk, and save as HuggingFace datasets.
    """
    from datasets import Dataset, concatenate_datasets

    log.info(f"\n{'='*60}")
    log.info(f"Processing split: {split_name}  ({len(df)} songs)")
    log.info(f"{'='*60}")

    rows = df.to_dict("records")

    # Set globals so child processes inherit them via copy-on-write (fork), skipping massive IPC pickling.
    global _midi_root, _remi_tok, _octuple_tok, _multi_instrument, _genre2id
    _midi_root, _remi_tok, _octuple_tok = midi_root, remi_tok, octuple_tok
    _multi_instrument, _genre2id = multi_instrument, genre2id

    all_remi_ds = []
    all_oct_ds = []
    fail_counts: dict[str, int] = {}
    BATCH_SIZE = 10000

    # Create the pool ONCE for the whole split
    pool = Pool(workers) if workers > 1 else None

    for i in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[i : i + BATCH_SIZE]
        log.info(f"    Processing batch {i//BATCH_SIZE + 1} / {(len(rows) + BATCH_SIZE - 1) // BATCH_SIZE}...")
        
        if pool is None:
            results = [process_one(r) for r in batch_rows]
        else:
            results = pool.map(process_one, batch_rows)

        remi_records = []
        oct_records = []
        for res in results:
            if isinstance(res, str):  # failure reason code
                fail_counts[res] = fail_counts.get(res, 0) + 1
                continue
            for chunk in res["remi_chunks"]:
                remi_records.append({
                    "input_ids": chunk,
                    "genre_id": res["genre_id"],
                    "genre": res["genre"],
                })
            for chunk in res["octuple_chunks"]:
                oct_records.append({
                    "input_ids": chunk,
                    "genre_id": res["genre_id"],
                    "genre": res["genre"],
                })
                
        if remi_records:
            all_remi_ds.append(Dataset.from_list(remi_records))
        if oct_records:
            all_oct_ds.append(Dataset.from_list(oct_records))

    total_failed = sum(fail_counts.values())
    log.info(f"  Songs processed : {len(df) - total_failed:,}")
    log.info(f"  Songs failed    : {total_failed:,}")
    if fail_counts:
        log.info(f"  Failure breakdown:")
        for reason, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
            log.info(f"    {reason:<20}: {count:,}")

    # Save HF datasets
    remi_out = out_dir / "remi_bpe" / split_name
    oct_out = out_dir / "octuple" / split_name
    remi_out.mkdir(parents=True, exist_ok=True)
    oct_out.mkdir(parents=True, exist_ok=True)

    if all_remi_ds:
        final_remi_ds = concatenate_datasets(all_remi_ds)
        final_remi_ds.save_to_disk(str(remi_out))
        log.info(f"  REMI dataset saved to {remi_out}  ({final_remi_ds.num_rows:,} rows)")

    if all_oct_ds:
        final_oct_ds = concatenate_datasets(all_oct_ds)
        final_oct_ds.save_to_disk(str(oct_out))
        log.info(f"  Octuple dataset saved to {oct_out}  ({final_oct_ds.num_rows:,} rows)")

    if pool is not None:
        pool.close()
        pool.join()


def main():
    notify_endpoint = os.getenv("NOTIFY_ENDPOINT")
    notify = Notify(endpoint=notify_endpoint) if (Notify is not None and notify_endpoint) else None
    parser = argparse.ArgumentParser(description="Tokenize GigaMIDI splits.")
    parser.add_argument("--multi-instrument", action="store_true",
                        help="If set, tokenizes all non-drum tracks with Program tokens (all_instrument_split).")
    parser.add_argument("--sample", type=int, default=0,
                        help="If >0, only use this many rows per split (for local testing).")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1),
                        help="Number of parallel workers.")
    parser.add_argument("--bpe-sample", type=int, default=BPE_TRAIN_SAMPLE,
                        help="Number of train tracks to use for BPE training.")
    parser.add_argument("--dataset-name", type=str, default="gt_1000_with_ties",
                        help="Subfolder in datasets_metadata (e.g. gt_1000_with_ties, top_5_no_ties)")
    args = parser.parse_args()

    domain = "multi-instrument" if args.multi_instrument else "piano"
    splits_dir = Path(os.getenv("MUSICGEN_SPLITS_DIR", PROJECT_ROOT / "datasets_metadata" / args.dataset_name / "split" / domain))
    out_dir = Path(os.getenv("MUSICGEN_OUTPUT_DIR", PROJECT_ROOT / "datasets_metadata" / args.dataset_name / "dataset" / domain))

    if notify:
        try:
            notify.send(f'Tokenization started for: {splits_dir}')
        except Exception as e:
            log.warning(f"Failed to send notification: {e}")
    
    log.info("Loading CSVs...")
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df   = pd.read_csv(splits_dir / "val.csv")
    test_df  = pd.read_csv(splits_dir / "test.csv")

    if args.sample > 0:
        log.info(f"[SAMPLE MODE] Truncating each split to {args.sample} rows.")
        train_df = train_df.sample(min(args.sample, len(train_df)), random_state=RANDOM_SEED)
        val_df   = val_df.sample(min(args.sample, len(val_df)), random_state=RANDOM_SEED)
        test_df  = test_df.sample(min(args.sample, len(test_df)), random_state=RANDOM_SEED)

    log.info(f"Split sizes — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")
    log.info(f"Workers: {args.workers}")

    midi_root = RAW_MIDI_ROOT

    t0 = time.time()
    remi_tok, octuple_tok = build_tokenizers(train_df, midi_root, args.bpe_sample, args.multi_instrument, out_dir)

    # ---------------------------------------------------------------------------
    # Dynamically build GENRE2ID from the train split
    # ---------------------------------------------------------------------------
    unique_genres = train_df["genre"].dropna().unique()
    genre_labels = sorted(list(set(str(g).strip() for g in unique_genres if str(g).strip())))
    genre2id = {g: i for i, g in enumerate(genre_labels)}
    log.info(f"Dynamically mapped {len(genre2id)} genres from train split: {genre_labels}")

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        process_split(df, split_name, midi_root, remi_tok, octuple_tok, args.workers, args.multi_instrument, out_dir, genre2id)

    log.info(f"\nAll done in {(time.time()-t0)/60:.1f} min.")
    if notify:
        try:
            notify.send(f'Tokenization finished for: {splits_dir}')
        except Exception as e:
            log.warning(f"Failed to send notification: {e}")


if __name__ == "__main__":
    main()
