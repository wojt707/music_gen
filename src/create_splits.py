"""
create_splits.py
----------------
Reads gigamidi_artist_prepared.csv and creates stratified 80/10/10 artist-level
Train / Val / Test splits for both piano and multi-instrument configurations.

Post-processing (applied before splitting):
  - Cap each artist to ARTIST_CAP rows (prevents single prolific composers
    dominating the dataset; unrelated to genre imbalance which is handled
    at training time via weighted sampling)

Splitting algorithm:
  Greedy notes-weighted stratified split by ARTIST (not by song).
  - Artists are grouped by their primary genre (genre with most songs).
  - Within each group, artists are sorted by their total note count.
  - Each artist is assigned greedily to the split whose current proportion
    of total musical content is furthest below its target (80/10/10).
  This guarantees that NO artist appears in more than one split (preventing
  style leakage), while keeping note-content volume roughly balanced.

Genre weights (for weighted sampling during training):
  Saved as genre_weights.json alongside each split. Formula:
    weight_i = N_total / (N_classes * count_i)
  This ensures that all genres are represented equally in expectation
  when the trainer samples with per-sample weights.

Output folder structure:
  datasets_metadata/
  └── split/
      ├── piano/
      │   ├── train.csv
      │   ├── val.csv
      │   ├── test.csv
      │   └── genre_weights.json
      └── multi-instrument/
          ├── train.csv
          ├── val.csv
          ├── test.csv
          └── genre_weights.json

Usage:
    python create_splits.py [--input datasets_metadata/gigamidi_artist_prepared.csv] [--output-dir datasets_metadata]
"""

import argparse
import json
import os
import sys

import pandas as pd
from genre_bucket_map import map_genres_to_buckets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_GENRE_TRACKS = 1000   # genres below this threshold are discarded
ARTIST_CAP       = 300    # max songs per artist (prevents Bach / Beethoven domination)

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10  (implicit: whatever remains)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_stats(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Rows              : {len(df):>10,}")
    if "artist" in df.columns:
        print(f"  Unique artists    : {df['artist'].nunique():>10,}")
    if "genre" in df.columns:
        valid = df["genre"].dropna()
        print(f"  Unique genres     : {valid.nunique():>10,}")
        print(f"  Top 10 genres:")
        for g, c in valid.value_counts().head(10).items():
            print(f"    {g:<35} {c:>8,}")
    print()


def compute_genre_weights(df: pd.DataFrame) -> dict:
    """
    Inverse-frequency class weights, normalised so that the average weight = 1.
    Formula: weight_i = N_total / (N_classes * count_i)
    """
    counts = df["genre"].value_counts()
    n_total   = len(df)
    n_classes = len(counts)
    weights = {genre: round(n_total / (n_classes * count), 6)
               for genre, count in counts.items()}
    return weights


def greedy_artist_split(df: pd.DataFrame, notes_col: str) -> pd.DataFrame:
    """
    Assigns each unique artist to exactly one of train/val/test using a
    greedy notes-weighted algorithm.

    Returns df with a new 'split' column added.
    """
    # Primary genre for each artist = the genre they have the most songs in
    artist_primary = (
        df.groupby("artist")["genre"]
        .apply(lambda x: x.value_counts().index[0])
    )

    # Total notes per artist (proxy for musical content volume)
    artist_notes = df.groupby("artist")[notes_col].sum().to_dict()

    # Group artists by primary genre
    genre_to_artists: dict[str, list] = {}
    for artist, genre in artist_primary.items():
        genre_to_artists.setdefault(genre, []).append(artist)

    train_artists: set = set()
    val_artists:   set = set()
    test_artists:  set = set()

    for genre, artists in genre_to_artists.items():
        # Sort largest-first so big artists get placed deterministically
        artists_sorted = sorted(artists, key=lambda a: artist_notes.get(a, 0), reverse=True)

        train_notes = val_notes = test_notes = 0

        for artist in artists_sorted:
            notes = artist_notes.get(artist, 0)
            total = train_notes + val_notes + test_notes

            if total == 0:
                # First artist always goes to train
                train_artists.add(artist)
                train_notes += notes
                continue

            # Compute how far each split is from its target proportion
            diff_train = TRAIN_RATIO - train_notes / total
            diff_val   = VAL_RATIO   - val_notes   / total
            diff_test  = (1.0 - TRAIN_RATIO - VAL_RATIO) - test_notes / total

            max_diff = max(diff_train, diff_val, diff_test)
            if max_diff == diff_train:
                train_artists.add(artist)
                train_notes += notes
            elif max_diff == diff_val:
                val_artists.add(artist)
                val_notes += notes
            else:
                test_artists.add(artist)
                test_notes += notes

    df = df.copy()
    df["split"] = df["artist"].map(
        lambda a: "train" if a in train_artists
                  else ("val" if a in val_artists else "test")
    )

    print(f"  Artist assignment: "
          f"train={len(train_artists):,}  "
          f"val={len(val_artists):,}  "
          f"test={len(test_artists):,}")
    return df


def make_split(df: pd.DataFrame, notes_col: str, out_dir: str) -> None:
    """
    Runs the full split pipeline for one instrument configuration and
    saves results to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    print_stats(df, f"Input for split ({os.path.basename(out_dir)})")

    # -- Genre weights (computed on the full set before splitting) -----------
    weights = compute_genre_weights(df)
    weights_path = os.path.join(out_dir, "genre_weights.json")
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    print(f"  Genre weights saved to {weights_path}")
    print(f"  Weight range: {min(weights.values()):.3f} – {max(weights.values()):.3f}")

    # Attach per-row class weight
    df = df.copy()
    df["class_weight"] = df["genre"].map(weights)

    # -- Greedy artist-level split -------------------------------------------
    print("\n  Running greedy artist-level split...")
    df = greedy_artist_split(df, notes_col)

    # -- Verify no artist leakage --------------------------------------------
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        a1 = set(df[df["split"] == s1]["artist"])
        a2 = set(df[df["split"] == s2]["artist"])
        overlap = a1 & a2
        if overlap:
            print(f"  [WARNING] {len(overlap)} artists appear in both {s1} and {s2}!")
        else:
            print(f"  [OK] No artist overlap between {s1} and {s2}.")

    # -- Print split stats ---------------------------------------------------
    print("\n" + "=" * 75)
    print("  PER-GENRE SPLIT STATISTICS")
    print("=" * 75)
    print(f"  {'Genre':<20} {'Total':>7} {'Train':>7} {'Val':>6} {'Test':>6}  {'Train%':>7} {'Val%':>6} {'Test%':>6}")
    print("  " + "-" * 73)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    all_genres = sorted(df['genre'].unique())
    
    for g in all_genres:
        total = (df['genre'] == g).sum()
        tr    = (train_df['genre'] == g).sum()
        vl    = (val_df['genre'] == g).sum()
        te    = (test_df['genre'] == g).sum()
        
        ptr = (tr/total*100) if total > 0 else 0
        pvl = (vl/total*100) if total > 0 else 0
        pte = (te/total*100) if total > 0 else 0
        
        print(f"  {g:<20} {total:>7,} {tr:>7,} {vl:>6,} {te:>6,}  {ptr:>6.1f}% {pvl:>5.1f}% {pte:>5.1f}%")
        
    print("=" * 75)
    
    print("\n  TOTALS:")
    total_tr = len(train_df)
    total_vl = len(val_df)
    total_te = len(test_df)
    total_all = len(df)
    print(f"    Train : {total_tr:>7,} rows ({total_tr/total_all*100:.1f}%)")
    print(f"    Val   : {total_vl:>7,} rows ({total_vl/total_all*100:.1f}%)")
    print(f"    Test  : {total_te:>7,} rows ({total_te/total_all*100:.1f}%)")
    print(f"    Unique artists : {df['artist'].nunique():,}\n")

    # -- Save CSVs -----------------------------------------------------------
    out_cols = [
        "file_path", "genre", "title", "artist",
        "piano_track_count", "total_piano_notes",
        "total_track_count", "total_notes",
        "class_weight",
    ]
    for split_name in ("train", "val", "test"):
        sub = df[df["split"] == split_name][out_cols].copy()
        path = os.path.join(out_dir, f"{split_name}.csv")
        sub.to_csv(path, index=False)
        print(f"  Saved {len(sub):,} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input = os.path.join(repo_root, "datasets_metadata", "gigamidi_artist_prepared.csv")
    if not os.path.exists(default_input):
        default_input = "gigamidi_artist_prepared.csv"
    default_output = os.path.join(repo_root, "datasets_metadata")

    parser = argparse.ArgumentParser(
        description="Create 80/10/10 artist-level splits from gigamidi_artist_prepared.csv."
    )
    parser.add_argument(
        "--input",
        default=default_input,
        help="Path to gigamidi_artist_prepared.csv (output of prepare_art_gigamidi.py).",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output,
        help="Root output directory. Splits saved under <output-dir>/split/.",
    )
    parser.add_argument(
        "--no-ties", action="store_true",
        help="If set, tracks resolved via priority tie-breaker are dropped (creates high-confidence dataset).",
    )
    parser.add_argument(
        "--top-n-genres", type=int, default=0,
        help="If set > 0, keeps only the top N genres by song count (e.g. 5) instead of all > MIN_GENRE_TRACKS.",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")

    # ── STEP 1: Load ────────────────────────────────────────────────────────
    print(f"\n[STEP 1] Loading {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  Loaded {len(df):,} rows.")
    print_stats(df, "After loading")

    # ── STEP 2: Map and post-process genres ─────────────────────────────────
    print("[STEP 2] Mapping and post-processing genres...")
    if args.no_ties:
        print("  [FLAG] --no-ties is SET. Discarding tie-break resolutions.")

    # Map comma-separated strings to a single broad bucket
    df['genre'] = df['matched_genres'].apply(
        lambda x: map_genres_to_buckets(x, allow_ties=not args.no_ties) if isinstance(x, str) else None
    )
    
    # Drop rows that couldn't be mapped to any bucket
    before_map = len(df)
    df = df[df['genre'].notna()].copy()
    print(f"  Rows unmapped/dropped (including disabled ties): {before_map - len(df):,}")

    # Drop genres below threshold (or filter to Top N)
    genre_counts = df["genre"].value_counts()
    
    if args.top_n_genres > 0:
        print(f"  [FLAG] --top-n-genres={args.top_n_genres}. Keeping top {args.top_n_genres} genres.")
        keep_genres = set(genre_counts.nlargest(args.top_n_genres).index)
        drop_genres = set(genre_counts.index) - keep_genres
    else:
        keep_genres = set(genre_counts[genre_counts >= MIN_GENRE_TRACKS].index)
        drop_genres = set(genre_counts[genre_counts <  MIN_GENRE_TRACKS].index)
        
    before = len(df)
    df = df[df["genre"].isin(keep_genres)].copy()
    print(f"\n  Genres kept  ({len(keep_genres):,}): {sorted(keep_genres)}")
    print(f"  Genres dropped ({len(drop_genres):,}): {sorted(drop_genres)}")
    print(f"  Rows dropped from genre filtering: {before - len(df):,}")
    print_stats(df, "After genre post-processing")

    # ── STEP 3: Per-artist row cap ───────────────────────────────────────────
    print(f"[STEP 3] Capping each artist to {ARTIST_CAP} rows...")
    before = len(df)
    df = (
        df.sample(frac=1, random_state=RANDOM_SEED)   # shuffle first for random selection
          .groupby("artist", sort=False)
          .head(ARTIST_CAP)
          .reset_index(drop=True)
    )
    print(f"  Dropped {before - len(df):,} rows from artist cap.")
    print_stats(df, "After artist cap")

    # ── STEP 4: Piano split ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIANO SPLIT")
    print("=" * 60)
    piano_df = df[
        (df["piano_track_count"] >= 1) &
        (df["total_piano_notes"] > 0)
    ].copy()
    print(f"  Piano-eligible rows: {len(piano_df):,}")
    piano_out = os.path.join(args.output_dir, "split", "piano")
    make_split(piano_df, notes_col="total_piano_notes", out_dir=piano_out)

    # ── STEP 5: Multi-instrument split ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MULTI-INSTRUMENT SPLIT")
    print("=" * 60)
    multi_df = df[
        (df["total_track_count"] >= 1) &
        (df["total_notes"] > 0)
    ].copy()
    print(f"  Multi-instrument-eligible rows: {len(multi_df):,}")
    multi_out = os.path.join(args.output_dir, "split", "multi-instrument")
    make_split(multi_df, notes_col="total_notes", out_dir=multi_out)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
