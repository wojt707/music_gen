import os
import pandas as pd
import time
import sys
from pathlib import Path
from genre_bucket_map import map_genres_to_buckets

# Replace unencodable characters with '?' to prevent PowerShell piping errors (mojibake)
sys.stdout.reconfigure(errors='replace')

# Input must be the filtered CSV produced by filter_tracks.py.
# If this file doesn't exist yet, run filter_tracks.py first (on Ares).
DEFAULT_INPUT_PATH = os.getenv("INPUT_CSV", "datasets_metadata/gigamidi_filtered.csv")
ARTIST_CAP = 300
GENRE_MIN_TRACKS = 1000
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

ARTISTS_TO_DROP = {'anonymous', 'tv themes'}


def print_stats(df, label, genre_col='matched_genres'):
    print(f"\n{'='*55}")
    print(f"  STATS: {label}")
    print(f"{'='*55}")
    print(f"  Total rows          : {len(df):>10,}")
    if 'global_artist_name' in df.columns:
        print(f"  Unique artists      : {df['global_artist_name'].nunique():>10,}")
    if genre_col in df.columns:
        # Count genres — handle multi-genre strings (before explode) by counting unique values
        if df[genre_col].str.contains(',', na=False).any():
            genres_series = df[genre_col].dropna().str.split(',').explode().str.strip()
        else:
            genres_series = df[genre_col].dropna().str.strip()
        genres_series = genres_series[genres_series != '']
        print(f"  Unique genres       : {genres_series.nunique():>10,}")
        top5 = genres_series.value_counts().head(5)
        print(f"  Top 5 genres:")
        for g, c in top5.items():
            print(f"    {g:<30} {c:>8,}")
    if 'global_artist_name' in df.columns:
        top5_artists = df['global_artist_name'].value_counts().head(5)
        print(f"  Top 5 artists by song count:")
        for a, c in top5_artists.items():
            print(f"    {a:<35} {c:>6,}")
    if 'title' in df.columns and 'global_artist_name' in df.columns:
        top5_songs = df.groupby(['title', 'global_artist_name']).size().nlargest(5)
        print(f"  Top 5 most-exploded songs (by title + artist):")
        for (t, a), c in top5_songs.items():
            label = f"{str(t)} - {str(a)}"
            print(f"    {label:<55} {c:>6,}")
    print(f"{'='*55}\n")


def main():
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Create per-artist Train/Val/Test splits.")
    parser.add_argument("--multi-instrument", action="store_true", 
                        help="If set, creates splits for all pitched instruments (no piano filter), outputting to all_instrument_split.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to filtered CSV.")
    args = parser.parse_args()

    if args.multi_instrument:
        OUTPUT_DIR = os.getenv("OUTPUT_DIR", "datasets_metadata/all_instrument_split")
        notes_col = 'total_notes'
        filter_col = 'total_track_count'
    else:
        OUTPUT_DIR = os.getenv("OUTPUT_DIR", "datasets_metadata/piano_split")
        notes_col = 'total_piano_notes'
        filter_col = 'piano_track_count'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------
    # STEP 1: Load
    # -------------------------------------------------------
    print(f"\n[STEP 1] Loading dataset from {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    print_stats(df, "After loading")

    # -------------------------------------------------------
    # STEP 2: Enforce instrument filter
    # -------------------------------------------------------
    print(f"[STEP 2] Enforcing filter ({filter_col} >= 1)...")
    if filter_col not in df.columns:
        print(f"  [WARNING] '{filter_col}' column not found!")
        print("  This CSV was not produced correctly by filter_tracks.py.")
        raise SystemExit(1)
    before = len(df)
    df = df[df[filter_col] >= 1].copy()
    print(f"  Dropped {before - len(df):,} songs that didn't meet the filter.")
    print_stats(df, "After filter")

    # -------------------------------------------------------
    # STEP 3: Drop placeholder artists
    # -------------------------------------------------------
    print("[STEP 3] Dropping placeholder artists (Anonymous, TV Themes)...")
    before = len(df)
    mask = df['global_artist_name'].str.lower().str.strip().isin(ARTISTS_TO_DROP)
    df = df[~mask].copy()
    print(f"  Dropped {before - len(df):,} rows ({before - len(df)} matched placeholder names)")
    print_stats(df, "After dropping placeholders")

    # -------------------------------------------------------
    # STEP 4: Map to broad buckets and explode
    # -------------------------------------------------------
    print("\n[STEP 4] Mapping to broad buckets and exploding...")
    before = len(df)
    
    # Map comma-separated strings to a single broad bucket
    df['genre'] = df['matched_genres'].apply(
        lambda x: map_genres_to_buckets(x) if isinstance(x, str) else None
    )
    
    # Drop rows that couldn't be mapped to any bucket
    df = df[df['genre'].notna()].copy()
    
    print(f"  Rows before explode: {before:,}  →  after: {len(df):,}")
    print_stats(df, "After exploding into broad buckets", genre_col='genre')

    # -------------------------------------------------------
    # STEP 5: Drop genres below threshold
    # -------------------------------------------------------
    print(f"[STEP 5] Dropping genres with fewer than {GENRE_MIN_TRACKS:,} tracks...")
    genre_counts = df['genre'].value_counts()
    genres_kept = genre_counts[genre_counts >= GENRE_MIN_TRACKS].index
    genres_dropped = genre_counts[genre_counts < GENRE_MIN_TRACKS].index
    before = len(df)
    df = df[df['genre'].isin(genres_kept)].copy()
    print(f"  Genres kept   : {len(genres_kept):,}")
    print(f"  Genres dropped: {len(genres_dropped):,}")
    print(f"  Rows dropped  : {before - len(df):,}")
    print_stats(df, f"After dropping genres < {GENRE_MIN_TRACKS}", genre_col='genre')

    # -------------------------------------------------------
    # STEP 6: Cap to 300 total rows per artist
    # -------------------------------------------------------
    print(f"[STEP 6] Capping to {ARTIST_CAP} total rows per artist (AFTER explosion)...")
    before = len(df)
    # Shuffle first so head(N) gives a random sample, not just the first N rows
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df = df.groupby('global_artist_name', sort=False).head(ARTIST_CAP).reset_index(drop=True)
    print(f"  Dropped {before - len(df):,} rows due to artist cap")
    print_stats(df, f"After capping at {ARTIST_CAP} total rows/artist", genre_col='genre')


    print("\n  Full genre distribution after threshold filter:")
    print("  " + "-" * 45)
    for g, c in df['genre'].value_counts().items():
        print(f"  {g:<35} {c:>8,}")
    print("  " + "-" * 45)

    # -------------------------------------------------------
    # STEP 7: Stratified Artist-level Train / Val / Test split
    # -------------------------------------------------------
    print("\n[STEP 7] Stratified 80/10/10 splitting by global_artist_name...")

    import random
    random.seed(RANDOM_SEED)
    
    # 1. Find the primary genre for each artist (the one they have the most tracks in)
    artist_primary_genre = df.groupby('global_artist_name', sort=False)['genre'].apply(
        lambda x: x.value_counts().index[0]
    )
    
    # 2. Group artists by their primary genre
    genre_to_artists = {}
    for artist, genre in artist_primary_genre.items():
        if genre not in genre_to_artists:
            genre_to_artists[genre] = []
        genre_to_artists[genre].append(artist)
        
    train_artists = set()
    val_artists = set()
    test_artists = set()
    
    # Calculate each artist's total notes across all their kept songs
    artist_notes = df.groupby('global_artist_name', sort=False)[notes_col].sum().to_dict()
    
    # 3. For each genre bucket, sort artists by notes and distribute greedily
    for genre, artists in genre_to_artists.items():
        artists_sorted = sorted(artists, key=lambda a: artist_notes.get(a, 0), reverse=True)
        
        train_notes = 0
        val_notes = 0
        test_notes = 0
        
        for artist in artists_sorted:
            notes = artist_notes.get(artist, 0)
            
            total_current = train_notes + val_notes + test_notes
            if total_current == 0:
                train_artists.add(artist)
                train_notes += notes
                continue
                
            p_train = train_notes / total_current
            p_val   = val_notes / total_current
            p_test  = test_notes / total_current
            
            diff_train = TRAIN_RATIO - p_train
            diff_val   = VAL_RATIO - p_val
            diff_test  = (1.0 - TRAIN_RATIO - VAL_RATIO) - p_test
            
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
        
    print(f"  Unique artists → train: {len(train_artists):,} | val: {len(val_artists):,} | test: {len(test_artists):,}")

    df['split'] = df['global_artist_name'].map(
        lambda a: 'train' if a in train_artists else ('val' if a in val_artists else 'test')
    )

    # -------------------------------------------------------
    # STEP 8: Save to disk
    # -------------------------------------------------------
    print(f"\n[STEP 8] Saving splits to {OUTPUT_DIR}...")
    
    train_df = df[df['global_artist_name'].isin(train_artists)].copy()
    val_df   = df[df['global_artist_name'].isin(val_artists)].copy()
    test_df  = df[df['global_artist_name'].isin(test_artists)].copy()
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    
    print("Done!")

    # -------------------------------------------------------
    # STEP 9: Per-genre statistics
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PER-GENRE SPLIT STATISTICS")
    print("=" * 70)
    print(f"  {'Genre':<30} {'Total':>7} {'Train':>7} {'Val':>6} {'Test':>6}  {'Train%':>7} {'Val%':>6} {'Test%':>6}")
    print("  " + "-" * 68)

    all_genres = sorted(df['genre'].unique())
    for g in all_genres:
        total = (df['genre'] == g).sum()
        tr    = (train_df['genre'] == g).sum()
        vl    = (val_df['genre'] == g).sum()
        te    = (test_df['genre'] == g).sum()
        
        ptr = (tr/total*100) if total > 0 else 0
        pvl = (vl/total*100) if total > 0 else 0
        pte = (te/total*100) if total > 0 else 0
        
        print(f"  {g:<30} {total:>7,} {tr:>7,} {vl:>6,} {te:>6,}  {ptr:>6.1f}% {pvl:>5.1f}% {pte:>5.1f}%")
        
    print("=" * 70)
    
    print("\n  TOTALS:")
    total_tr = len(train_df)
    total_vl = len(val_df)
    total_te = len(test_df)
    total_all = len(df)
    print(f"    Train : {total_tr:>7,} rows ({total_tr/total_all*100:.1f}%)")
    print(f"    Val   : {total_vl:>7,} rows ({total_vl/total_all*100:.1f}%)")
    print(f"    Test  : {total_te:>7,} rows ({total_te/total_all*100:.1f}%)")
    print("\nFinished successfully.\n")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nDone in {time.time() - start:.1f} seconds.")
