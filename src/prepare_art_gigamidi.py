import argparse
import sys
import time
import re
import os
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from notify_run import Notify

import pandas as pd

try:
    import symusic
except ImportError:
    print("[FATAL] 'symusic' is not installed. Please activate the venv.", file=sys.stderr)
    sys.exit(1)

# ── MIDI programs considered "piano" (0-indexed, General MIDI) ─────────────
PIANO_PROGRAMS = set(range(8))  # 0–7 inclusive
DRUM_CHANNEL   = 9              # 0-indexed; channel 10 in 1-indexed MIDI spec

def heavy_normalize(name):
    """
    Aggressively normalizes the artist name for maximum match rate.
    """
    if pd.isna(name):
        return name
    name = str(name).lower()
    
    # Remove accents (e.g., Beyoncé -> beyonce)
    name = ''.join(c for c in unicodedata.normalize('NFD', name)
                  if unicodedata.category(c) != 'Mn')
                  
    # Handle "Last, First" -> "First Last"
    if ',' in name:
        parts = name.split(',', 1)
        if len(parts) == 2:
            name = parts[1].strip() + ' ' + parts[0].strip()
            
    # Remove text in parentheses/brackets (e.g., "Nirvana (Band)" -> "Nirvana")
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\[.*?\]', '', name)
    
    # Replace & with and
    name = name.replace('&', ' and ')
    
    # Remove all punctuation (keep alphanumeric and spaces)
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    
    # Remove "the " from the beginning
    name = re.sub(r'^the\s+', '', name)
    
    # Remove extra spaces
    name = ' '.join(name.split())
    return name


def count_tracks(midi_path: str) -> tuple[int, int, int, int]:
    """Return (piano_track_count, total_piano_notes, total_track_count, total_notes)."""
    try:
        score = symusic.Score(midi_path)
        piano_count = 0
        total_piano_notes = 0
        total_count = 0
        total_notes = 0
        for track in score.tracks:
            if not track.is_drum and len(track.notes) > 0:
                total_count += 1
                total_notes += len(track.notes)
                if track.program in PIANO_PROGRAMS:
                    piano_count += 1
                    total_piano_notes += len(track.notes)
        return piano_count, total_piano_notes, total_count, total_notes
    except Exception:
        return -1, 0, -1, 0  # parse error


def resolve_path(file_path: str, midi_root: Path | None) -> str:
    """
    Directly appends the relative file_path from the CSV to midi_root.
    """
    p = Path(file_path)
    if p.is_absolute():
        return str(p)
    if midi_root is not None:
        return str(midi_root / p)
    return str(p)


def worker(args):
    """Top-level function (picklable for ProcessPoolExecutor)."""
    idx, file_path, midi_root_str = args
    midi_root = Path(midi_root_str) if midi_root_str else None
    resolved = resolve_path(file_path, midi_root)
    
    if not Path(resolved).exists():
        return idx, -2, 0, -2, 0  # -2 = file not found
        
    piano_count, total_piano_notes, total_count, total_notes = count_tracks(resolved)
    return idx, piano_count, total_piano_notes, total_count, total_notes


def main():
    parser = argparse.ArgumentParser(description="Match artists and filter GigaMIDI tracks in one go.")
    parser.add_argument("--gigamidi",
                        help="Path to the raw GigaMIDI CSV.")
    parser.add_argument("--artists",
                        help="Path to the Global Music Artists CSV.")
    parser.add_argument("--output", default="gigamidi_artist_prepared.csv",
                        help="Output prepared CSV path.")
    parser.add_argument("--midi-root", default=None,
                        help="Root directory prepended to relative file_path values")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes for MIDI scanning")
    args = parser.parse_args()

    # 1. Load and Heavy Normalize Global Music Artists
    print("1. Loading and Heavily Normalizing Global Music Artists...")
    df_artists = pd.read_csv(args.artists, usecols=['artist_name', 'artist_genre'], on_bad_lines='skip')
    df_artists['clean_name'] = df_artists['artist_name'].apply(heavy_normalize)
    df_artists = df_artists.dropna(subset=['clean_name'])
    df_artists = df_artists.drop_duplicates(subset=['clean_name'])
    
    global_artist_name_map = df_artists.set_index('clean_name')['artist_name'].to_dict()
    global_genres_map = df_artists.set_index('clean_name')['artist_genre'].to_dict()
    print(f"   Loaded {len(df_artists):,} unique normalized artists.\n")

    # 2. Load raw GigaMIDI CSV
    print("2. Loading GigaMIDI dataset...")
    cols_to_use = ['file_path', 'title', 'artist']
    df_giga = pd.read_csv(args.gigamidi, usecols=cols_to_use, low_memory=False, on_bad_lines='skip')
    initial_len = len(df_giga)
    print(f"   Loaded {initial_len:,} rows from GigaMIDI.\n")

    # 3. Drop rows without artists
    print("3. Filtering data...")
    df_giga['artist'] = df_giga['artist'].replace(r'^\s*$', float('nan'), regex=True)
    df_giga = df_giga.dropna(subset=['artist'])
    filtered_len = len(df_giga)
    print(f"   Rows remaining after keeping those with an artist: {filtered_len:,}\n")

    # 4. Exact Matching by artist name
    print("4. Exact Matching...")
    df_giga['clean_artist'] = df_giga['artist'].apply(heavy_normalize)
    df_giga['global_artist_name'] = df_giga['clean_artist'].map(global_artist_name_map)
    df_giga['matched_genres'] = df_giga['clean_artist'].map(global_genres_map)
    
    exact_matches = df_giga['matched_genres'].notna().sum()
    print(f"   Exact match successfully labeled {exact_matches:,} tracks.\n")

    # 5. Normalize genre column
    print("5. Normalizing genre strings...")
    df_matched = df_giga.dropna(subset=['matched_genres']).copy()
    df_matched['matched_genres'] = df_matched['matched_genres'].str.lower().str.strip()
    df_matched['matched_genres'] = df_matched['matched_genres'].str.replace(r'\s*,\s*', ', ', regex=True)

    # 6. Deduplicate by normalized (artist, title) pair
    print("6. Deduplicating...")
    final_cols = ['file_path', 'title', 'artist', 'global_artist_name', 'matched_genres']
    df_save = df_matched[final_cols].copy()
    before_dedup = len(df_save)
    df_save = df_save.drop_duplicates(subset=['title', 'global_artist_name'])
    print(f"   Dropped {before_dedup - len(df_save):,} duplicate songs (by title + global_artist_name).")
    print(f"   {len(df_save):,} unique tracks remaining.\n")

    # 7. Scan every MIDI file for track/note metrics
    print(f"7. Scanning {len(df_save):,} MIDI files with {args.workers} workers...")
    tasks = [(i, fp, args.midi_root) for i, fp in enumerate(df_save["file_path"])]
    
    piano_counts = [0] * len(df_save)
    total_piano_notes_arr = [0] * len(df_save)
    total_counts = [0] * len(df_save)
    total_notes_arr = [0] * len(df_save)
    
    errors = 0
    missing_files = 0
    done = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx, p_count, p_notes, t_count, t_notes = future.result()
            piano_counts[idx] = p_count
            total_piano_notes_arr[idx] = p_notes
            total_counts[idx] = t_count
            total_notes_arr[idx] = t_notes
            done += 1
            
            if p_count == -1:
                errors += 1
            elif p_count == -2:
                missing_files += 1
                
            if done % 1000 == 0 or done == len(df_save):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done:>6,}/{len(df_save):,}] {done/len(df_save)*100:5.1f}% | {rate:,.0f} files/s | missing: {missing_files:,}, corrupted: {errors:,}")

    df_save["piano_track_count"] = piano_counts
    df_save["total_piano_notes"] = total_piano_notes_arr
    df_save["total_track_count"] = total_counts
    df_save["total_notes"] = total_notes_arr

    # 8. Drop invalid files (missing, corrupted, no pitched tracks)
    print("\n8. Filtering invalid files...")
    has_any_track = df_save["total_track_count"] >= 1
    df_filtered = df_save[has_any_track].copy()
    print(f"   Dropped {len(df_save) - len(df_filtered):,} invalid files (missing/corrupted/empty).")

    # 9. Save gigamidi_artist_prepared.csv
    print(f"\n9. Saving final prepared dataset...")
    df_filtered.to_csv(args.output, index=False)
    print(f"   Saved {len(df_filtered):,} valid songs to '{args.output}'")
    print(f"\nExecution finished in {time.time() - t0:.1f} seconds.")


if __name__ == "__main__":
    notify_endpoint = os.getenv("NOTIFY_ENDPOINT")
    notify = None
    if notify_endpoint and Notify is not None:
        try:
            notify = Notify(endpoint=notify_endpoint)
            notify.send('Artist Prepare started')
        except Exception:
            pass
    main()
    if notify is not None:
        try:
            notify.send('Artist Prepare finished')
        except Exception:
            pass
