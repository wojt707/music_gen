import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Delete unlisted MIDI files to free up space.")
    parser.add_argument("--data-dir", type=str, 
                        default=os.getenv("DATA_DIR", str(repo_root / "datasets_metadata")),
                        help="Path to datasets_metadata directory")
    parser.add_argument("--raw-dir", type=str, 
                        default=os.getenv("RAW_DATASET_DIR", str(repo_root / "raw_dataset")),
                        help="Path to raw_dataset directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "gigamidi_artist_prepared.csv"
    raw_dir = Path(args.raw_dir)

    if not csv_path.exists():
        print(f"Error: Could not find CSV at {csv_path}")
        return

    print(f"Loading valid paths from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Store valid absolute paths in a set for O(1) lookup
    valid_paths = set()
    for fp in df['file_path'].dropna():
        # fp is like "./Final_GigaMIDI_V1.1_Final/..."
        if fp.startswith("./"):
            fp = fp[2:]
        
        # We use .resolve() to get absolute canonical paths so exact string matching works
        full_path = (raw_dir / fp).resolve()
        valid_paths.add(str(full_path))
        
    print(f"Found {len(valid_paths)} valid files referenced in CSV.")
    
    target_dir = raw_dir / "Final_GigaMIDI_V1.1_Final"
    if not target_dir.exists():
        print(f"Error: Could not find raw dataset directory at {target_dir}")
        return
        
    print(f"Scanning {target_dir} for unlisted .mid files...")
    
    deleted_count = 0
    kept_count = 0
    bytes_freed = 0
    
    # Walk through the target directory to find all .mid and .midi files
    all_midis = list(target_dir.rglob("*.mid")) + list(target_dir.rglob("*.midi"))
    
    for midi_file in tqdm(all_midis, desc="Checking files"):
        # Get the absolute canonical path
        midi_str = str(midi_file.resolve())
        
        if midi_str not in valid_paths:
            # Not in the CSV -> Delete it
            bytes_freed += midi_file.stat().st_size
            midi_file.unlink()
            deleted_count += 1
        else:
            kept_count += 1
            
    print(f"\nCleanup complete!")
    print(f"Kept:    {kept_count:,} files")
    print(f"Deleted: {deleted_count:,} files")
    print(f"Space freed: {bytes_freed / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
