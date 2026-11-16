"""
tokenize_moonbeam.py

Custom wrapper for Moonbeam tokenization.
Reads our GigaMIDI CSV splits (train/val/test) and uses Moonbeam's 
native data_preprocess logic to tokenize the MIDI files into .npy FME format.
Preserves artist-level splits.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import json
from tqdm import tqdm
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Import our standard preprocessing functions
from tokenize_dataset import extract_midi_tracks, transpose_to_c

def process_one(args):
    # Unpack arguments
    midi_path, split_name, onset_vocab_size, dur_vocab_size, out_dir, log_file, multi_instrument = args
    
    # We must import inside the worker if using ProcessPoolExecutor on some platforms,
    # or just rely on the global path setup.
    try:
        import data_preprocess
        from data_preprocess import process_midi_file_safe_v2
        from llama_recipes.datasets.music_tokenizer import MusicTokenizer
        
        # Inject the tokenizer globally into the module if it doesn't exist
        if getattr(data_preprocess, 'tokenizer', None) is None:
            data_preprocess.tokenizer = MusicTokenizer()
        
        # 1. Apply our custom preprocessing (filtering + transpose)
        score_result = extract_midi_tracks(midi_path, multi_instrument)
        if isinstance(score_result, str):
            return [None] # Failed to parse or no valid tracks
            
        score, _ = transpose_to_c(score_result, midi_path)
        
        # 2. Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            tmp_path = f.name
            
        score.dump_midi(tmp_path)
        
        # 3. Pass the preprocessed temporary file to Moonbeam
        # We must manually append the split name to the out_dir so they don't all get mixed together!
        split_out_dir = out_dir / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)
        
        res = process_midi_file_safe_v2(
            midi_file=tmp_path,
            split=split_name,
            onset_vocab_size=onset_vocab_size,
            dur_vocab_size=dur_vocab_size,
            output_folder=str(split_out_dir),
            log_file=str(log_file),
            silence_threshold=None
        )
        
        # 4. Clean up
        os.remove(tmp_path)
        
        return res
    except Exception as e:
        log.error(f"Failed to process {midi_path}: {e}")
        return [None]

def main():
    parser = argparse.ArgumentParser(description="Moonbeam Tokenization Wrapper")
    parser.add_argument("--splits-dir", type=str, required=True, help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--midi-root", type=str, required=True, help="Root directory for raw MIDI files")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for moonbeam dataset")
    parser.add_argument("--moonbeam-repo", type=str, required=True, help="Path to cloned Moonbeam repository")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--multi-instrument", action="store_true", help="Extract all instruments instead of just piano")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    midi_root = Path(args.midi_root)
    out_dir = Path(args.out_dir)
    moonbeam_repo = Path(args.moonbeam_repo)
    
    if not moonbeam_repo.exists():
        log.error(f"Moonbeam repo not found at {moonbeam_repo}")
        sys.exit(1)

    # Add Moonbeam repo to sys.path so we can import its data_preprocess.py
    sys.path.insert(0, str(moonbeam_repo))

    # Read config to get vocab sizes
    config_path = moonbeam_repo / "src" / "llama_recipes" / "configs" / "model_config.json"
    with open(config_path, "r") as f:
        model_config = json.load(f)
        
    onset_vocab_size = model_config["onset_vocab_size"]
    dur_vocab_size = model_config["dur_vocab_size"]
    
    log.info(f"Loaded Moonbeam config: onset_vocab={onset_vocab_size}, dur_vocab={dur_vocab_size}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "moonbeam_tokenize.log"

    # Process each split
    for split_name in ["train", "val", "test"]:
        csv_path = splits_dir / f"{split_name}.csv"
        if not csv_path.exists():
            log.warning(f"Split file not found: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        log.info(f"Processing {split_name} ({len(df)} files)...")
        
        # Prepare arguments for multiprocessing
        tasks = []
        for _, row in df.iterrows():
            midi_path = midi_root / row["file_path"]
            if midi_path.exists():
                tasks.append((midi_path, split_name, onset_vocab_size, dur_vocab_size, out_dir, log_file, args.multi_instrument))
            else:
                log.warning(f"MIDI file missing: {midi_path}")
                
        # Run in parallel
        success_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_one, t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name} Split"):
                res = future.result()
                if res and res[0] is not None:
                    success_count += 1
                    
        log.info(f"Successfully processed {success_count}/{len(tasks)} files for {split_name}.")

    log.info("Moonbeam tokenization complete.")

if __name__ == "__main__":
    main()
