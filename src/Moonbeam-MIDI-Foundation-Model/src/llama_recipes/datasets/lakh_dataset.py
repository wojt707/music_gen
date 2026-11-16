import copy
import json

import torch
from torch.utils.data import Dataset
import glob
import mido
from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor
import tqdm
import numpy as np
import pandas as pd
import os

class LakhDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.data_dir = dataset_config.data_dir
        self.tokenizer = tokenizer
        self.file_paths = []

        # Check if csv_file exists and is valid
        csv_file = getattr(dataset_config, "csv_file", None)
        if csv_file and os.path.isfile(csv_file):
            try:
                split_data = pd.read_csv(csv_file)
                file_basenames = split_data['file_base_name'].values
                splits = split_data['split'].values
                part = "train" if partition == "train" else ("val" if "val" in splits else "test")
                self.file_paths = [
                    os.path.join(self.data_dir, "processed", f) if os.path.exists(os.path.join(self.data_dir, "processed", f))
                    else os.path.join(self.data_dir, f)
                    for f, s in zip(file_basenames, splits) if s == part or s == partition
                ]
            except Exception as e:
                print(f"Notice: Failed to read CSV {csv_file} ({e}), falling back to direct directory globbing.")

        # If no CSV or CSV gave 0 files, search directly in partition directory
        if not self.file_paths:
            # Check partition folder (e.g. data_dir/train, data_dir/val, data_dir/test)
            candidates = [partition]
            if partition in ("test", "val"):
                candidates = ["val", "test"]
            
            for candidate in candidates:
                candidate_dir = os.path.join(self.data_dir, candidate)
                if os.path.isdir(candidate_dir):
                    found = sorted(glob.glob(os.path.join(candidate_dir, "*.npy")))
                    if found:
                        self.file_paths = found
                        print(f"[{partition}] Loaded {len(self.file_paths)} files from {candidate_dir}")
                        break
            
            # If still not found, check if .npy files are directly in self.data_dir
            if not self.file_paths:
                self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.npy")))
                if self.file_paths:
                    print(f"[{partition}] Loaded {len(self.file_paths)} files directly from {self.data_dir}")

        if len(self.file_paths) == 0:
            print(f"WARNING: LakhDataset found 0 files for partition '{partition}' in {self.data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        raw_tokens = np.load(self.file_paths[index])
        
        encoded_tokens = self.tokenizer.encode_series(raw_tokens, if_add_sos = True, if_add_eos = True) #SOS and EOS are added in the tokenizer
        
        encoded_tokens_label = self.tokenizer.encode_series_labels(encoded_tokens, if_added_sos = True, if_added_eos = True)

        return {
            "input_ids": encoded_tokens,
            "labels": encoded_tokens_label,
            "attention_mask":[] #Mask to be calculated dynamically during concatenation
        }


