# Symbolic Music Generation: Comparative Deep Learning Analysis

This repository contains the official codebase, dataset splits, generated samples, and evaluation results for the comparative analysis of deep learning architectures for symbolic music generation.

The study systematically evaluates four distinct model paradigms across four genre taxonomies in both **Solo Piano** and **Multi-Instrument** settings:
1. **Custom Recurrent Baseline** (3-layer LSTM with persistent genre embeddings)
2. **Custom Transformer** (6-layer autoregressive causal Transformer trained from scratch)
3. **Cross-Modal Text Foundation Model** (LLaMA 3.2 1B fine-tuned via MIDI token vocabulary expansion)
4. **Native MIDI Foundation Model** (Moonbeam 839M with continuous Fundamental Music Embeddings and Multidimensional Relative Attention)

Evaluation includes objective musical metrics (Pitch Class Entropy, Scale Consistency, Polyphony Rate, Groove Consistency), training memorization analysis ($N$-gram / sequence overlap), and stylistic steerability assessed by a dedicated RoBERTa-based reverse classifier.

---

## Repository Structure

```text
├── datasets_metadata/        # Curated splits & metadata (artist-disjoint)
│   ├── gigamidi_artist_prepared.csv
│   ├── Global Music Artists.csv
│   ├── top_5_no_ties/        # 5 genres, strict artist-genre mapping
│   ├── top_5_with_ties/      # 5 genres, multi-genre artists preserved
│   ├── gt_1000_no_ties/      # 10 genres (support > 1,000 tracks)
│   └── gt_1000_with_ties/    # 12 genres (support > 1,000 tracks)
├── evaluation_results/       # Full evaluation benchmarks
│   ├── classifier/           # Reverse RoBERTa steerability metrics & confusion matrices
│   └── metrics/              # Objective musical metrics & memorization statistics
├── sample_generated_midis/   # Generated compositions across all experimental conditions
├── src/                      # Source code
│   ├── slurm/                # SLURM batch job scripts for PLGrid HPC clusters
│   ├── Moonbeam-MIDI-Foundation-Model/ # Submodule / base code for Moonbeam
│   ├── prepare_art_gigamidi.py   # Dataset cleaning & Spotify metadata matching
│   ├── create_splits.py          # Artist-disjoint train/val/test split generator
│   ├── tokenize_dataset.py       # REMI+BPE and Octuple tokenization pipeline
│   ├── tokenize_moonbeam.py      # Moonbeam FME tokenization wrapper
│   ├── train_classifier.py       # Reverse RoBERTa genre classifier training
│   ├── train_lstm.py             # Custom LSTM training
│   ├── train_transformer.py      # Custom Transformer training
│   ├── train_llama.py            # LLaMA 3.2 1B fine-tuning
│   ├── generate.py               # Autoregressive MIDI sampling
│   ├── evaluate_classifier.py    # Genre steerability evaluation
│   ├── evaluate_metrics.py       # Objective metrics & memorization evaluation
│   ├── build_thesis_results.py   # Aggregates evaluation CSVs into LaTeX summary tables
│   └── build_thesis_figures.py   # Generates publication-ready figures & plots
└── requirements.txt
```

---

## Quickstart & Reproduction

### 1. Environment Setup

Create and activate a virtual environment:

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
.\venv\Scripts\activate

pip install -r requirements.txt
```

For LLaMA 3.2 1B fine-tuning, log in to Hugging Face or export your token:
```bash
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"
```

### 2. Dataset Preparation

1. Download raw MIDI files from the [GigaMIDI v2.0 repository](https://huggingface.co/datasets/Metacreation/GigaMIDI) and place them in `raw_dataset/`.
2. Generate REMI+BPE and Octuple tokenized datasets:
```bash
# Piano split
python src/tokenize_dataset.py --dataset-name top_5_no_ties --workers 16

# Multi-instrument split
python src/tokenize_dataset.py --dataset-name top_5_no_ties --multi-instrument --workers 16
```

### 3. Training Models

Train locally or submit batch jobs to a cluster:

```bash
# Local execution examples:
python src/train_classifier.py --dataset-name top_5_no_ties
python src/train_lstm.py --dataset-name top_5_no_ties --epochs 30
python src/train_transformer.py --dataset-name top_5_no_ties --epochs 50
python src/train_llama.py --dataset-name top_5_no_ties --epochs 10
```

### 4. Running on PLGrid HPC

Training and evaluation were conducted on Poland's National Supercomputing Infrastructure (**PLGrid**):
- **Athena Cluster**: NVIDIA A100 (40GB/80GB) SXM4 nodes for GPU training (`train_llama.py`, `train_transformer.py`, `train_lstm.py`, `submit_moonbeam_train.sh`).
- **Ares Cluster**: Multi-core CPU compute nodes for dataset tokenization, MIDI synthesis, and parallel evaluation.

To launch jobs on PLGrid:
```bash
# 1. Update the account grant in src/slurm/*.sh (<YOUR_PLGRID_CPU_GRANT> / <YOUR_PLGRID_GPU_GRANT>)
# 2. Submit jobs:
sbatch src/slurm/submit_tokenization.sh top_5_no_ties
sbatch src/slurm/submit_classifier.sh top_5_no_ties --multi-instrument
sbatch src/slurm/submit_transformer.sh top_5_no_ties
sbatch src/slurm/submit_llama.sh top_5_no_ties
sbatch src/slurm/submit_lstm.sh top_5_no_ties
```

### 5. Generation & Evaluation

Generate compositions with conditioned prompts:
```bash
python src/generate.py --model llama --dataset-name top_5_no_ties --split multi-instrument --songs-per-genre 100
```

Run quantitative evaluations:
```bash
# Classify generated MIDIs using the reverse RoBERTa classifier
python src/evaluate_classifier.py --dataset-name top_5_no_ties --split multi-instrument --model llama

# Evaluate musical distribution metrics and training memorization
python src/evaluate_metrics.py --dataset-name top_5_no_ties --split multi-instrument --model llama
```

To compile summary tables and figures:
```bash
python src/build_thesis_results.py
python src/build_thesis_figures.py
```

---

## License

- The source code in this repository is licensed under the [MIT License](LICENSE).
- The dataset manifests, split metadata, and annotations derived from GigaMIDI are distributed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license, in accordance with the original [GigaMIDI license](https://huggingface.co/datasets/Metacreation/GigaMIDI).
- Raw MIDI compositions are subject to the original copyright of their respective authors and the upstream dataset distribution.
