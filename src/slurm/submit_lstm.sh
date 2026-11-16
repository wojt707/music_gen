#!/bin/bash
#SBATCH --job-name=musicgen_lstm
#SBATCH --output=%x_%j/output.out
#SBATCH --error=%x_%j/error.err
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=<YOUR_PLGRID_GPU_GRANT>
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR=${PROJECT_DIR:-$PWD}
VENV=${VENV:-$PROJECT_DIR/venv_athena}
SCRATCH_DIR=${SCRATCH:-/tmp}/musicgen_lstm_$SLURM_JOB_ID

# Read the dataset name argument (defaulting to gt_1000_with_ties if not provided)
DATASET_NAME=${1:-gt_1000_with_ties}

# Shift positional parameters so "$@" only contains extra args for the python script
if [ "$#" -gt 0 ] && [[ "$1" != "-"* ]]; then 
    shift 
fi

# ── Setup ─────────────────────────────────────────────────────────────────────
echo "=== Loading modules ==="
module load GCCcore/13.2.0
module load Python/3.11.5

echo "=== Setting up scratch dir ==="
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

echo "=== Copying script and activating venv ==="
cp $PROJECT_DIR/src/train_lstm.py $SCRATCH_DIR/
source $VENV/bin/activate

export MUSICGEN_SYNC_DIR=$PROJECT_DIR

# ── Determine split for output folder naming ───────────────────────────────────
if [[ "$*" == *"--multi-instrument"* ]]; then
    SPLIT="multi-instrument"
else
    SPLIT="piano"
fi

echo "=== Training LSTM | dataset=$DATASET_NAME | split=$SPLIT ==="

if [[ "$*" == *"--resume"* ]]; then
    LATEST_RUN=$(ls -td $PROJECT_DIR/models/lstm/$DATASET_NAME/$SPLIT/run_* 2>/dev/null | head -1)
    if [ ! -z "$LATEST_RUN" ] && [ -f "$LATEST_RUN/best_model.pt" ]; then
        echo "=== Found previous run: $LATEST_RUN ==="
        echo "=== Copying checkpoint to scratch directory for resume... ==="
        mkdir -p lstm_output/$SPLIT
        cp "$LATEST_RUN/best_model.pt" "lstm_output/$SPLIT/best_model.pt"
    else
        echo "=== WARNING: --resume passed but no previous best_model.pt found in $PROJECT_DIR ==="
    fi
fi

python train_lstm.py --dataset-name $DATASET_NAME "$@"

echo "=== Copying output back to project directory ==="
TARGET_DIR=$PROJECT_DIR/models/lstm/$DATASET_NAME/$SPLIT/run_$SLURM_JOB_ID
mkdir -p $TARGET_DIR
if [ -f "lstm_output/$SPLIT/best_model.pt" ]; then
    python -c "
import torch
ckpt = torch.load('lstm_output/$SPLIT/best_model.pt', map_location='cpu')
if isinstance(ckpt, dict):
    ckpt.pop('optimizer_state_dict', None)
torch.save(ckpt, '$TARGET_DIR/best_model.pt')
"
fi
cp lstm_output/$SPLIT/*.json $TARGET_DIR/ 2>/dev/null || true
echo "=== Saved to $TARGET_DIR ==="
echo "=== Done! ==="
