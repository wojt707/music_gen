#!/bin/bash
#SBATCH --job-name=musicgen_transformer
#SBATCH --account=<YOUR_PLGRID_GPU_GRANT>
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=musicgen_transformer_%j/output.out
#SBATCH --error=musicgen_transformer_%j/error.err

PROJECT_DIR=${PROJECT_DIR:-$PWD}
VENV=${VENV:-$PROJECT_DIR/venv_athena}
SCRATCH_DIR=${SCRATCH:-/tmp}/musicgen_transformer_$SLURM_JOB_ID

# Read the dataset name argument (defaulting to gt_1000_with_ties if not provided)
DATASET_NAME=${1:-gt_1000_with_ties}

# Shift positional parameters so "$@" only contains extra args for the python script
if [ "$#" -gt 0 ] && [[ "$1" != "-"* ]]; then 
    shift 
fi

# ── Setup ────────────────────────────────────────────────────────────────────
echo "=== Loading modules ==="
module load GCCcore/13.2.0
module load Python/3.11.5

echo "=== Setting up scratch dir ==="
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

echo "=== Copying script and activating venv ==="
cp $PROJECT_DIR/src/train_transformer.py $SCRATCH_DIR/
source $VENV/bin/activate

export MUSICGEN_SYNC_DIR=$PROJECT_DIR

echo "=== Starting Transformer Training ==="
# We pass through the dataset name and any extra arguments (e.g., --multi-instrument)
python train_transformer.py --dataset-name $DATASET_NAME "$@"

echo "=== Copying models back to project directory ==="
if [[ "$*" == *"--multi-instrument"* ]]; then
    SPLIT="multi-instrument"
else
    SPLIT="piano"
fi
TARGET_DIR=$PROJECT_DIR/models/transformer/$DATASET_NAME/$SPLIT/run_$SLURM_JOB_ID
mkdir -p $TARGET_DIR
cp -r transformer_output/$SPLIT/* $TARGET_DIR/ || true
echo "=== Done! ==="
