#!/bin/bash
#SBATCH --job-name=musicgen_classifier
#SBATCH --output=%x_%j/output.out
#SBATCH --error=%x_%j/error.err
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=<YOUR_PLGRID_GPU_GRANT>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR=${PROJECT_DIR:-$PWD}
VENV=${VENV:-$PROJECT_DIR/venv_athena}
SCRATCH_DIR=${SCRATCH:-/tmp}/musicgen_classifier_$SLURM_JOB_ID

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
cp $PROJECT_DIR/src/train_classifier.py $SCRATCH_DIR/
source $VENV/bin/activate

export MUSICGEN_SYNC_DIR=$PROJECT_DIR

echo "=== Starting Classifier Training ==="
# We pass through the dataset name and any extra arguments (e.g., --multi-instrument)
python train_classifier.py --dataset-name $DATASET_NAME "$@"

echo "=== Copying models back to project directory ==="
# The script automatically saves models to classifier_output in the scratch dir
if [[ "$*" == *"--multi-instrument"* ]]; then
    SPLIT="multi-instrument"
else
    SPLIT="piano"
fi
# Append the Slurm Job ID to the target directory to save unique runs!
TARGET_DIR=$PROJECT_DIR/models/classifier/$DATASET_NAME/$SPLIT/run_$SLURM_JOB_ID
mkdir -p $TARGET_DIR
cp -r classifier_output/$SPLIT/* $TARGET_DIR/ || true
echo "=== Done! ==="
