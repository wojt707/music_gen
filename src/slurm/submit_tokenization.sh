#!/bin/bash
#SBATCH --job-name=musicgen_tokenize
#SBATCH --output=%x_%j/output.out
#SBATCH --error=%x_%j/error.err
#SBATCH --partition=plgrid
#SBATCH --account=<YOUR_PLGRID_CPU_GRANT>
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR=${PROJECT_DIR:-$PWD}
VENV=${VENV:-$PROJECT_DIR/venv}
SCRATCH_DIR=${SCRATCH:-/tmp}/musicgen_tokenize_$SLURM_JOB_ID

# Read the dataset name argument (defaulting to gt_1000_with_ties if not provided)
DATASET_NAME=${1:-gt_1000_with_ties}

# Shift positional parameters so "$@" only contains extra args for the python script
if [ "$#" -gt 0 ] && [[ "$1" != "-"* ]]; then 
    shift 
fi

# ── Setup scratch ─────────────────────────────────────────────────────────────
echo "=== Setting up scratch dir ==="
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# Copy inputs to scratch for fast I/O
echo "=== Copying data to scratch ==="
cp -r $PROJECT_DIR/datasets_metadata/$DATASET_NAME/split       $SCRATCH_DIR/ 2>/dev/null || true
cp    $PROJECT_DIR/src/tokenize_dataset.py                    $SCRATCH_DIR/

# Set up output directories inside scratch
mkdir -p $SCRATCH_DIR/dataset/piano
mkdir -p $SCRATCH_DIR/dataset/multi-instrument

module load python/3.11

echo "=== Activating venv ==="
source $VENV/bin/activate

if [[ "$*" == *"--multi-instrument"* ]]; then
    echo "Mode: Multi-Instrument"
    export MUSICGEN_SPLITS_DIR=$SCRATCH_DIR/split/multi-instrument
    export MUSICGEN_OUTPUT_DIR=$SCRATCH_DIR/dataset/multi-instrument
else
    echo "Mode: Piano-Only"
    export MUSICGEN_SPLITS_DIR=$SCRATCH_DIR/split/piano
    export MUSICGEN_OUTPUT_DIR=$SCRATCH_DIR/dataset/piano
fi
export MUSICGEN_MIDI_ROOT=${MUSICGEN_MIDI_ROOT:-$PROJECT_DIR/raw_dataset}

echo "=== Starting tokenization ==="
python $SCRATCH_DIR/tokenize_dataset.py \
    --workers $SLURM_CPUS_PER_TASK \
    --bpe-sample 5000 "$@"

echo "=== Copying results back to project directory ==="
mkdir -p $PROJECT_DIR/datasets_metadata/$DATASET_NAME/dataset/
cp -r $SCRATCH_DIR/dataset/*  $PROJECT_DIR/datasets_metadata/$DATASET_NAME/dataset/
echo "=== Done! ==="
