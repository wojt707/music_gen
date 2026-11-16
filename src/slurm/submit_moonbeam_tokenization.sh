#!/bin/bash
#SBATCH --job-name=moonbeam_tokenization
#SBATCH --account=<YOUR_PLGRID_CPU_GRANT>
#SBATCH --partition=plgrid
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=moonbeam_tokenize_%j/output.out
#SBATCH --error=moonbeam_tokenize_%j/error.err

PROJECT_DIR=${PROJECT_DIR:-$PWD}
VENV=${VENV:-$PROJECT_DIR/venv}
SCRATCH_DIR=${SCRATCH:-/tmp}/moonbeam_tokenize_$SLURM_JOB_ID

DATASET_NAME=${1:-gt_1000_with_ties}

# We need to process both piano and multi-instrument splits
echo "=== Loading modules ==="
module load python/3.11

echo "=== Activating VENV ==="
source $VENV/bin/activate

MOONBEAM_REPO=$PROJECT_DIR/src/Moonbeam-MIDI-Foundation-Model
if [ ! -d "$MOONBEAM_REPO" ]; then
    echo "Cloning Moonbeam..."
    git clone https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model.git "$MOONBEAM_REPO"
fi

echo "=== Processing Splits ==="
for SPLIT in piano multi-instrument; do
    echo "Processing $SPLIT..."
    SPLITS_DIR=$PROJECT_DIR/datasets_metadata/$DATASET_NAME/split/$SPLIT
    OUT_DIR=$PROJECT_DIR/datasets_metadata/$DATASET_NAME/dataset/$SPLIT/moonbeam_preprocessed
    MIDI_ROOT=${MIDI_ROOT:-$PROJECT_DIR/raw_dataset}
    
    MULTI_FLAG=""
    if [ "$SPLIT" = "multi-instrument" ]; then
        MULTI_FLAG="--multi-instrument"
    fi
    
    mkdir -p $OUT_DIR
    
    python $PROJECT_DIR/src/tokenize_moonbeam.py \
      --splits-dir "$SPLITS_DIR" \
      --midi-root "$MIDI_ROOT" \
      --out-dir "$OUT_DIR" \
      --moonbeam-repo "$MOONBEAM_REPO" \
      --workers 8 \
      $MULTI_FLAG
done

echo "=== Done ==="
