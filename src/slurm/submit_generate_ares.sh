#!/bin/bash
#SBATCH --job-name=musicgen_gen_cpu
#SBATCH --account=<YOUR_PLGRID_CPU_GRANT>
#SBATCH --partition=plgrid
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --output=musicgen_gen_cpu_%j/output.out
#SBATCH --error=musicgen_gen_cpu_%j/error.err

# Usage examples (on Ares CPU cluster):
#   sbatch submit_generate_ares.sh lstm gt_1000_with_ties piano 100
#   sbatch submit_generate_ares.sh transformer top_5_with_ties multi-instrument 100

MODEL=${1:-lstm}
DATASET_NAME=${2:-gt_1000_with_ties}
SPLIT=${3:-piano}
SONGS_PER_GENRE=${4:-100}
EXTRA_ARGS=("${@:5}")

PROJECT_DIR=${PROJECT_DIR:-$PWD}
VENV=${VENV:-$PROJECT_DIR/venv}
OUT_BASE=${SCRATCH:-/tmp}/generated_midis
ARCHIVE_DIR=$PROJECT_DIR/sample_generated_midis

echo "=== Loading Python on Ares ==="
module load python/3.11
source $VENV/bin/activate

set -e

export MUSICGEN_SYNC_DIR=$PROJECT_DIR
export MUSICGEN_MODELS_ROOT=${MUSICGEN_MODELS_ROOT:-$PROJECT_DIR/models}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "$OUT_BASE" "$ARCHIVE_DIR"
TARGET_OUT="$OUT_BASE/$DATASET_NAME/$SPLIT/$MODEL"

echo "============================================================"
echo "=== [Ares CPU] Generating $SONGS_PER_GENRE songs/genre | model=$MODEL | dataset=$DATASET_NAME | split=$SPLIT ==="
echo "=== Output: $TARGET_OUT ==="
echo "============================================================"

python "$PROJECT_DIR/src/generate.py" \
    --model "$MODEL" \
    --dataset-name "$DATASET_NAME" \
    --split "$SPLIT" \
    --songs-per-genre "$SONGS_PER_GENRE" \
    --output-dir "$TARGET_OUT" \
    "${EXTRA_ARGS[@]}"

ARCHIVE_FILE="$ARCHIVE_DIR/${DATASET_NAME}_${SPLIT}_${MODEL}.tar.gz"
TMP_ARCHIVE="$ARCHIVE_DIR/.${DATASET_NAME}_${SPLIT}_${MODEL}.${SLURM_JOB_ID}.tar.gz"
echo "=== Archiving generated MIDIs to $ARCHIVE_FILE ==="
tar --warning=no-file-changed -czf "$TMP_ARCHIVE" -C "$OUT_BASE/$DATASET_NAME/$SPLIT" "$MODEL" || true
mv -f "$TMP_ARCHIVE" "$ARCHIVE_FILE" 2>/dev/null || true

echo "=== Generation complete! ==="

