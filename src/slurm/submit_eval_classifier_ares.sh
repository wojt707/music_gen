#!/bin/bash
#SBATCH --job-name=musicgen_eval_cls
#SBATCH --account=<YOUR_PLGRID_CPU_GRANT>
#SBATCH --partition=plgrid
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=musicgen_eval_cls_%j/output.out
#SBATCH --error=musicgen_eval_cls_%j/error.err

# Usage examples (on Ares CPU cluster):
#   1. Evaluate all models across all datasets & splits:
#      sbatch submit_eval_classifier_ares.sh all all all
#   2. Evaluate all models for a specific dataset & split:
#      sbatch submit_eval_classifier_ares.sh gt_1000_with_ties piano all
#   3. Evaluate a single model:
#      sbatch submit_eval_classifier_ares.sh top_5_with_ties multi-instrument transformer

DATASET_NAME=${1:-all}
SPLIT=${2:-all}
MODEL=${3:-all}
EXTRA_ARGS=("${@:4}")

PROJECT_DIR=${PROJECT_DIR:-$PWD}
VENV=${VENV:-$PROJECT_DIR/venv}

echo "=== Loading Python on Ares ==="
module load python/3.11
source $VENV/bin/activate

set -e

export MUSICGEN_SYNC_DIR=$PROJECT_DIR
export MUSICGEN_MODELS_ROOT=${MUSICGEN_MODELS_ROOT:-$PROJECT_DIR/models}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "============================================================"
echo "=== [Ares CPU] Classifier Evaluation | dataset=$DATASET_NAME | split=$SPLIT | model=$MODEL ==="
echo "============================================================"

python "$PROJECT_DIR/src/evaluate_classifier.py" \
    --dataset-name "$DATASET_NAME" \
    --split "$SPLIT" \
    --model "$MODEL" \
    --generated-root "${GENERATED_ROOT:-$PROJECT_DIR/sample_generated_midis}" \
    "${EXTRA_ARGS[@]}"

echo "=== Classifier Evaluation complete! ==="

