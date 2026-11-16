#!/bin/bash
#SBATCH --job-name=musicgen_eval_metrics
#SBATCH --account=<YOUR_PLGRID_CPU_GRANT>
#SBATCH --partition=plgrid
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=musicgen_eval_metrics_%j/output.out
#SBATCH --error=musicgen_eval_metrics_%j/error.err

# Usage examples (on Ares CPU cluster):
#   1. Evaluate Real Baseline + all generated models across all datasets & splits:
#      sbatch submit_eval_metrics_ares.sh all all all
#   2. Evaluate ONLY the Real Dataset Baseline across all datasets & splits:
#      sbatch submit_eval_metrics_ares.sh all all real
#   3. Evaluate a specific dataset, split, and model:
#      sbatch submit_eval_metrics_ares.sh gt_1000_with_ties piano transformer

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
echo "=== [Ares CPU] Objective Metrics Evaluation | dataset=$DATASET_NAME | split=$SPLIT | model=$MODEL ==="
echo "============================================================"

python "$PROJECT_DIR/src/evaluate_metrics.py" \
    --dataset-name "$DATASET_NAME" \
    --split "$SPLIT" \
    --model "$MODEL" \
    --generated-root "${GENERATED_ROOT:-$PROJECT_DIR/sample_generated_midis}" \
    "${EXTRA_ARGS[@]}"

echo "=== Objective Metrics Evaluation complete! ==="

