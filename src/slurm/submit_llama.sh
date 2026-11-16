#!/bin/bash
#SBATCH --job-name=musicgen_llama
#SBATCH --account=<YOUR_PLGRID_GPU_GRANT>
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=musicgen_llama_%j/output.out
#SBATCH --error=musicgen_llama_%j/error.err
#SBATCH --signal=B:TERM@600

PROJECT_DIR=${PROJECT_DIR:-$PWD}
VENV=${VENV:-$PROJECT_DIR/venv_athena}
SCRATCH_DIR=${SCRATCH:-/tmp}/musicgen_llama_$SLURM_JOB_ID

# Read the dataset name argument (defaulting to gt_1000_with_ties if not provided)
DATASET_NAME=${1:-gt_1000_with_ties}

# Determine split type before shifting
SPLIT="piano"
if [[ "$*" == *"--multi-instrument"* || "$*" == *"multi-instrument"* ]]; then
    SPLIT="multi-instrument"
fi

# Build clean arguments for train_llama.py, ignoring stray positional 'piano' or 'multi-instrument' strings
PY_ARGS=()
for arg in "$@"; do
    if [ "$arg" == "$DATASET_NAME" ] || [ "$arg" == "piano" ] || [ "$arg" == "multi-instrument" ]; then
        continue
    fi
    PY_ARGS+=("$arg")
done

if [ "$SPLIT" == "multi-instrument" ]; then
    PY_ARGS+=(--multi-instrument)
fi

TARGET_DIR=$PROJECT_DIR/models/llama/$DATASET_NAME/$SPLIT/run_$SLURM_JOB_ID

# Copy only what is needed:
#  - best_model / final model weights
#  - trainer_state.json (full loss history for plots) from the newest checkpoint
# Full checkpoints with optimizer stay on $SCRATCH for resuming.
copy_results() {
    mkdir -p $TARGET_DIR
    [ -d llama_output/$SPLIT/best_model ] && cp -r llama_output/$SPLIT/best_model $TARGET_DIR/
    LAST_CKPT=$(ls -d llama_output/$SPLIT/checkpoint-* 2>/dev/null | sort -V | tail -1)
    [ -n "$LAST_CKPT" ] && cp $LAST_CKPT/trainer_state.json $TARGET_DIR/ 2>/dev/null
    cp llama_output/$SPLIT/*.json $TARGET_DIR/ 2>/dev/null || true
    echo "Latest full checkpoint (for --resume): $SCRATCH_DIR/$LAST_CKPT"
}

# Trap: SLURM sends SIGTERM 10 min before the time limit (--signal=B:TERM@600)
cleanup_and_save() {
    echo "=== SLURM SIGNAL RECEIVED: stopping training and saving results ==="
    kill -TERM $PY_PID 2>/dev/null; wait $PY_PID 2>/dev/null
    copy_results
    echo "=== Results saved before job termination! ==="
    exit 0
}
trap cleanup_and_save SIGTERM SIGINT

# ── Setup ────────────────────────────────────────────────────────────────────
echo "=== Loading modules ==="
module load GCCcore/13.2.0
module load Python/3.11.5

echo "=== Setting up scratch dir ==="
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

echo "=== Copying script and activating venv ==="
cp $PROJECT_DIR/src/train_llama.py $SCRATCH_DIR/
source $VENV/bin/activate

export MUSICGEN_SYNC_DIR=$PROJECT_DIR
export HF_TOKEN="${HF_TOKEN:-<YOUR_HUGGINGFACE_TOKEN>}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=== Starting LLaMA Fine-Tuning ($DATASET_NAME - $SPLIT) ==="
# Run in background + wait, so bash can react to SIGTERM immediately
python train_llama.py --dataset-name $DATASET_NAME "${PY_ARGS[@]}" &
PY_PID=$!
wait $PY_PID

echo "=== Copying models back to project directory ==="
copy_results
echo "=== Done! ==="
