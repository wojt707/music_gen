#!/bin/bash
#SBATCH --job-name=moonbeam_train
#SBATCH --account=<YOUR_PLGRID_GPU_GRANT>
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=moonbeam_train_%j/output.out
#SBATCH --error=moonbeam_train_%j/error.err

PROJECT_DIR=${PROJECT_DIR:-$PWD}
VENV=${VENV:-$PROJECT_DIR/venv_athena}
SCRATCH_DIR=${SCRATCH:-/tmp}/moonbeam_train_$SLURM_JOB_ID

DATASET_NAME=${1:-gt_1000_with_ties}

if [[ "$*" == *"--multi-instrument"* ]]; then
    SPLIT="multi-instrument"
    NUM_EPOCHS=${NUM_EPOCHS:-1}   # ~17-23h per epoch on A100
else
    SPLIT="piano"
    NUM_EPOCHS=${NUM_EPOCHS:-5}
fi

echo "=== Loading modules ==="
module load GCCcore/13.2.0
module load Python/3.11.5

echo "=== Setting up scratch dir ==="
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

echo "=== Activating VENV ==="
source $VENV/bin/activate

echo "=== Copying Moonbeam Code ==="
cp -r $PROJECT_DIR/src/Moonbeam-MIDI-Foundation-Model/* .

export MUSICGEN_SYNC_DIR=$PROJECT_DIR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Prepend Moonbeam's custom transformers and source to PYTHONPATH
export PYTHONPATH="$SCRATCH_DIR/src/llama_recipes/transformers_minimal/src:$SCRATCH_DIR/src:$PYTHONPATH"

CKPT_PATH=$PROJECT_DIR/src/Moonbeam-MIDI-Foundation-Model/checkpoints/moonbeam_839M.pt
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Pretrained checkpoint not found at $CKPT_PATH!"
    echo "Please download it first using:"
    echo "huggingface-cli download guozixunnicolas/moonbeam-midi-foundation-model moonbeam_839M.pt --local-dir $PROJECT_DIR/src/Moonbeam-MIDI-Foundation-Model/checkpoints"
    exit 1
fi

DATA_DIR=$PROJECT_DIR/datasets_metadata/$DATASET_NAME/dataset/$SPLIT/moonbeam_preprocessed
echo "=== Dynamically patching datasets.py ==="
sed -i "s|/PATH/TO/DATA/DIR|$DATA_DIR|g" src/llama_recipes/configs/datasets.py

echo "=== Starting Moonbeam Fine-Tuning ($DATASET_NAME - $SPLIT) ==="
mkdir -p moonbeam_output

MASTER_PORT=$((15000 + ($RANDOM % 40000)))
echo "=== Master Port: $MASTER_PORT ==="

EXTRA_TRAIN_ARGS=()
if [[ "$SPLIT" == "multi-instrument" && "$DATASET_NAME" == gt_1000* ]]; then
    EXTRA_TRAIN_ARGS+=(--max_train_step 16001)
fi

torchrun --nnodes 1 --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint 127.0.0.1:$MASTER_PORT recipes/finetuning/real_finetuning_uncon_gen.py \
  --model_name moonbeam_839M \
  --trained_checkpoint_path "$CKPT_PATH" \
  --dataset lakhmidi_dataset \
  --data_dir "$DATA_DIR" \
  --lr 5e-5 \
  --batch_size_training 2 \
  --gradient_accumulation_steps 4 \
  --val_batch_size 2 \
  --context_length 1024 \
  --num_epochs $NUM_EPOCHS \
  --pure_bf16 True \
  --enable_ddp True \
  --run_validation True \
  --validation_interval 2000 \
  --save_metrics True \
  --dist_checkpoint_root_folder moonbeam_output \
  --dist_checkpoint_folder ddp \
  --output_dir moonbeam_output \
  "${EXTRA_TRAIN_ARGS[@]}"

echo "=== Copying models back to project directory ==="
TARGET_DIR=$PROJECT_DIR/models/moonbeam/$DATASET_NAME/$SPLIT/run_$SLURM_JOB_ID
mkdir -p $TARGET_DIR
CKPT_DIR=moonbeam_output/ddp-moonbeam_839M
if [ -f "$CKPT_DIR/best_model.pt" ]; then
    python -c "
import torch
c = torch.load('$CKPT_DIR/best_model.pt', map_location='cpu')
c.pop('optimizer_state_dict', None)
torch.save(c, '$TARGET_DIR/best_model.pt')
print('slim best_model.pt saved, epoch', c.get('epoch'), 'step', c.get('step'), 'val_loss', c.get('best_val_loss'))
"
fi
cp moonbeam_output/*.json $TARGET_DIR/ 2>/dev/null || true
cp $CKPT_DIR/*.yaml $TARGET_DIR/ 2>/dev/null || true

if [ -n "$NOTIFY_ENDPOINT" ]; then
    echo "=== Sending completion notification ==="
    python -c "
import os
try:
    from notify_run import Notify
    endpoint = os.getenv('NOTIFY_ENDPOINT')
    if endpoint:
        Notify(endpoint=endpoint).send('Moonbeam FINISHED: $DATASET_NAME ($SPLIT) [Job $SLURM_JOB_ID]')
except Exception:
    pass
"
fi

echo "=== Done! ==="
