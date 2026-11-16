#!/bin/bash -l
#SBATCH --job-name=musicgen_art_prepare
#SBATCH --output=%x_%j/output.out
#SBATCH --error=%x_%j/error.err
#SBATCH --account=<YOUR_PLGRID_CPU_GRANT>
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR=${PROJECT_DIR:-$PWD}
VENV=${VENV:-$PROJECT_DIR/venv}

# Raw GigaMIDI MIDI files location on the cluster
MIDI_ROOT=${MIDI_ROOT:-$PROJECT_DIR/raw_dataset}

# Scratch dir for fast I/O
SCRATCH_DIR=${SCRATCH:-/tmp}/musicgen_prepare_$(date +%Y%m%d_%H%M%S)

# Raw GigaMIDI metadata CSV (large, ~2.1M rows) and artists CSV
GIGAMIDI_CSV="${GIGAMIDI_CSV:-$PROJECT_DIR/datasets_metadata/Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv}"
ARTISTS_CSV="${ARTISTS_CSV:-$PROJECT_DIR/datasets_metadata/Global Music Artists.csv}"

# ── Setup ─────────────────────────────────────────────────────────────────────
echo "[$(date)] Setting up scratch dir: $SCRATCH_DIR"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

module load python/3.11

echo "[$(date)] Activating venv..."
source $VENV/bin/activate

# Copy the script to scratch (so edits don't affect a running job)
echo "[$(date)] Copying script to scratch..."
cp $PROJECT_DIR/src/prepare_art_gigamidi.py $SCRATCH_DIR/

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[$(date)] Starting prepare_art_gigamidi.py with $SLURM_CPUS_PER_TASK workers..."
python $SCRATCH_DIR/prepare_art_gigamidi.py \
    --gigamidi  "$GIGAMIDI_CSV" \
    --artists   "$ARTISTS_CSV" \
    --midi-root "$MIDI_ROOT" \
    --output    "$SCRATCH_DIR/gigamidi_artist_prepared.csv" \
    --workers   $SLURM_CPUS_PER_TASK 

echo "[$(date)] Script finished."

# ── Stage out ─────────────────────────────────────────────────────────────────
echo "[$(date)] Copying result CSV back to project directory..."
mkdir -p "$PROJECT_DIR/datasets_metadata"
cp $SCRATCH_DIR/gigamidi_artist_prepared.csv "$PROJECT_DIR/datasets_metadata/gigamidi_artist_prepared.csv"
echo "[$(date)] Saved to $PROJECT_DIR/datasets_metadata/gigamidi_artist_prepared.csv"

echo "[$(date)] Done."
