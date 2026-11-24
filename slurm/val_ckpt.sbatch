#!/bin/bash
# [ARPL-WM] Validation loss of one saved checkpoint, computed as train.py does (scripts/val_ckpt.py).
#   sbatch slurm/val_ckpt.sbatch <run_name> <epoch>
# Check first: point_maze_2s epoch 3 should give 0.0090 (what training printed).
#SBATCH --job-name=dinowm_val
#SBATCH --account=fc_auto
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -eo pipefail
RUN=${1:?usage: sbatch slurm/val_ckpt.sbatch <run_name> <epoch>}; EPOCH=${2:?epoch}
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/env.sh"
cd "$ARPL_REPO"
arpl_record_run "$WM/logs/$RUN/val" val_ckpt "$RUN" "$EPOCH"
python scripts/val_ckpt.py "$ARPL_REPO" "$WM/outputs/$RUN" "$EPOCH" 2>&1 | tee "$WM/logs/$RUN/val_ep${EPOCH}_${SLURM_JOB_ID:-local}.log"
