#!/bin/bash
# [ARPL-WM] Depth pilot (scripts/depth_pilot.py): RGB vs DeFM depth vs concat on Granular, cached frozen features.
#   sbatch slurm/depth_pilot.sbatch [extra args for depth_pilot.py, e.g. --n_ep 1000 --epochs 20 --seed 1]
# Needs: data in $DATASET_DIR/deformable/granular (scripts/get_deformable.sh) and DeFM in $WM/src/defm
# (setup/savio_setup.sh clones it). Colab defaults: 200 episodes, 5 epochs, seed 0.
#SBATCH --job-name=dinowm_depth_pilot
#SBATCH --account=fc_auto
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/env.sh"
export PYTHONPATH=$WM/src/defm:${PYTHONPATH:-}
OUT=$WM/outputs/depth_pilot_${SLURM_JOB_ID:-local}
cd "$ARPL_REPO"
arpl_record_run "$OUT/run_info" depth_pilot "$@"
python scripts/depth_pilot.py --data_root "$DATASET_DIR" --out "$OUT" "$@" 2>&1 | tee "$OUT/stdout.log"
