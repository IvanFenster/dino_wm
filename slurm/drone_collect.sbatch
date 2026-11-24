#!/bin/bash
# [ARPL-WM] Drone dataset collection: one array task = one room (room seed = array index).
#   sbatch --array=0-0 slurm/drone_collect.sbatch                 # one room, 10 episodes (first test)
#   EPISODES=200 sbatch --array=0-9 slurm/drone_collect.sbatch    # 10 rooms x 200 episodes
#   sbatch --array=0-9 slurm/drone_collect.sbatch --noise 0.2     # extra args go to scripts/drone/collect.py
# Then pack once (no GPU needed, run inside any short job):
#   python scripts/drone/collect.py --pack --out $WM/data/drone_raw --dset $DATASET_DIR/drone_nav
# Rendering 224x224 frames does not need a big GPU, so this uses the cheapest one (GTX 2080 Ti,
# 2 CPUs, 7.34 SU per GPU-hour). For the A5000 use: -p savio4_gpu --qos=a5k_gpu4_normal
# --gres=gpu:A5000:1 -c 4. Rerunning skips episodes that already exist.
#SBATCH --job-name=drone_collect
#SBATCH --account=fc_auto
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -eo pipefail
export ARPL_CONDA_ENV=${ARPL_CONDA_ENV:-/global/scratch/users/$USER/conda/drone_env}
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/env.sh"
export TRANSPORT_WORLD=${TRANSPORT_WORLD:-$WM/src/transport-world}
export ARPL_EGL_DIR=$WM/cache/egl_${SLURM_JOB_ID:-local}
OUT=${DRONE_OUT:-$WM/data/drone_raw}
ROOM=${SLURM_ARRAY_TASK_ID:-0}
cd "$ARPL_REPO"
arpl_record_run "$OUT/run_info" drone_collect room=$ROOM "$@"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python scripts/drone/collect.py --out "$OUT" --room-seed "$ROOM" --episodes "${EPISODES:-10}" "$@"
