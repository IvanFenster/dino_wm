#!/bin/bash
# [ARPL-WM] One-time Savio setup. Safe to rerun: every step skips itself when already done.
# Run it inside a short interactive job, not on the login node (conda solving is heavy):
#   srun -A fc_auto -p savio4_gpu --qos=a5k_gpu4_normal --gres=gpu:A5000:1 -c 4 -t 01:00:00 --pty bash
#   cd ~/dino_wm && bash setup/savio_setup.sh
# Steps: scratch folders, conda env from environment.yaml on scratch, MuJoCo 2.1, DeFM code,
# DINOv2 weights (Hugging Face fallback). Datasets are separate: see ARPL_NOTES.md.
set -eo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/slurm/env.sh"
echo "[ARPL-WM] repo $REPO | work dir $WM | conda env $ARPL_CONDA_ENV"

# 1. Conda env on scratch (home is only 50 GB). Same package list as upstream environment.yaml.
if [ ! -d "$ARPL_CONDA_ENV" ]; then
    mkdir -p "$(dirname "$ARPL_CONDA_ENV")"
    export CONDA_PKGS_DIRS=$ARPL_SCRATCH/conda/pkgs
    conda env create -p "$ARPL_CONDA_ENV" -f "$REPO/environment.yaml"
fi
eval "$(conda shell.bash hook)"
conda activate "$ARPL_CONDA_ENV"

# 2. MuJoCo 2.1 for mujoco-py (PointMaze), as in the upstream README
if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
    mkdir -p "$HOME/.mujoco"
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210_$USER.tgz
    tar -xzf /tmp/mujoco210_$USER.tgz -C "$HOME/.mujoco" && rm /tmp/mujoco210_$USER.tgz
fi

# 3. DeFM depth encoder code (not pip-installable); used by scripts/depth_pilot.py
[ -d "$WM/src/defm" ] || git clone -q https://github.com/leggedrobotics/defm.git "$WM/src/defm"

# 4. DINOv2 weights into torch hub's cache (Meta's server had an expired certificate)
python "$REPO/scripts/fetch_dinov2_weights.py" "$WM/cache/dinov2_vits14_pretrain.pth"

# 5. Import check (the first mujoco_py import compiles it, which takes a few minutes)
python -c "import d4rl, mujoco_py, gym, hydra, decord, einops, accelerate, wandb, pymunk, shapely, torch; print('imports ok')"
echo "[ARPL-WM] setup done. Next: sbatch slurm/gpu_test.sbatch"
