#!/bin/bash
# [ARPL-WM] 10-minute smoke test: GPU visible, conda env imports, DINOv2 loads, fp16 + fused attention run.
#   sbatch slurm/gpu_test.sbatch        (submit from the repo root)
#SBATCH --job-name=dinowm_gpu_test
#SBATCH --account=fc_auto
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x_%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/env.sh"
cd "$ARPL_REPO"
arpl_record_run "$WM/logs/gpu_test" gpu_test
nvidia-smi
python -c "import d4rl, mujoco_py, gym, hydra, decord, einops, accelerate, wandb, pymunk, shapely, torch; print('imports ok')"
python -c "import torch; print('GPU visible:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/check_equivalence.py --device cuda
python -c "
import torch
m = torch.hub.load('facebookresearch/dinov2:85a24602099d397264d5b30461ad7f3bfd726ca1', 'dinov2_vits14', trust_repo=True, skip_validation=True).cuda().eval()
with torch.no_grad():
    print('DINOv2 ok:', tuple(m.forward_features(torch.randn(1, 3, 196, 196).cuda())['x_norm_patchtokens'].shape))
"
