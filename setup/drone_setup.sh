#!/bin/bash
# [ARPL-WM] One-time setup for drone data collection on Savio. Safe to rerun.
# Run inside a short job, not on the login node (conda solving and pip builds are heavy):
#   srun -A fc_auto -p savio3_gpu --qos=gtx2080_gpu3_normal --gres=gpu:GTX2080TI:1 -c 2 -t 01:00:00 --pty bash
#   cd ~/dino_wm && bash setup/drone_setup.sh
# Steps: conda env drone_env on scratch, Jupyter kernel "drone_env", transport-world clone,
# GPU rendering check.
set -eo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export ARPL_CONDA_ENV=${ARPL_CONDA_ENV:-/global/scratch/users/$USER/conda/drone_env}
source "$REPO/slurm/env.sh"
echo "[ARPL-WM] repo $REPO | work dir $WM | conda env $ARPL_CONDA_ENV"

# 1. Conda env on scratch
if [ ! -d "$ARPL_CONDA_ENV" ]; then
    export CONDA_PKGS_DIRS=$ARPL_SCRATCH/conda/pkgs
    export PIP_CACHE_DIR=$ARPL_SCRATCH/pip_cache
    conda env create -p "$ARPL_CONDA_ENV" -f "$REPO/setup/drone_env.yaml"
fi
eval "$(conda shell.bash hook)"
conda activate "$ARPL_CONDA_ENV"

# 2. Jupyter kernel for Open OnDemand (lands in ~/.local/share/jupyter/kernels/drone_env)
python -m ipykernel install --user --name drone_env --display-name "drone_env (Genesis)"

# 3. JoonHo's transport-world (private): SE3Controller and race2 drone config.
#    Needs your GitHub access on Savio: an SSH key added to GitHub, or a classic token with repo scope.
TW=$WM/src/transport-world
if [ ! -d "$TW/payload_mappo" ]; then
    git clone --depth 1 git@github.com:JHLee0513/transport-world.git "$TW" || {
        echo "Clone failed. Either add an SSH key on Savio to GitHub, or run:"
        echo "  git clone https://<user>:<classic token>@github.com/JHLee0513/transport-world.git $TW"
        echo "  git -C $TW remote set-url origin https://github.com/JHLee0513/transport-world.git"
        exit 1; }
fi

# 4. Imports and a GPU rendering check (one frame; should be well under 0.3 s on a GPU)
cd "$REPO"
TRANSPORT_WORLD=$TW python - <<'PY'
import time
from drone_sim.egl import setup_gpu_rendering
setup_gpu_rendering()
from drone_sim.env import DroneFPVEnv
t = time.time(); env = DroneFPVEnv(); print(f"build {time.time() - t:.0f} s")
env.render(); t = time.time(); env.render(); print(f"render {time.time() - t:.3f} s/frame")
obs = env.reset(); print({k: v.shape for k, v in obs.items()})
PY
echo "[ARPL-WM] drone setup done. Next: notebooks/drone/drone_collect.ipynb, or sbatch --array=0-0 slurm/drone_collect.sbatch"
