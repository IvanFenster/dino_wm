# [ARPL-WM] Shared environment for Savio jobs. Sourced by every slurm/*.sbatch file and by setup/savio_setup.sh.
# Everything can be overridden by exporting the variable before sbatch (sbatch passes your environment on).
# Layout on scratch (not backed up, so keep code in git and copy results you care about):
#   $WM/data          datasets (DATASET_DIR)
#   $WM/outputs/<run> training runs: hydra.yaml, checkpoints/, epoch_losses.csv, run_info/
#   $WM/plan_outputs  planning results
#   $WM/logs/<run>    arpl_logger output (sessions, epoch timing, GPU use, stdout)
#   $WM/cache         DINOv2 weights, torch hub, Hugging Face cache
#   $WM/src           third-party code that is not pip-installable (DeFM)

export ARPL_SCRATCH=${ARPL_SCRATCH:-/global/scratch/users/$USER}
export WM=${WM:-$ARPL_SCRATCH/arpl_wm}
export DATASET_DIR=${DATASET_DIR:-$WM/data}
export ARPL_CONDA_ENV=${ARPL_CONDA_ENV:-$ARPL_SCRATCH/conda/dino_wm}
# Repo root: the folder you ran sbatch from (submit from the repo root), else this file's parent folder
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/train.py" ]; then
    export ARPL_REPO=${ARPL_REPO:-$SLURM_SUBMIT_DIR}
else
    export ARPL_REPO=${ARPL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
fi

export TORCH_HOME=${TORCH_HOME:-$WM/cache/torch}      # torch.hub code + weights (DINOv2 lands here)
export HF_HOME=${HF_HOME:-$WM/cache/hf}
export WANDB_MODE=${WANDB_MODE:-offline}              # log locally; sync later with `wandb sync` if wanted
export WANDB_DIR=${WANDB_DIR:-$WM/wandb}
export MUJOCO_GL=${MUJOCO_GL:-egl}                    # render without a screen
export PYTHONUNBUFFERED=1                             # logs fill up live
export D4RL_SUPPRESS_IMPORT_ERROR=1                   # hide harmless mjrl/flow/carla warnings
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
# ARPL speedups, on by default for our runs (see ARPL_NOTES.md). Set to 0 for the exact upstream code path.
export ARPL_AMP=${ARPL_AMP:-1}
export ARPL_SDPA=${ARPL_SDPA:-1}

mkdir -p "$WM"/{data,outputs,plan_outputs,logs,cache,src} "$TORCH_HOME" "$HF_HOME" "$WANDB_DIR"

# Conda. Check the module name with `module avail anaconda` if this fails.
if ! command -v conda >/dev/null 2>&1; then
    module load anaconda3 2>/dev/null || true
fi
if [ -d "$ARPL_CONDA_ENV" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$ARPL_CONDA_ENV"
fi

# Write everything needed to reproduce a run into <dir>: git commit and diff, Slurm job, GPU, versions, command.
arpl_record_run() {
    local dir=$1; shift
    mkdir -p "$dir"
    local f="$dir/run_${SLURM_JOB_ID:-local}_$(date -u +%Y%m%d_%H%M%S).txt"
    {
        echo "date_utc: $(date -u '+%Y-%m-%d %H:%M:%S')"
        echo "command: $*"
        echo "git_commit: $(git -C "$ARPL_REPO" rev-parse HEAD)"
        echo "git_branch: $(git -C "$ARPL_REPO" rev-parse --abbrev-ref HEAD)"
        echo "git_dirty_files:"; git -C "$ARPL_REPO" status --porcelain | sed 's/^/  /'
        echo "slurm: job=${SLURM_JOB_ID:-} name=${SLURM_JOB_NAME:-} partition=${SLURM_JOB_PARTITION:-} node=${SLURMD_NODENAME:-$(hostname)} gpus=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
        echo "arpl: AMP=$ARPL_AMP SDPA=$ARPL_SDPA CEM_CHUNK=${ARPL_CEM_CHUNK:-50}"
        echo "python: $(command -v python) $(python --version 2>&1)"
        python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)" 2>&1
        echo "nvidia-smi:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | sed 's/^/  /'
    } > "$f"
    git -C "$ARPL_REPO" diff HEAD > "${f%.txt}.diff"   # uncommitted changes, empty when clean
    echo "[ARPL-WM] run info: $f"
}
