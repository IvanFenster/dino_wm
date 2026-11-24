#!/bin/bash
# [ARPL-WM] Planning evaluation (CEM + MPC) of one checkpoint. Submit from the repo root:
#   sbatch slurm/plan.sbatch <model_name> <epoch> <seed> [max_rounds] [config] [hydra overrides...]
#   max_rounds: MPC replanning cap. "null" = the repo's protocol (replan without limit). Our point_maze_2s
#   numbers (48/50) used 3. config defaults to plan_point_maze.yaml. 10 episodes per job (n_evals=10).
# Example, one of the 5 parts of the 50-episode PointMaze evaluation:
#   sbatch slurm/plan.sbatch point_maze_2s 10 99 3
#SBATCH --job-name=dinowm_plan
#SBATCH --account=fc_auto
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -eo pipefail
MODEL=${1:?usage: sbatch slurm/plan.sbatch <model_name> <epoch> <seed> [max_rounds] [config] [overrides...]}
EPOCH=${2:?epoch}; SEED=${3:?seed}; ROUNDS=${4:-null}; CONFIG=${5:-plan_point_maze.yaml}
shift $(( $# < 5 ? $# : 5 ))
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/env.sh"
OUT=$WM/plan_outputs/${MODEL}_ep${EPOCH}_s${SEED}_r${ROUNDS}
CMD=(python plan.py --config-name "$CONFIG" ckpt_base_path=$WM model_name=$MODEL model_epoch=$EPOCH
     n_evals=${N_EVALS:-10} seed=$SEED planner.max_iter=$ROUNDS hydra.run.dir=$OUT "$@")
cd "$ARPL_REPO"
arpl_record_run "$OUT/run_info" "${CMD[@]}"
"${CMD[@]}" 2>&1 | tee "$OUT.log"
grep -h "success_rate" "$OUT/logs.json" 2>/dev/null | tail -3 || true
