#!/bin/bash
# [ARPL-WM] Train (or resume) a DINO-WM run until TARGET_EPOCH exists. Submit from the repo root:
#   sbatch slurm/train.sbatch <run_name> <target_epoch> [hydra overrides...]
# Example, our PointMaze reproduction (point_maze_2s settings):
#   sbatch slurm/train.sbatch point_maze_2s 10 env=point_maze frameskip=5 num_hist=3 has_decoder=False model.train_decoder=False
# Resubmitting the same command resumes from the last checkpoint (train.py runs `training.epochs` MORE epochs,
# so this script passes only the number still missing). Longer than 24 h: just resubmit after it ends.
#SBATCH --job-name=dinowm_train
#SBATCH --account=fc_auto
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -eo pipefail
RUN=${1:?usage: sbatch slurm/train.sbatch <run_name> <target_epoch> [hydra overrides...]}
TARGET=${2:?target epoch}
shift 2
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/slurm/env.sh"
BATCH=${BATCH:-32}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}

RUN_DIR=$WM/outputs/$RUN
DONE=$(ls "$RUN_DIR/checkpoints" 2>/dev/null | sed -n 's/^model_\([0-9]\+\)\.pth$/\1/p' | sort -n | tail -1 || true)
DONE=${DONE:-0}
REMAINING=$((TARGET - DONE))
echo "[ARPL-WM] $RUN: last saved epoch $DONE, training $REMAINING more to reach $TARGET"
[ "$REMAINING" -gt 0 ] || exit 0

export ARPL_RUN=$RUN ARPL_BATCH=$BATCH ARPL_EPOCHS=$REMAINING ARPL_SESSION=${SLURM_JOB_ID:-$(date -u +%Y%m%d_%H%M)}
CMD=(python train.py training.batch_size=$BATCH env.num_workers=$WORKERS training.epochs=$REMAINING
     training.save_every_x_epoch=1 ckpt_base_path=$WM hydra.run.dir=$RUN_DIR "$@")
cd "$ARPL_REPO"
arpl_record_run "$RUN_DIR/run_info" "${CMD[@]}"
STDOUT=$WM/logs/$RUN/train_$ARPL_SESSION.log
mkdir -p "$(dirname "$STDOUT")"

"${CMD[@]}" > "$STDOUT" 2>&1 &
TRAIN_PID=$!
python scripts/arpl_logger.py "$RUN" "$STDOUT" "$ARPL_SESSION" "$TRAIN_PID" > "$WM/logs/$RUN/logger_$ARPL_SESSION.out" 2>&1 &
tail -n +1 -f "$STDOUT" --pid=$TRAIN_PID &   # also show training output in the Slurm .out file
STATUS=0; wait $TRAIN_PID || STATUS=$?
wait   # let the logger write its final pass
echo "[ARPL-WM] train.py exited with $STATUS"
exit $STATUS
