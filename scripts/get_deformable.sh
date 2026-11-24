#!/bin/bash
# [ARPL-WM] Download DINO-WM's deformable dataset (Rope + Granular, with depth) from the Hugging Face mirror
# gaoyuezhou/dino_wm (byte-identical to OSF), join the split zip, and extract it.
# Moved from Depth_addition.ipynb (cells 1, 2 and 8). Run on Savio's transfer node (dtn.brc.berkeley.edu)
# or inside a job, never on the login node. 15.8 GB download.
# Usage: bash scripts/get_deformable.sh <dest_dir> [granular|rope|all] [n_episodes]
#   dest_dir/deformable/<task>/... is what DATASET_DIR expects. Set HF_TOKEN to avoid throttling.
set -euo pipefail
DEST=${1:?usage: get_deformable.sh <dest_dir> [granular|rope|all] [n_episodes]}
TASK=${2:-all}
N_EP=${3:-}
DL="$DEST/_download"
mkdir -p "$DL"

python - "$DL" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
for f in ["deformable.z01", "deformable.z02", "deformable.z03", "deformable.zip"]:
    print(hf_hub_download("gaoyuezhou/dino_wm", f"datasets/{f}", repo_type="dataset", local_dir=sys.argv[1]))
EOF

cd "$DL/datasets"
[ -f deformable_full.zip ] || zip -s- deformable.zip -O deformable_full.zip   # join the split archive
if [ "$TASK" = all ]; then
    unzip -q -n deformable_full.zip -d "$DEST"
elif [ -z "$N_EP" ]; then
    unzip -q -n deformable_full.zip "deformable/$TASK/*" -d "$DEST"
else  # first N episodes only, like the depth pilot (frame 20 is not used for training)
    pats=$(for i in $(seq 0 $((N_EP - 1))); do printf '"deformable/%s/%06d/*" ' "$TASK" "$i"; done)
    eval unzip -q -n deformable_full.zip "deformable/$TASK/actions.pth" "deformable/$TASK/states.pth" $pats -x '"*/20.h5"' -d "$DEST"
fi
du -sh "$DEST/deformable"
echo "Done. Keep $DL/datasets/deformable_full.zip as the persistent copy, or delete it to save space."
