# [ARPL-WM] Validation loss of one saved checkpoint, computed exactly as train.py does at the end of an epoch
# (same config, same 90/10 split from the same seed, plain fp32, original attention, dropout off).
# Usage: python val_ckpt.py <repo dir> <run dir> <epoch>
import os, sys, tempfile
from collections import OrderedDict
import torch
from omegaconf import OmegaConf

repo, run_dir, epoch = sys.argv[1], sys.argv[2], int(sys.argv[3])
ckpt = os.path.join(run_dir, "checkpoints", f"model_{epoch}.pth")
assert os.path.exists(ckpt), f"no checkpoint {ckpt}"

if not torch.cuda.is_available():  # CPU fallback: checkpoints hold GPU tensors
    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "map_location": k.get("map_location", "cpu")})

os.environ.pop("ARPL_AMP", None)   # validation is fp32 anyway
if os.environ.get("ARPL_VAL_SDPA") == "1":   # fused attention, as during training-time validation
    os.environ["ARPL_SDPA"] = "1"
else:                                        # authors' original attention code
    os.environ.pop("ARPL_SDPA", None)
print("attention:", "fused (ARPL_SDPA=1)" if os.environ.get("ARPL_SDPA") == "1" else "original")

cfg = OmegaConf.load(os.path.join(run_dir, "hydra.yaml"))
work = tempfile.mkdtemp(prefix=f"val_ep{epoch}_")      # scratch run folder, nothing is written to the real one
os.makedirs(os.path.join(work, "checkpoints"))
os.symlink(ckpt, os.path.join(work, "checkpoints", "model_latest.pth"))
cfg.saved_folder = work
os.chdir(work)
sys.path.insert(0, repo)
import train

class _NotMultirun:  # train.py asks Hydra whether it is a multirun; we are not running under @hydra.main
    mode = train.RunMode.RUN
train.HydraConfig.get = lambda: _NotMultirun

trainer = train.Trainer(cfg)
assert trainer.epoch == epoch, f"loaded epoch {trainer.epoch}, expected {epoch}"
trainer.train_traj_dset = []   # skip the random open-loop rollout plots; they are not part of val_loss
trainer.epoch_log = OrderedDict()
trainer.val()
count, total = trainer.epoch_log["val_loss"]
print(f"\nRESULT epoch {epoch}: validation loss {total / count:.4f}  ({count} batches)")
for k in ("val_z_loss", "val_z_visual_loss", "val_z_proprio_loss"):
    if k in trainer.epoch_log:
        c, t = trainer.epoch_log[k]; print(f"  {k}: {t / c:.4f}")
