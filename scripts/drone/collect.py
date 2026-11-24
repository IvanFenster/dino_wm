#!/usr/bin/env python
"""[ARPL-WM] Collect planned drone trajectories in one Genesis room (batch version of the notebook).

One process = one room, because building a Genesis scene compiles kernels (slow) and the pillars
are fixed at build time. Run several rooms as a Slurm array (slurm/drone_collect.sbatch), where
the array index becomes the room seed. Episodes go to <out>/raw/room<seed>_ep<k>.npz; rerunning
skips episodes that already exist, so a killed job can simply be resubmitted.

    python scripts/drone/collect.py --out $WM/data/drone_raw --room-seed 0 --episodes 200
    python scripts/drone/collect.py --pack --out $WM/data/drone_raw --dset $WM/data/drone_nav
New for this fork.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

p = argparse.ArgumentParser()
p.add_argument("--out", required=True, help="folder for raw episodes (raw/) and logs (logs/)")
p.add_argument("--room-seed", type=int, default=0)
p.add_argument("--room-size", type=float, nargs=2, default=(12.0, 12.0), metavar=("X", "Y"))
p.add_argument("--num-pillars", type=int, default=14)
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--seed", type=int, default=None, help="start/goal seed (default: 1000 + room seed)")
p.add_argument("--image-size", type=int, default=224)
p.add_argument("--z-range", type=float, nargs=2, default=(1.0, 2.0))
p.add_argument("--min-distance", type=float, default=3.0)
p.add_argument("--clearance", type=float, default=0.45)
p.add_argument("--speed-range", type=float, nargs=2, default=(0.6, 1.5))
p.add_argument("--noise", type=float, default=0.1, help="action noise std (diversity for the world model)")
p.add_argument("--start-yaw", choices=["path", "random"], default="path")
p.add_argument("--keep-collisions", action="store_true", help="do not stop episodes at the first collision")
p.add_argument("--pack", action="store_true", help="only pack <out>/raw/*.npz into --dset and exit")
p.add_argument("--dset", default=None)
p.add_argument("--only-success", action="store_true")
args = p.parse_args()

from drone_sim.collect import EpisodeConfig, pack_dataset, run_episode, save_episode  # noqa: E402

if args.pack:
    info = pack_dataset(os.path.join(args.out, "raw", "*.npz"), args.dset, only_success=args.only_success)
    print(json.dumps(info, indent=1))
    sys.exit(0)

from drone_sim.egl import setup_gpu_rendering  # noqa: E402
setup_gpu_rendering()
import numpy as np  # noqa: E402
import torch  # noqa: E402

from drone_sim.follower import FollowerConfig  # noqa: E402
from drone_sim.planner import PlannerConfig  # noqa: E402
from drone_sim.room import RoomConfig  # noqa: E402

os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
job = os.environ.get("SLURM_JOB_ID", "local")
log_path = os.path.join(args.out, "logs", f"room{args.room_seed:03d}_{job}.jsonl")


def git(*a):
    try:
        return subprocess.check_output(["git", "-C", REPO, *a], text=True).strip()
    except Exception:
        return None


run_info = {
    "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "args": vars(args),
    "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
    "slurm_job": job, "node": platform.node(), "python": sys.version.split()[0], "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
try:
    import importlib.metadata as md
    run_info["genesis"] = md.version("genesis-world")
except Exception:
    pass

t_build = time.time()
from drone_sim.env import DroneFPVEnv  # noqa: E402

room_cfg = RoomConfig(size_x=args.room_size[0], size_y=args.room_size[1], num_pillars=args.num_pillars,
                      seed=args.room_seed)
env = DroneFPVEnv(room_cfg, image_size=args.image_size)
t0 = time.time(); env.render(); run_info["render_seconds_per_frame"] = time.time() - t0
run_info["build_seconds"] = time.time() - t_build
print(json.dumps(run_info, indent=1))
with open(log_path, "a") as f:
    f.write(json.dumps({"run_info": run_info}) + "\n")
if run_info["render_seconds_per_frame"] > 0.3:
    print("WARNING: rendering looks CPU-bound (>0.3 s/frame). Check the [egl] line above.")

planner_cfg = PlannerConfig(z_range=tuple(args.z_range), min_distance=args.min_distance, clearance=args.clearance)
follower_cfg = FollowerConfig(action_noise_std=args.noise, yaw_rate_scale=env.yaw_rate_scale)
ep_cfg = EpisodeConfig(stop_on_collision=not args.keep_collisions, start_yaw=args.start_yaw,
                       speed_range=tuple(args.speed_range))
seed = args.seed if args.seed is not None else 1000 + args.room_seed

for k in range(args.episodes):
    path = os.path.join(args.out, "raw", f"room{args.room_seed:03d}_ep{k:05d}.npz")
    if os.path.exists(path):
        continue
    rng = np.random.default_rng([seed, k])     # episode k is reproducible on its own
    ep = run_episode(env, planner_cfg, follower_cfg, ep_cfg, rng)
    save_episode(ep, path)
    m = ep["meta"]
    row = {"episode": k, "file": os.path.basename(path), **{x: m[x] for x in (
        "outcome", "steps", "path_length_m", "final_goal_distance_m", "any_collision",
        "min_clearance_m", "cruise_speed", "seconds", "steps_per_second")}}
    with open(log_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[room {args.room_seed} ep {k}] {m['outcome']:9s} steps {m['steps']:4d} "
          f"path {m['path_length_m']:.1f} m  goal err {m['final_goal_distance_m']:.2f} m  "
          f"{m['steps_per_second']:.1f} steps/s", flush=True)
