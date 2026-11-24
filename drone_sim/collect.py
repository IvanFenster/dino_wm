"""[ARPL-WM] Fly planned start-to-goal trajectories and save them; pack them for DINO-WM.

Raw format (one file per episode, written while collecting, safe to run many jobs in parallel):
    <out>/raw/room<R>_ep<E>.npz
        rgb     uint8   (T+1, S, S, 3)
        depth   float16 (T+1, S, S)   meters, clipped to max_depth
        state   float32 (T+1, 13)     position, quaternion wxyz, linear vel, angular vel (world)
        action  float32 (T, 4)        [forward, left, up, yaw_rate]; action[t] moves frame t -> t+1
        collision, clearance (T,)     per step
        path    float32 (N, 3)        planned waypoints
        meta    json string           start, goal, room config, planner/follower config, outcome, timing

DINO-WM format (`pack_dataset`), same layout as the PointMaze dataset so the loaders look alike:
    <dset>/states.pth       float32 (N, T_max, 13), zero padded
    <dset>/actions.pth      float32 (N, T_max, 4),  zero padded; the last real frame gets a zero action
    <dset>/seq_lengths.pth  int64 (N,) frames per episode (T+1)
    <dset>/obses/episode_XXX.pth   uint8 (T+1, S, S, 3)
    <dset>/depths/episode_XXX.pth  float16 (T+1, S, S) meters
    <dset>/episodes.json    per-episode metadata (source file, room seed, start, goal, outcome)
    <dset>/dataset_info.json
Written new for this fork.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .follower import FollowerConfig, PathFollower
from .planner import PlannerConfig, plan_path, sample_start_goal


@dataclass
class EpisodeConfig:
    max_steps: Optional[int] = None      # None: 2 x (path length / cruise speed) + 30 steps
    hover_steps: int = 5                 # keep recording this many steps after reaching the goal
    stop_on_collision: bool = True       # end the episode at the first collision
    start_yaw: str = "path"              # "path": face the first path segment; "random"
    speed_range: Optional[tuple] = None  # draw cruise_speed per episode from this range (diversity)

    def to_dict(self) -> dict:
        return asdict(self)


def run_episode(env, planner_cfg: PlannerConfig, follower_cfg: FollowerConfig, ep_cfg: EpisodeConfig,
                rng: np.random.Generator, start=None, goal=None) -> dict:
    """Plan, fly and record one trajectory. Returns a dict of arrays plus `meta`."""
    room = env.room
    if start is None or goal is None:
        start, goal, path = sample_start_goal(room, planner_cfg, rng)
    else:
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        path = plan_path(room, start, goal, planner_cfg)
        if path is None:
            raise ValueError(f"no collision-free path from {start} to {goal}")

    fcfg = FollowerConfig(**follower_cfg.to_dict())
    if ep_cfg.speed_range is not None:
        fcfg.cruise_speed = float(rng.uniform(*ep_cfg.speed_range))
    follower = PathFollower(path, fcfg, rng)
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    dt_action = env.dt * env.substeps
    max_steps = ep_cfg.max_steps or int(2 * length / fcfg.cruise_speed / dt_action) + 30

    if ep_cfg.start_yaw == "random":
        yaw0 = float(rng.uniform(-math.pi, math.pi))
    else:
        k = min(5, len(path) - 1)
        yaw0 = math.atan2(path[k, 1] - path[0, 1], path[k, 0] - path[0, 0])

    t0 = time.time()
    obs = env.reset(position=start, yaw=yaw0)
    rgb, depth, state, action, coll, clear = [obs["rgb"]], [obs["depth"]], [obs["state"]], [], [], []
    outcome, hover_left = "timeout", None
    for _ in range(max_steps):
        a = follower(obs["state"])
        if hover_left is not None:
            a = np.zeros(4, np.float32)
        obs, info = env.step(a)
        rgb.append(obs["rgb"]); depth.append(obs["depth"]); state.append(obs["state"]); action.append(a)
        coll.append(info["collision"]); clear.append(info["clearance"])
        if info["crashed"]:
            outcome = "crashed"; break
        if info["collision"] and ep_cfg.stop_on_collision:
            outcome = "collision"; break
        if hover_left is None and follower.reached(obs["state"][:3]):
            hover_left = ep_cfg.hover_steps
        if hover_left is not None:
            if hover_left == 0:
                outcome = "success"; break
            hover_left -= 1
    elapsed = time.time() - t0
    final = np.asarray(state[-1][:3])
    meta = {
        "start": start.tolist(), "goal": goal.tolist(), "start_yaw": yaw0,
        "outcome": outcome, "steps": len(action), "path_length_m": length,
        "final_goal_distance_m": float(np.linalg.norm(final - goal)),
        "any_collision": bool(np.any(coll)), "min_clearance_m": float(np.min(clear)) if clear else None,
        "cruise_speed": fcfg.cruise_speed, "seconds": elapsed,
        "steps_per_second": len(action) / max(elapsed, 1e-9),
        "room": room.cfg.to_dict(), "planner": planner_cfg.to_dict(), "follower": fcfg.to_dict(),
        "episode": ep_cfg.to_dict(),
        "env": {"image_size": env.image_size, "dt": env.dt, "substeps": env.substeps,
                "action_scale": env.action_scale, "yaw_rate_scale": env.yaw_rate_scale,
                "rpm_tau": env.rpm_tau, "max_depth": env.max_depth, "collision_radius": env.collision_radius},
    }
    return {
        "rgb": np.stack(rgb).astype(np.uint8),
        "depth": np.stack(depth).astype(np.float16),
        "state": np.stack(state).astype(np.float32),
        "action": np.stack(action).astype(np.float32) if action else np.zeros((0, 4), np.float32),
        "collision": np.asarray(coll, bool), "clearance": np.asarray(clear, np.float32),
        "path": path.astype(np.float32), "meta": meta,
    }


def save_episode(ep: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {k: v for k, v in ep.items() if k != "meta"}
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(ep["meta"])), **arrays)
    os.replace(tmp, path)       # never leave a half-written episode behind if a job is killed
    return path


def load_episode(path: str) -> dict:
    with np.load(path, allow_pickle=False) as f:
        ep = {k: f[k] for k in f.files if k != "meta"}
        ep["meta"] = json.loads(str(f["meta"]))
    return ep


def pack_dataset(raw_glob: str, out_dir: str, only_success: bool = False, min_steps: int = 2) -> dict:
    """Convert raw episode files into the DINO-WM layout described at the top of this file."""
    import torch

    files = sorted(glob.glob(raw_glob))
    if not files:
        raise FileNotFoundError(raw_glob)
    os.makedirs(os.path.join(out_dir, "obses"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depths"), exist_ok=True)
    states, actions, lengths, metas, skipped = [], [], [], [], 0
    for f in files:
        ep = load_episode(f)
        if (only_success and ep["meta"]["outcome"] != "success") or len(ep["action"]) < min_steps:
            skipped += 1
            continue
        i = len(lengths)
        T1 = len(ep["state"])
        act = np.concatenate([ep["action"], np.zeros((1, 4), np.float32)])   # pad: one action per frame
        torch.save(torch.from_numpy(ep["rgb"]), os.path.join(out_dir, "obses", f"episode_{i:03d}.pth"))
        torch.save(torch.from_numpy(ep["depth"]), os.path.join(out_dir, "depths", f"episode_{i:03d}.pth"))
        states.append(ep["state"]); actions.append(act); lengths.append(T1)
        m = dict(ep["meta"]); m["source_file"] = os.path.basename(f); m["index"] = i
        m.pop("planner", None); m.pop("follower", None); m.pop("env", None); m.pop("episode", None)
        metas.append(m)
    T = max(lengths)
    S = np.zeros((len(lengths), T, 13), np.float32)
    A = np.zeros((len(lengths), T, 4), np.float32)
    for i, (s, a) in enumerate(zip(states, actions)):
        S[i, :len(s)] = s; A[i, :len(a)] = a
    torch.save(torch.from_numpy(S), os.path.join(out_dir, "states.pth"))
    torch.save(torch.from_numpy(A), os.path.join(out_dir, "actions.pth"))
    torch.save(torch.tensor(lengths, dtype=torch.long), os.path.join(out_dir, "seq_lengths.pth"))
    with open(os.path.join(out_dir, "episodes.json"), "w") as f:
        json.dump(metas, f, indent=1)
    first = load_episode(files[0])["meta"]
    outcomes = {o: sum(m["outcome"] == o for m in metas) for o in ("success", "collision", "crashed", "timeout")}
    info = {
        "n_episodes": len(lengths), "skipped": skipped, "max_len": T,
        "total_frames": int(sum(lengths)), "outcomes": outcomes,
        "rooms": sorted({m["room"]["seed"] for m in metas}),
        "state_layout": "pos(3) quat_wxyz(4) lin_vel(3) ang_vel(3), world frame",
        "action_layout": "forward, left, up, yaw_rate in [-1,1], heading frame; action[t]: frame t -> t+1",
        "depth_units": "meters, float16", "image_size": first["env"]["image_size"],
        "env": first["env"], "packed_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info
