"""[ARPL-WM] Pure-pursuit path follower that outputs the environment's 4-D action.

The environment's action is [forward, left, up, yaw_rate] in [-1, 1], in the drone's heading
frame. Because the SE(3) controller chases "current position + action", the action behaves like
a velocity command (measured in Colab: about 2.96 m/s per unit horizontally, 1.78 m/s per unit
vertically, time constant about 0.42 s). So the follower picks a desired world velocity toward a
look-ahead point on the path, rotates it into the heading frame and divides by those gains.

Yaw: the drone turns to face its direction of travel, so the forward camera sees where it goes,
and it slows down while the heading error is large.

Optional action noise (held for `noise_hold` steps) widens the action distribution, which a world
model needs: with perfectly smooth expert actions it never sees what other actions do.
Written new for this fork.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


def yaw_from_quat_wxyz(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class FollowerConfig:
    cruise_speed: float = 1.0        # m/s along the path
    lookahead: float = 0.6           # m
    goal_tolerance: float = 0.25     # m, 3D distance that counts as reached
    slow_radius: float = 1.0         # start slowing down this far from the goal, m
    speed_per_action_xy: float = 2.96
    speed_per_action_z: float = 1.78
    yaw_rate_scale: float = 1.0      # must match DroneFPVEnv.yaw_rate_scale (rad/s per unit)
    yaw_gain: float = 2.0            # yaw_rate = gain * heading error
    face_motion: bool = True
    turn_slowdown: bool = True       # scale speed by cos(heading error), floor 0.15
    action_noise_std: float = 0.0    # Gaussian noise added to each action dimension
    noise_hold: int = 5              # hold each noise sample this many steps

    def to_dict(self) -> dict:
        return asdict(self)


class PathFollower:
    def __init__(self, path: np.ndarray, cfg: FollowerConfig, rng: Optional[np.random.Generator] = None):
        self.path = np.asarray(path, float)
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(0)
        self.idx = 0
        self.t = 0
        self.noise = np.zeros(4)
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))])

    def reached(self, pos) -> bool:
        return float(np.linalg.norm(np.asarray(pos) - self.path[-1])) < self.cfg.goal_tolerance

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """state: 13-D [pos, quat wxyz, lin vel, ang vel] -> action (4,) float32 in [-1, 1]."""
        cfg, pos = self.cfg, np.asarray(state[:3], float)
        yaw = yaw_from_quat_wxyz(state[3:7])

        # Progress: nearest waypoint, searched only forward of the last one (no jumping back).
        window = self.path[self.idx: self.idx + 40]
        self.idx += int(np.argmin(np.linalg.norm(window - pos, axis=1)))
        target_s = self.s[self.idx] + cfg.lookahead
        j = min(int(np.searchsorted(self.s, target_s)), len(self.path) - 1)
        target = self.path[j]

        to_target = target - pos
        dist_goal = float(np.linalg.norm(self.path[-1] - pos))
        speed = cfg.cruise_speed * min(1.0, dist_goal / cfg.slow_radius)
        direction = to_target / max(np.linalg.norm(to_target), 1e-6)
        heading_err = 0.0
        if cfg.face_motion and np.linalg.norm(to_target[:2]) > 0.05:
            heading_err = wrap(math.atan2(to_target[1], to_target[0]) - yaw)
            if cfg.turn_slowdown:
                speed *= max(0.15, math.cos(heading_err))
        v = direction * speed

        c, s_ = math.cos(yaw), math.sin(yaw)
        forward = (c * v[0] + s_ * v[1]) / cfg.speed_per_action_xy
        left = (-s_ * v[0] + c * v[1]) / cfg.speed_per_action_xy
        up = v[2] / cfg.speed_per_action_z
        yaw_rate = cfg.yaw_gain * heading_err / cfg.yaw_rate_scale
        action = np.array([forward, left, up, yaw_rate])

        if cfg.action_noise_std > 0:
            if self.t % cfg.noise_hold == 0:
                self.noise = self.rng.normal(0.0, cfg.action_noise_std, size=4)
            action = action + self.noise
        self.t += 1
        return np.clip(action, -1.0, 1.0).astype(np.float32)
