"""[ARPL-WM] Start/goal sampling and collision-free path planning on the room's floor plan.

Planning is 2D (x, y) on an occupancy grid at the flight altitude, with altitude interpolated
linearly from the start height to the goal height. Steps:
  1. A* on an 8-connected grid (obstacles inflated by `clearance`).
  2. Shortcutting: drop waypoints while the straight segment stays collision-free.
  3. Resampling to evenly spaced points, which the path follower chases.
Written new for this fork; no planner code comes from Genesis or transport-world.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

from .room import Grid, Room


@dataclass
class PlannerConfig:
    resolution: float = 0.1          # grid cell, m
    clearance: float = 0.45          # drone half-size (~0.15 m) + safety margin, m
    z_range: Tuple[float, float] = (1.0, 2.0)   # start and goal heights are drawn from this range
    min_distance: float = 3.0        # straight-line start-goal distance, m
    max_distance: Optional[float] = None
    spacing: float = 0.1             # distance between resampled waypoints, m
    shortcut: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _line_free(grid: Grid, a, b) -> bool:
    n = max(2, int(math.ceil(np.hypot(b[0] - a[0], b[1] - a[1]) / (grid.res / 2))))
    for t in np.linspace(0.0, 1.0, n):
        if not grid.free_xy(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])):
            return False
    return True


def astar(grid: Grid, start_xy, goal_xy) -> Optional[np.ndarray]:
    """Shortest 8-connected grid path, returned as (N, 2) world points, or None."""
    s, g = grid.to_cell(*start_xy), grid.to_cell(*goal_xy)
    if not grid.free(*s) or not grid.free(*g):
        return None
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]

    def h(c):  # octile distance, admissible for 8-connected moves
        dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    open_heap = [(h(s), 0.0, s)]
    came, cost = {s: None}, {s: 0.0}
    while open_heap:
        _, c_cost, c = heapq.heappop(open_heap)
        if c == g:
            break
        if c_cost > cost[c]:
            continue
        for di, dj, w in moves:
            n = (c[0] + di, c[1] + dj)
            if not grid.free(*n):
                continue
            if di and dj and not (grid.free(c[0] + di, c[1]) and grid.free(c[0], c[1] + dj)):
                continue  # no corner cutting
            nc = c_cost + w
            if nc < cost.get(n, math.inf):
                cost[n], came[n] = nc, c
                heapq.heappush(open_heap, (nc + h(n), nc, n))
    if g not in came:
        return None
    cells, c = [], g
    while c is not None:
        cells.append(c); c = came[c]
    pts = np.array([grid.to_world(*c) for c in cells[::-1]])
    pts[0], pts[-1] = start_xy, goal_xy
    return pts


def shortcut(grid: Grid, pts: np.ndarray) -> np.ndarray:
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _line_free(grid, pts[i], pts[j]):
            j -= 1
        out.append(pts[j]); i = j
    return np.array(out)


def resample(pts: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(s[-1] / spacing)) + 1)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, pts[:, k]) for k in range(pts.shape[1])], axis=1)


def plan_path(room: Room, start, goal, cfg: PlannerConfig) -> Optional[np.ndarray]:
    """(N, 3) collision-free waypoints from start to goal, or None when no path exists."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    grid = room.occupancy_grid(cfg.resolution, cfg.clearance, (min(start[2], goal[2]), max(start[2], goal[2])))
    xy = astar(grid, start[:2], goal[:2])
    if xy is None:
        return None
    if cfg.shortcut:
        xy = shortcut(grid, xy)
    xy = resample(xy, cfg.spacing)
    frac = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    frac = frac / max(frac[-1], 1e-9)
    z = start[2] + frac * (goal[2] - start[2])
    return np.column_stack([xy, z])


def sample_start_goal(room: Room, cfg: PlannerConfig, rng: np.random.Generator, max_tries: int = 1000):
    """Random start and goal in free space, at least `min_distance` apart, with a path between them."""
    for _ in range(max_tries):
        a, b = (np.array([rng.uniform(-room.half_x, room.half_x),
                          rng.uniform(-room.half_y, room.half_y),
                          rng.uniform(*cfg.z_range)]) for _ in range(2))
        d = float(np.linalg.norm(a[:2] - b[:2]))
        if d < cfg.min_distance or (cfg.max_distance is not None and d > cfg.max_distance):
            continue
        path = plan_path(room, a, b, cfg)
        if path is not None:
            return a, b, path
    raise RuntimeError("no valid start/goal pair found; loosen min_distance or clearance")
