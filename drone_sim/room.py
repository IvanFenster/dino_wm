"""[ARPL-WM] Room layout for the Genesis drone environment, independent of Genesis.

A room is a rectangle with four walls plus box obstacles ("pillars"), all axis-aligned.
Keeping the layout as plain numbers (not Genesis entities) lets the planner, the collision
check and the plots use the exact same geometry the simulator builds.

Default values reproduce the Colab arena (`DroneFPVEnv._add_arena` in the "Drone environment"
notebook): 12 x 12 m, 3 m walls, 14 pillars, the same random draws in the same order, so
`RoomConfig(seed=0)` gives the same pillars as `DroneFPVEnv(seed=0)` in Colab.

To grow the room, change `size_x` / `size_y` / `num_pillars`, or add your own boxes
(interior walls, doorways) through `extra_boxes`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

WALL_COLORS = [(0.85, 0.35, 0.25), (0.25, 0.55, 0.85), (0.35, 0.75, 0.40), (0.90, 0.75, 0.25)]


@dataclass
class Box:
    """Axis-aligned box: center (x, y, z) and full size (sx, sy, sz), in meters."""
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    color: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    kind: str = "pillar"            # "wall", "pillar" or "extra"

    @property
    def lo(self) -> np.ndarray:
        return np.asarray(self.center) - np.asarray(self.size) / 2

    @property
    def hi(self) -> np.ndarray:
        return np.asarray(self.center) + np.asarray(self.size) / 2


@dataclass
class RoomConfig:
    size_x: float = 12.0              # inner extent along x, wall center to wall center (Colab: 2 * 6)
    size_y: float = 12.0
    wall_height: float = 3.0
    wall_thickness: float = 0.2
    num_pillars: int = 14
    pillar_width: Tuple[float, float] = (0.3, 0.8)    # square footprint, uniform range
    pillar_height: Tuple[float, float] = (0.6, 2.8)
    pillar_margin: float = 0.8        # pillar centers stay this far from the walls
    spawn_clear_radius: float = 1.5   # no pillar center within this radius of (0, 0)
    extra_boxes: List[Box] = field(default_factory=list)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RoomConfig":
        d = dict(d)
        d["extra_boxes"] = [Box(**b) if isinstance(b, dict) else b for b in d.get("extra_boxes", [])]
        for k in ("pillar_width", "pillar_height"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


class Room:
    """Concrete layout drawn from a RoomConfig. `boxes` is everything the simulator builds."""

    def __init__(self, cfg: RoomConfig):
        self.cfg = cfg
        self.half_x = cfg.size_x / 2
        self.half_y = cfg.size_y / 2
        self.boxes: List[Box] = self._walls() + self._pillars() + list(cfg.extra_boxes)

    # Same geometry and colors as the Colab arena.
    def _walls(self) -> List[Box]:
        hx, hy, h, t = self.half_x, self.half_y, self.cfg.wall_height, self.cfg.wall_thickness
        specs = [((hx, 0, h / 2), (t, 2 * hy, h)), ((-hx, 0, h / 2), (t, 2 * hy, h)),
                 ((0, hy, h / 2), (2 * hx, t, h)), ((0, -hy, h / 2), (2 * hx, t, h))]
        return [Box(tuple(map(float, p)), tuple(map(float, s)), c, "wall") for (p, s), c in zip(specs, WALL_COLORS)]

    def _pillars(self) -> List[Box]:
        # Draw order matches the Colab code: (x, y), then width, height, color. Do not reorder.
        cfg, rng = self.cfg, np.random.default_rng(self.cfg.seed)
        m = cfg.pillar_margin
        lo = np.array([-self.half_x + m, -self.half_y + m])
        hi = np.array([self.half_x - m, self.half_y - m])
        pillars = []
        while len(pillars) < cfg.num_pillars:
            x, y = rng.uniform(lo, hi, size=2)
            if math.hypot(x, y) < cfg.spawn_clear_radius:
                continue
            width = float(rng.uniform(*cfg.pillar_width))
            height = float(rng.uniform(*cfg.pillar_height))
            color = tuple(float(c) for c in rng.uniform(0.15, 0.95, size=3))
            pillars.append(Box((float(x), float(y), height / 2), (width, width, height), color, "pillar"))
        return pillars

    # ------------------------------------------------------------------ geometry
    def point_clearance(self, p: Sequence[float]) -> float:
        """Distance from a 3D point to the nearest box surface (negative inside a box) or the floor."""
        p = np.asarray(p, dtype=np.float64)
        best = p[2]                                    # floor at z = 0
        for b in self.boxes:
            d = np.maximum(b.lo - p, 0) + np.maximum(p - b.hi, 0)
            outside = float(np.linalg.norm(d))
            if outside > 0:
                best = min(best, outside)
            else:
                best = min(best, -float(np.min(np.minimum(p - b.lo, b.hi - p))))
        return best

    def inside(self, p: Sequence[float]) -> bool:
        t = self.cfg.wall_thickness / 2
        return abs(p[0]) < self.half_x - t and abs(p[1]) < self.half_y - t and 0 < p[2] < self.cfg.wall_height

    def occupancy_grid(self, resolution: float, clearance: float, z_band: Tuple[float, float]) -> "Grid":
        """2D grid of the floor plan. A cell is blocked when it lies within `clearance` (meters)
        of a box whose height range overlaps z_band widened by `clearance`. So the drone may fly
        over a short pillar when the whole path is high enough."""
        x0, y0 = -self.half_x, -self.half_y
        nx = int(math.ceil(self.cfg.size_x / resolution))
        ny = int(math.ceil(self.cfg.size_y / resolution))
        xs = x0 + (np.arange(nx) + 0.5) * resolution
        ys = y0 + (np.arange(ny) + 0.5) * resolution
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        blocked = np.zeros((nx, ny), dtype=bool)
        zlo, zhi = z_band[0] - clearance, z_band[1] + clearance
        for b in self.boxes:
            if b.hi[2] < zlo or b.lo[2] > zhi:
                continue
            dx = np.maximum(np.maximum(b.lo[0] - X, X - b.hi[0]), 0)
            dy = np.maximum(np.maximum(b.lo[1] - Y, Y - b.hi[1]), 0)
            blocked |= np.hypot(dx, dy) < clearance
        return Grid(blocked, x0, y0, resolution)

    def plot(self, ax=None, path=None, flown=None, start=None, goal=None, title=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        ax = ax or plt.subplots(figsize=(6, 6))[1]
        for b in self.boxes:
            ax.add_patch(Rectangle(b.lo[:2], b.size[0], b.size[1], color=b.color, alpha=0.9))
            if b.kind == "pillar":
                ax.text(b.center[0], b.center[1], f"{b.size[2]:.1f}", fontsize=6, ha="center", va="center")
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], "k--", lw=1, label="planned")
        if flown is not None:
            ax.plot(flown[:, 0], flown[:, 1], "m-", lw=1.5, label="flown")
        if start is not None:
            ax.scatter(*start[:2], c="g", s=60, zorder=5, label="start")
        if goal is not None:
            ax.scatter(*goal[:2], c="r", marker="*", s=120, zorder=5, label="goal")
        ax.set_xlim(-self.half_x - 0.3, self.half_x + 0.3)
        ax.set_ylim(-self.half_y - 0.3, self.half_y + 0.3)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(title or f"room seed {self.cfg.seed} (numbers = pillar height, m)")
        if path is not None or start is not None:
            ax.legend(loc="upper right", fontsize=7)
        return ax


@dataclass
class Grid:
    blocked: np.ndarray      # (nx, ny) bool
    x0: float
    y0: float
    res: float

    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return int((x - self.x0) // self.res), int((y - self.y0) // self.res)

    def to_world(self, i: int, j: int) -> Tuple[float, float]:
        return self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res

    def free(self, i: int, j: int) -> bool:
        return 0 <= i < self.blocked.shape[0] and 0 <= j < self.blocked.shape[1] and not self.blocked[i, j]

    def free_xy(self, x: float, y: float) -> bool:
        return self.free(*self.to_cell(x, y))
