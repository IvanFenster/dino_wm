"""[ARPL-WM] DroneFPVEnv: one race2 quadrotor with a forward-facing RGB-D camera in a Genesis room.

Ported from the Colab "Drone environment" notebook (cell 2). The control loop in `step` is
unchanged: same SE(3) controller, motor lag, heading-frame position-offset action and timing.
Changes for dataset collection:
  - the arena comes from `drone_sim.room.RoomConfig` (configurable size, pillars, extra boxes),
  - collisions are detected (geometric check against the room's boxes, plus Genesis contacts
    when this Genesis version exposes them) instead of only floor hits and flips,
  - frames are only kept when `keep_frames=True` (long collection runs would fill memory).

Credits: Genesis provides physics, rendering, the drone model and the camera. JoonHo Lee's
transport-world (private) provides SE3Controller, Race2Config, SE3Config and the timing
conventions. The room, collision check and data interface are new.

Import order matters: call `drone_sim.egl.setup_gpu_rendering()` before this module is imported,
otherwise Genesis may silently render on the CPU.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional

import numpy as np
import torch

# transport-world is not pip-installable; point TRANSPORT_WORLD at its clone.
_TW = os.environ.get("TRANSPORT_WORLD", os.path.expandvars("$WM/src/transport-world"))
if os.path.isdir(_TW) and _TW not in sys.path:
    sys.path.insert(0, _TW)

import genesis as gs  # noqa: E402
from payload_mappo.config import Race2Config, SE3Config  # noqa: E402
from payload_mappo.controller import SE3Controller, quat_to_matrix  # noqa: E402

from .room import Room, RoomConfig  # noqa: E402

# Camera pose in the drone's frame. Genesis cameras look along their own -Z axis with +Y up,
# so this matrix points the camera along the drone's +X (forward). Same as Colab.
_CAM_ROTATION = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def init_genesis(logging_level: str = "warning") -> None:
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, precision="32",
                logging_level=logging_level)


class DroneFPVEnv:
    """Action: [forward, left, up, yaw_rate] in [-1, 1], heading frame (see Colab notes).
    Observation: rgb uint8 (S, S, 3), depth float32 (S, S) meters, state float32 (13,)
    = position, quaternion wxyz, linear velocity, angular velocity (world frame)."""

    def __init__(
        self,
        room: RoomConfig | Room | None = None,
        image_size: int = 224,
        fov: float = 90.0,
        dt: float = 0.01,            # physics and controller step: 100 Hz
        substeps: int = 10,          # physics steps per action: 10 Hz actions
        action_scale: float = 1.0,
        yaw_rate_scale: float = 1.0,
        rpm_tau: float = 0.43,       # motor lag, same value as transport-world
        max_depth: float = 20.0,
        collision_radius: float = 0.2,   # drone counted as touching a box closer than this, m
        keep_frames: bool = False,
    ):
        init_genesis()
        self.room = room if isinstance(room, Room) else Room(room or RoomConfig())
        self.image_size, self.dt, self.substeps = image_size, dt, substeps
        self.action_scale, self.yaw_rate_scale, self.rpm_tau = action_scale, yaw_rate_scale, rpm_tau
        self.max_depth, self.collision_radius, self.keep_frames = max_depth, collision_radius, keep_frames
        self.device = gs.device
        self.race2 = Race2Config()

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=1),
            rigid_options=gs.options.RigidOptions(dt=dt, constraint_solver=gs.constraint_solver.Newton,
                                                  enable_collision=True),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.obstacles = [
            self.scene.add_entity(gs.morphs.Box(pos=b.center, size=b.size, fixed=True),
                                  surface=gs.surfaces.Default(color=b.color))
            for b in self.room.boxes
        ]
        z0 = 1.5
        self.drone = self.scene.add_entity(gs.morphs.Drone(
            file=self.race2.urdf, model=self.race2.model, propellers_spin=self.race2.propellers_spin,
            convexify=True, decimate=False, pos=(0.0, 0.0, z0)))
        self.camera = self.scene.add_camera(res=(image_size, image_size), pos=(0.0, 0.0, z0),
                                            lookat=(1.0, 0.0, z0), fov=fov, GUI=False)
        self.scene.build()

        offset = np.eye(4)
        offset[:3, :3] = _CAM_ROTATION
        offset[:3, 3] = (0.12, 0.0, 0.03)   # just ahead of the body so it is not in view
        self.camera.attach(self.drone.base_link, offset)

        self.controller = SE3Controller(SE3Config(dt=dt), self.race2, device=self.device)
        self.n_rotors = len(self.race2.propellers_spin)
        self.applied_rpm = torch.full((self.n_rotors,), self.race2.hover_cmd, device=self.device)
        self.yaw_target = torch.zeros(1, device=self.device)
        self.frames: list = []
        self._contacts_ok = hasattr(self.drone, "get_contacts")

    # ------------------------------------------------------------------ state
    def _root_state(self) -> torch.Tensor:
        return torch.cat((self.drone.get_pos(), self.drone.get_quat(), self.drone.get_vel(), self.drone.get_ang()),
                         dim=-1).reshape(1, 13).to(torch.float32)

    @staticmethod
    def _yaw(rotation: torch.Tensor) -> torch.Tensor:
        return torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])

    def render(self):
        if hasattr(self.camera, "move_to_attach"):
            self.camera.move_to_attach()
        rgb, depth, _, _ = self.camera.render(rgb=True, depth=True)
        rgb = np.asarray(rgb.cpu() if torch.is_tensor(rgb) else rgb).astype(np.uint8)
        depth = np.asarray(depth.cpu() if torch.is_tensor(depth) else depth).astype(np.float32)
        return rgb, np.clip(depth, 0.0, self.max_depth)

    def _observe(self) -> dict:
        rgb, depth = self.render()
        if self.keep_frames:
            self.frames.append(rgb)
        return {"rgb": rgb, "depth": depth, "state": self._root_state()[0].cpu().numpy()}

    def _genesis_contact(self) -> bool:
        if not self._contacts_ok:
            return False
        try:
            c = self.drone.get_contacts()
            for key in ("position", "geom_a", "link_a"):
                if key in c:
                    return len(c[key]) > 0 and int(np.prod(np.shape(c[key]))) > 0
        except Exception:
            self._contacts_ok = False   # API differs in this Genesis version; keep the geometric check only
        return False

    # -------------------------------------------------------------------- api
    def reset(self, position=None, yaw: float = 0.0) -> dict:
        position = (0.0, 0.0, 1.5) if position is None else tuple(float(p) for p in position)
        quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        self.drone.set_pos(torch.tensor(position, device=self.device, dtype=torch.float32), zero_velocity=True)
        self.drone.set_quat(torch.tensor(quat, device=self.device, dtype=torch.float32), zero_velocity=True)
        self.drone.zero_all_dofs_velocity()
        self.controller.reset_integrals()
        self.applied_rpm[:] = self.race2.hover_cmd
        self.yaw_target[:] = yaw
        self.frames = []
        return self._observe()

    def step(self, action) -> tuple:
        # Control loop unchanged from Colab.
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(4).clamp(-1.0, 1.0)
        offset_body = action[:3] * self.action_scale
        contact = False
        for _ in range(self.substeps):
            state = self._root_state()
            rotation = quat_to_matrix(state[:, 3:7])
            yaw = self._yaw(rotation)
            cos, sin = torch.cos(yaw), torch.sin(yaw)
            offset_world = torch.stack(
                (cos * offset_body[0] - sin * offset_body[1],
                 sin * offset_body[0] + cos * offset_body[1],
                 offset_body[2].expand_as(cos)), dim=-1)
            self.yaw_target += action[3] * self.yaw_rate_scale * self.dt
            target_rpm = self.controller.compute(state, state[:, :3] + offset_world,
                                                 desired_yaw=self.yaw_target)[0].clamp(0.0, self.race2.max_rpm)
            self.applied_rpm = self.applied_rpm + self.rpm_tau * (target_rpm - self.applied_rpm)
            self.drone.set_propellers_rpm(self.applied_rpm)
            self.scene.step()
            contact = contact or self._genesis_contact()

        obs = self._observe()
        state = self._root_state()
        up_z = quat_to_matrix(state[:, 3:7])[0, 2, 2].item()
        pos = state[0, :3].cpu().numpy()
        clearance = self.room.point_clearance(pos)
        collision = bool(clearance < self.collision_radius) or contact
        flipped = up_z < 0.2
        out_of_room = not self.room.inside(pos)
        crashed = pos[2] < 0.15 or flipped or out_of_room
        return obs, {"crashed": bool(crashed), "collision": collision, "genesis_contact": bool(contact),
                     "clearance": float(clearance), "rpm": self.applied_rpm.cpu().numpy()}

    def save_video(self, path: str = "flight.mp4", frames=None, fps: Optional[int] = None) -> str:
        import imageio.v2 as imageio
        fps = fps or round(1.0 / (self.dt * self.substeps))
        imageio.mimsave(path, frames if frames is not None else self.frames, fps=fps)
        return path
