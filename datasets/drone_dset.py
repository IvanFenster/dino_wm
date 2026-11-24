# [ARPL-WM] Loader for the Genesis drone navigation dataset (drone_sim/collect.py, pack_dataset).
# Modeled on datasets/point_maze_dset.py (same file layout and normalization), with two additions:
#   - obs["depth"] (T, 1, H, W) in meters when return_depth=True, for the depth (DeFM) experiments,
#   - proprio can be a subset of the 13-D state (proprio_mode), and validation can hold out rooms.
# New for this fork.
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange

from .traj_dset import TrajDataset, TrajSlicerDataset, TrajSubset, get_train_val_sliced

# State: pos(0:3) quat_wxyz(3:7) lin_vel(7:10) ang_vel(10:13)
PROPRIO_SLICES = {
    "full": list(range(13)),                 # same choice as PointMaze (proprio = full state)
    "no_position": list(range(3, 13)),       # attitude + velocities only, like an IMU + odometry
    "velocity": list(range(7, 13)),
}


class DroneDataset(TrajDataset):
    supports_partial_frames = True  # used by TrajSlicerDataset's fast path

    def __init__(
        self,
        data_path: str = "data/drone_nav",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
        return_depth: bool = False,
        depth_size: Optional[int] = None,     # resize depth to this (set to img_size); None keeps it
        proprio_mode: str = "full",
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.return_depth = return_depth
        self.depth_size = depth_size
        self.normalize_action = normalize_action
        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "actions.pth").float() / action_scale
        self.seq_lengths = torch.load(self.data_path / "seq_lengths.pth")
        meta_file = self.data_path / "episodes.json"
        self.episodes = json.load(open(meta_file)) if meta_file.exists() else None

        n = n_rollout or len(self.states)
        self.states, self.actions, self.seq_lengths = self.states[:n], self.actions[:n], self.seq_lengths[:n]
        self.proprios = self.states[..., PROPRIO_SLICES[proprio_mode]].clone()
        print(f"Loaded {n} drone rollouts (proprio_mode={proprio_mode}, depth={return_depth})")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]
        if normalize_action:
            self.action_mean, self.action_std = self._mean_std(self.actions)
            self.state_mean, self.state_std = self._mean_std(self.states)
            self.proprio_mean, self.proprio_std = self._mean_std(self.proprios)
        else:
            self.action_mean, self.action_std = torch.zeros(self.action_dim), torch.ones(self.action_dim)
            self.state_mean, self.state_std = torch.zeros(self.state_dim), torch.ones(self.state_dim)
            self.proprio_mean, self.proprio_std = torch.zeros(self.proprio_dim), torch.ones(self.proprio_dim)
        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def _mean_std(self, data):
        rows = torch.vstack([data[i, : self.seq_lengths[i]] for i in range(len(self.seq_lengths))])
        return rows.mean(0), rows.std(0).clamp_min(1e-6)

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return torch.cat([self.actions[i, : self.seq_lengths[i]] for i in range(len(self.seq_lengths))], dim=0)

    @staticmethod
    def _load(path):
        try:
            return torch.load(path, mmap=True)   # read only the requested frames from disk
        except Exception:
            return torch.load(path)

    def get_frames(self, idx, frames):
        image = self._load(self.data_path / "obses" / f"episode_{idx:03d}.pth")[frames]
        image = rearrange(image.float() / 255.0, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": self.proprios[idx, frames]}
        if self.return_depth:
            depth = self._load(self.data_path / "depths" / f"episode_{idx:03d}.pth")[frames].float()
            depth = depth.unsqueeze(1)                                   # T 1 H W, meters
            if self.depth_size and depth.shape[-1] != self.depth_size:
                depth = F.interpolate(depth, size=(self.depth_size, self.depth_size), mode="nearest")
            obs["depth"] = depth
        return obs, self.actions[idx, frames], self.states[idx, frames], {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_drone_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/drone_nav",
    normalize_action=False,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    return_depth=False,
    depth_size=None,
    proprio_mode="full",
    val_rooms: Optional[Sequence[int]] = None,
):
    """val_rooms: room seeds kept out of training (tests generalization to unseen layouts).
    None: random split by episode, as in the other DINO-WM loaders."""
    dset = DroneDataset(data_path=data_path, n_rollout=n_rollout, transform=transform,
                        normalize_action=normalize_action, return_depth=return_depth,
                        depth_size=depth_size, proprio_mode=proprio_mode)
    if val_rooms:
        rooms = [ep["room"]["seed"] for ep in dset.episodes[: len(dset)]]
        val_idx = [i for i, r in enumerate(rooms) if r in set(val_rooms)]
        train_idx = [i for i, r in enumerate(rooms) if r not in set(val_rooms)]
        dset_train, dset_val = TrajSubset(dset, train_idx), TrajSubset(dset, val_idx)
        n = num_hist + num_pred
        train_slices = TrajSlicerDataset(dset_train, n, frameskip)
        val_slices = TrajSlicerDataset(dset_val, n, frameskip)
    else:
        dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
            traj_dataset=dset, train_fraction=split_ratio, num_frames=num_hist + num_pred, frameskip=frameskip)
    return {"train": train_slices, "valid": val_slices}, {"train": dset_train, "valid": dset_val}
