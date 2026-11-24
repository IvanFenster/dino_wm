# ARPL notes for this fork

This fork of [DINO-WM](https://github.com/gaoyuezhou/dino_wm) (Zhou, Pan, LeCun, Pinto, arXiv 2411.04983) is used at the Agile Robotics and Perception Lab (ARPL), UC Berkeley, to reproduce the paper and extend it toward drones: depth (DeFM), first-person views, and several drones sharing one latent state.

The upstream code is unchanged unless listed below. Every change is either bit-identical or behind an `ARPL_*` switch that is off by default in the Python code, so the original code path stays available and results stay comparable with the paper.

## Branches and tags

- `main`: the working line. Run from here on Savio.
- Short feature branches (`savio-setup`, `depth-defm`, `drone-env`, ...), merged into `main` once they run.
- `upstream-0a9492f`: tag on the last upstream commit, the paper's code.
- Result tags (for example `pointmaze-repro-v1`) mark the code that produced a reported number.

## Savio quick start

Code lives in home (`~/dino_wm`). Data, checkpoints, logs and the conda env live on scratch under `/global/scratch/users/$USER/arpl_wm`, which is **not backed up**. Never train on the login node.

```bash
git clone https://github.com/IvanFenster/dino_wm.git ~/dino_wm && cd ~/dino_wm
# one-time setup, inside a short interactive GPU job
srun -A fc_auto -p savio4_gpu --qos=a5k_gpu4_normal --gres=gpu:A5000:1 -c 4 -t 01:00:00 --pty bash
bash setup/savio_setup.sh
exit
# 10-minute smoke test
sbatch slurm/gpu_test.sbatch
```

Copy datasets into `$WM/data` through the transfer node (`dtn.brc.berkeley.edu`), not the login node. For the deformable dataset (Rope and Granular, with depth), run `bash scripts/get_deformable.sh $WM/data`.

Jobs (submit from the repo root; output goes to `slurm/logs/<job>_<id>.out`):

| Job | Command |
|---|---|
| Train / resume until epoch N | `sbatch slurm/train.sbatch point_maze_2s 10 env=point_maze frameskip=5 num_hist=3 has_decoder=False model.train_decoder=False` |
| Plan, 10 episodes | `sbatch slurm/plan.sbatch point_maze_2s 10 99 3` (model, epoch, seed, MPC round cap) |
| Validation loss of one checkpoint | `sbatch slurm/val_ckpt.sbatch point_maze_2s 3` |
| Depth pilot | `sbatch slurm/depth_pilot.sbatch --n_ep 200 --epochs 5` |

Every job writes a `run_*.txt` (git commit, Slurm job, GPU, versions, exact command) and a `.diff` of uncommitted changes into its run folder. Training also runs `scripts/arpl_logger.py`, which writes `sessions.csv`, `epoch_timing.csv`, GPU use and a copy of stdout to `$WM/logs/<run>/`, and `train.py` appends `epoch_losses.csv` to the run folder.

`slurm/env.sh` turns `ARPL_AMP=1` and `ARPL_SDPA=1` on for jobs. Use `ARPL_AMP=0 ARPL_SDPA=0 sbatch ...` for the exact upstream path.

## Changes against upstream

| Change | Files | Switch | Checked | Origin |
|---|---|---|---|---|
| Pin DINOv2 to commit `85a2460` (Python 3.9) | `models/dino.py` | always | DINO-WM issue #25 | new |
| Planning videos decoded in chunks under `no_grad` | `planning/evaluator.py` | always | only affects saved videos | new |
| Wall / PointMaze loaders read only the needed frames (memory-mapped) | `datasets/traj_dset.py`, `wall_dset.py`, `point_maze_dset.py` | always | identical samples; 11x / 16x faster | new |
| Validation under `torch.no_grad` | `train.py` | always | no backward in validation | new |
| CEM rolls out candidates in chunks | `planning/cem.py` | `ARPL_CEM_CHUNK` (default 50) | scores match to 2e-7, same top-30 | new |
| fp16 mixed precision, training step only | `train.py` | `ARPL_AMP=1` | losses identical to 4 decimals; 2.2x faster | new |
| Fused attention (`scaled_dot_product_attention`) | `models/vit.py` | `ARPL_SDPA=1` | max diff 1.9e-6 fp32, no future-frame leak (`scripts/check_equivalence.py`); 1.7x faster | new |
| Resume keeps the loaded action / proprio encoders | `train.py` | always | upstream bug: they were re-created randomly on resume. Epoch-3 retrain from epoch 2: val 0.0090 with the bug vs 0.0093 with the fix (train loss 0.0145 vs 0.0123), so the damage was gone within one epoch | new |
| `epoch_losses.csv` per run | `train.py` | always | logging only | new |
| Run logger, validation script, DINOv2 weights from Hugging Face, depth pilot, Slurm jobs | `scripts/`, `slurm/`, `setup/` | new files | see each file's header | new |

Known upstream behavior left unchanged: on resume the optimizer state (and the fp16 loss scale) restarts, and `train.py` runs `training.epochs` more epochs rather than up to that number. `slurm/train.sbatch` handles the second point.

fp16 plus fused attention together gave about 3x per epoch in the real PointMaze run (about 3 h down to about 1 h on a T4), and about 3.7x on the training step alone.

## Results so far

**PointMaze reproduction** (`point_maze_2s`, 10 epochs, batch 32, frameskip 5, history 3, no decoder, fp16 + fused attention, Colab T4):

- Validation loss by epoch: 0.0196, 0.0104, 0.0090, 0.0091, 0.0089, 0.0064, 0.0056, 0.0049, 0.0050, 0.0045.
- Planning at epoch 10, 50 episodes (seeds 99 to 103), CEM + MPC from `plan_point_maze.yaml` with MPC capped at 3 rounds: **48/50 = 96%** (paper 98%). Within the first round: 43/50 = 86% (paper's open-loop CEM: 80%).
- Protocol note: the repo's `plan_point_maze.yaml` replans without limit (`max_iter: null`), and CEM checks the current plan in the real simulator after every iteration (`eval_every: 1`), stopping early once an episode succeeds. Success is sticky. Always state the round cap when comparing.

**Depth pilot** (Granular, 200 episodes, camera 1, cached frozen features, H = 1, 5 epochs, seed 0, Colab T4). Relative loss = val MSE / MSE of "next frame = current frame":

| Variant | Token D | Predictor params | Val MSE | Copy MSE | Relative |
|---|---|---|---|---|---|
| RGB | 384 | 20.0M | 1.9421 | 0.3024 | 6.42 |
| Depth | 384 | 20.0M | 0.1167 | 0.0458 | 2.55 |
| RGB+depth concat | 768 | 38.9M | 1.9371 | 0.2261 | 8.57 |

The pipeline works end to end, but training had barely started (every variant is still worse than copying the current frame), so there is no conclusion about depth yet.

## Drone navigation dataset (Genesis)

Planned start-to-goal flights in the Genesis room from the Colab "Drone environment" notebook, recorded as RGB, depth, state and actions for DINO-WM (and later LeWM). Runs in its own conda env (`setup/drone_env.yaml`, Python 3.11), because Genesis needs Python 3.10 or newer and the dino_wm env is 3.9.

| Piece | File |
|---|---|
| Room layout (configurable size, pillars, extra boxes; seed 0 = the Colab arena) | `drone_sim/room.py` |
| Start/goal sampling, A* on the floor plan, shortcutting | `drone_sim/planner.py` |
| Path follower that outputs `[forward, left, up, yaw_rate]` | `drone_sim/follower.py` |
| Genesis env, ported from Colab with the same control loop, plus collision check | `drone_sim/env.py` |
| GPU (EGL) rendering fix for headless nodes | `drone_sim/egl.py` |
| Episode runner, raw `.npz` writer, packer into the DINO-WM layout | `drone_sim/collect.py` |
| Loader (optional depth, proprio subsets, held-out rooms) and config | `datasets/drone_dset.py`, `conf/env/drone_nav.yaml` |
| Notebook for setup and the first trajectories | `notebooks/drone/drone_collect.ipynb` |
| Batch collection, one Slurm array task per room | `scripts/drone/collect.py`, `slurm/drone_collect.sbatch` |

```bash
bash setup/drone_setup.sh                                  # once, inside a short GPU job (needs transport-world access)
sbatch --array=0-0 slurm/drone_collect.sbatch               # 1 room x 10 episodes
EPISODES=200 sbatch --array=0-9 slurm/drone_collect.sbatch  # 10 rooms x 200 episodes
python scripts/drone/collect.py --pack --out $WM/data/drone_raw --dset $WM/data/drone_nav
```

Each episode file holds `rgb (T+1,S,S,3) uint8`, `depth (T+1,S,S) float16 meters`, `state (T+1,13)` (position, quaternion wxyz, linear and angular velocity), `action (T,4)`, per-step collision flags and clearance, the planned path and a JSON `meta` (start, goal, room, all configs, outcome, timing). Every job also writes a `.jsonl` log per room plus the usual `run_*.txt`.

Checked before the first Savio run (cloud CPU machine, no GPU): planner, follower and writer on a kinematic stand-in with the Colab-measured velocity gains (30 of 30 episodes in 3 rooms reached the goal, closest approach 0.44 m), the packer, and the loader (shapes, held-out rooms, partial-frame reads equal full reads). With Genesis at the pinned commit (1.2.1) on CPU, `drone_sim/env.py` built the room, rendered RGB and depth from the attached camera, ran a short episode and packed it, using a stand-in controller because transport-world is private; `get_contacts` exists in that version and reported contacts against a pillar. Not run yet: the real SE(3) controller, GPU rendering on Savio, the notebook and the sbatch file.

## Not verified yet on Savio

`scripts/val_ckpt.py` was verified on Colab: with the resume fix it reproduces epoch 3's 0.0090 (and recomputed the unlogged epoch 2 as 0.0104). Before the fix it gave 0.0749, because reloading replaced the action and proprio encoders with random ones; that is how the resume bug was found.


- `setup/savio_setup.sh` and all `slurm/*.sbatch` files (written for Savio, not run there yet).
- `scripts/depth_pilot.py` now loads the pinned DINOv2 commit instead of the latest; DeFM may need a newer Python than the dino_wm env (3.9).
- fp16 check 3 in `scripts/check_equivalence.py` needs a GPU (`gpu_test.sbatch` runs it).

## Colab notebooks

`notebooks/colab/` keeps the Colab notebooks that produced the results above, with outputs stripped, for reference. They patch the code on Google Drive with `%%writefile` cells; those patches are now normal commits, so the notebooks are not the way to run anything on Savio.

## Credits

- DINO-WM model, training, planning and environments: upstream [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm).
- DINOv2: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2); weights via Hugging Face `facebook/dinov2-small`.
- DeFM depth encoder: [leggedrobotics/defm](https://github.com/leggedrobotics/defm).
- Genesis drone simulation: [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) (physics, rendering, drone model, camera). SE(3) controller, race2 drone config and timing: JoonHo Lee's transport-world (private).
- Everything marked "new" above, and `drone_sim/`: written for this fork at ARPL.
