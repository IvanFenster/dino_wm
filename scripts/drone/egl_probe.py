#!/usr/bin/env python
"""[ARPL-WM] Find which EGL device can render on this node (run inside a GPU job).

On multi-GPU nodes the NVIDIA EGL driver lists every GPU of the node, but Slurm only lets the
job touch the GPU it was given. Genesis (pyrender) tries EGL devices in order, and opening a
device the job may not use can crash the process (segmentation fault) instead of raising.
This script tries each EGL device in a separate child process, so a crash only kills that
child, and prints which EGL_DEVICE_ID works. Then export it before running Genesis:
    export EGL_DEVICE_ID=<working id>
New for this fork.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

CHILD = r"""
import os, sys
sys.path.insert(0, %r)
from drone_sim.egl import setup_gpu_rendering
setup_gpu_rendering(verbose=False)
import genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
from genesis.ext.pyrender.platforms import egl
dev = os.environ.get("EGL_DEVICE_ID")
p = egl.EGLPlatform(64, 64, None if dev is None else int(dev))
p.init_context(); p.make_current()
from OpenGL import GL
print("OK renderer:", GL.glGetString(GL.GL_RENDERER).decode(), "| EGL devices listed:", len(egl.query_devices()))
""" % REPO


def run(dev):
    env = dict(os.environ)
    env.pop("EGL_DEVICE_ID", None)
    if dev is not None:
        env["EGL_DEVICE_ID"] = str(dev)
    r = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=300)
    out = (r.stdout.strip().splitlines() or [""])[-1]
    if r.returncode == 0 and out.startswith("OK"):
        return f"works   {out}"
    if r.returncode < 0:
        return f"CRASH   signal {-r.returncode} (segfault is 11)"
    lines = (r.stdout + "\n" + r.stderr).strip().splitlines() or ["?"]
    err = [l for l in lines if "Error" in l or "error" in l] or lines
    return "fails   " + err[-1].strip()[:150]


print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"), "| SLURM_JOB_GPUS =", os.environ.get("SLURM_JOB_GPUS"))
subprocess.run("nvidia-smi -L", shell=True)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
print(f"{'EGL_DEVICE_ID':>14}  result")
print(f"{'(unset)':>14}  {run(None)}")
for d in range(n):
    res = run(d)
    print(f"{d:>14}  {res}")
    if "Invalid device ID" in res:
        break
