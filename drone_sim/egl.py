"""[ARPL-WM] Make Genesis render on the NVIDIA GPU in headless jobs (Colab, Savio).

Problem seen in Colab: only Mesa's software EGL vendor file was registered, so Genesis rendered
on the CPU without any error (about 1.6 s per frame instead of milliseconds). The Colab fix wrote
/usr/share/glvnd/egl_vendor.d/10_nvidia.json, which needs root. On Savio we cannot write there,
so we write the same vendor file into our own cache folder and point
__EGL_VENDOR_LIBRARY_FILENAMES at it (libglvnd reads that variable). Must run before Genesis
(or pyrender / PyOpenGL) is first imported in the process.
"""
from __future__ import annotations

import glob
import os

_SEARCH = ["/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64-nvidia", "/usr/lib/nvidia", "/usr/lib"]


def find_nvidia_egl() -> str | None:
    dirs = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d] + _SEARCH
    for d in dirs:
        hits = sorted(glob.glob(os.path.join(d, "libEGL_nvidia.so*")))
        if hits:
            return hits[0]
    return None


def setup_gpu_rendering(cache_dir: str | None = None, verbose: bool = True) -> str | None:
    """Register NVIDIA's EGL for this process. Returns the library path, or None (CPU rendering)."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    lib = find_nvidia_egl()
    if lib is None:
        if verbose:
            print("[egl] libEGL_nvidia not found: rendering will fall back to the CPU (no GPU on this node?)")
        return None
    cache_dir = cache_dir or os.path.expandvars(os.environ.get("ARPL_EGL_DIR", "$HOME/.cache/arpl_egl"))
    os.makedirs(cache_dir, exist_ok=True)
    vendor = os.path.join(cache_dir, "10_nvidia.json")
    with open(vendor, "w") as f:
        f.write('{"file_format_version": "1.0.0", "ICD": {"library_path": "%s"}}\n' % lib)
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = vendor
    if verbose:
        print(f"[egl] using {lib} via {vendor}")
    return lib
