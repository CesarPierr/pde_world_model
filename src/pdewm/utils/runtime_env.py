from __future__ import annotations

import glob
import os
import sys
import sysconfig
from pathlib import Path


def prepare_runtime_environment() -> None:
    """Ensure CUDA wheel libraries are visible before importing torch.

    Some recent PyTorch/CUDA wheel combinations rely on NVRTC libraries shipped in
    site-packages (for example ``nvidia/cu13/lib``), but those directories may not
    be present in ``LD_LIBRARY_PATH`` when scripts are launched from a fresh shell.
    If we detect such wheel libraries on Linux, prepend them for the current process.
    """
    if sys.platform != "linux":
        return
    if "torch" in sys.modules:
        return

    lib_dirs = _discover_nvidia_lib_dirs()
    if not lib_dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_entries = [entry for entry in current.split(":") if entry]
    missing = [entry for entry in lib_dirs if entry not in current_entries]
    if not missing:
        return

    new_entries = missing + current_entries
    os.environ["LD_LIBRARY_PATH"] = ":".join(new_entries)


def _discover_nvidia_lib_dirs() -> list[str]:
    purelib = Path(sysconfig.get_paths()["purelib"])
    candidates: list[Path] = []

    candidates.extend(sorted(purelib.glob("nvidia/cu*/lib"), reverse=True))
    candidates.extend(
        path
        for path in [
            purelib / "nvidia" / "cuda_nvrtc" / "lib",
            purelib / "nvidia" / "cuda_runtime" / "lib",
            purelib / "torch" / "lib",
        ]
        if path.is_dir()
    )

    valid: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        path_str = str(path)
        if path_str in seen:
            continue
        has_cuda_libs = bool(glob.glob(str(path / "libnvrtc*.so*"))) or bool(glob.glob(str(path / "libcudart.so*")))
        if has_cuda_libs:
            valid.append(path_str)
            seen.add(path_str)
    return valid
