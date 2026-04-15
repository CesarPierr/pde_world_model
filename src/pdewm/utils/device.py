from __future__ import annotations

import torch


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve a device string to a ``torch.device``.

    When *device_str* is ``"auto"`` the function picks the best available
    accelerator:

    * ``cuda`` if NVIDIA CUDA is available,
    * ``mps`` if Apple Metal Performance Shaders are available (macOS),
    * ``cpu`` otherwise.

    Any other value (e.g. ``"cpu"``, ``"cuda:0"``) is passed through
    directly to ``torch.device``.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
