from __future__ import annotations

from collections.abc import Iterable

import numpy as np


DEFAULT_CONTEXT_FEATURES: tuple[str, ...] = (
    "dt",
    "viscosity",
    "ic_amplitude",
    "ic_bandwidth",
    "forcing_amplitude",
    "forcing_mode",
    "domain_length",
    "grid_size",
)


def build_pde_index(pde_ids: Iterable[str]) -> dict[str, int]:
    return {pde_id: index for index, pde_id in enumerate(sorted(set(pde_ids)))}


def metadata_to_context_vector(
    metadata: dict[str, object],
    feature_keys: tuple[str, ...] = DEFAULT_CONTEXT_FEATURES,
) -> np.ndarray:
    pde_params = metadata.get("pde_params", {}) or {}
    forcing_descriptor = metadata.get("forcing_descriptor", {}) or {}
    grid_descriptor = metadata.get("grid_descriptor", {}) or {}

    values: list[float] = []
    for key in feature_keys:
        if key == "dt":
            values.append(float(metadata.get("dt", 0.0)))
        elif key == "forcing_amplitude":
            values.append(float(forcing_descriptor.get("amplitude", 0.0)))
        elif key == "forcing_mode":
            values.append(float(forcing_descriptor.get("mode", 0.0)))
        elif key == "grid_size":
            values.append(float(grid_descriptor.get("grid_size", 0.0)))
        elif key == "domain_length":
            values.append(float(grid_descriptor.get("domain_length", pde_params.get("domain_length", 0.0))))
        else:
            values.append(float(pde_params.get(key, 0.0)))
    return np.asarray(values, dtype=np.float32)
