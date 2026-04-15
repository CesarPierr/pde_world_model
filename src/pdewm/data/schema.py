from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(slots=True)
class TransitionMetadata:
    time_index: int
    dt: float
    pde_id: str
    pde_params: dict[str, float]
    bc_descriptor: dict[str, Any]
    forcing_descriptor: dict[str, Any]
    grid_descriptor: dict[str, Any]
    trajectory_id: str
    split: str
    sample_origin: str
    solver_status: str
    solver_runtime_sec: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


@dataclass(slots=True)
class TransitionRecord:
    state: np.ndarray
    next_state: np.ndarray
    metadata: TransitionMetadata


@dataclass(slots=True)
class DatasetManifest:
    dataset_name: str
    dataset_version: str
    generator_git_hash: str
    solver_version: str
    parameter_space_signature: dict[str, Any]
    seed_policy: dict[str, Any]
    samples: int
    trajectories: int
    split_counts: dict[str, int]
    created_at: str
    sample_shape: tuple[int, ...]
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))

