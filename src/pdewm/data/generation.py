from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from pdewm.data.schema import DatasetManifest, TransitionMetadata, TransitionRecord
from pdewm.solvers.base import BasePDESolver, SimulationResult
from pdewm.solvers.contexts import PDEContext
from pdewm.utils.git import get_git_commit_hash


def sample_context(solver_cfg: DictConfig, split_cfg: DictConfig, rng: np.random.Generator) -> PDEContext:
    parameter_space = OmegaConf.to_container(solver_cfg.parameter_space, resolve=True)
    raw_overrides = split_cfg.get("param_overrides")
    overrides = OmegaConf.to_container(raw_overrides, resolve=True) if raw_overrides is not None else {}
    merged_space = _merge_dicts(parameter_space, overrides)

    if solver_cfg.name == "burgers_1d":
        forcing_amplitude = _sample_from_spec(merged_space["forcing_amplitude_range"], rng)
        return PDEContext(
            pde_id="burgers_1d",
            parameters={
                "viscosity": float(_sample_from_spec(merged_space["viscosity_range"], rng)),
                "ic_amplitude": float(_sample_from_spec(merged_space["ic_amplitude_range"], rng)),
                "ic_bandwidth": float(_sample_from_spec(merged_space["ic_bandwidth_range"], rng)),
            },
            forcing_descriptor={
                "type": "sinusoidal",
                "amplitude": float(forcing_amplitude),
                "mode": int(_sample_from_spec(merged_space["forcing_mode_choices"], rng)),
                "phase": 0.0,
            },
            grid_descriptor={"grid_size": int(solver_cfg.grid_size), "domain_length": float(solver_cfg.domain_length)},
            dt=float(solver_cfg.dt),
            dimension=1,
        )

    if solver_cfg.name == "ks_1d":
        domain_length = float(_sample_from_spec(merged_space["domain_length_range"], rng))
        return PDEContext(
            pde_id="ks_1d",
            parameters={
                "domain_length": domain_length,
                "ic_amplitude": float(_sample_from_spec(merged_space["ic_amplitude_range"], rng)),
                "ic_bandwidth": float(_sample_from_spec(merged_space["ic_bandwidth_range"], rng)),
            },
            grid_descriptor={"grid_size": int(solver_cfg.grid_size), "domain_length": domain_length},
            dt=float(solver_cfg.dt),
            dimension=1,
        )

    raise ValueError(f"Unsupported solver for context sampling: {solver_cfg.name}")


def simulation_to_records(
    result: SimulationResult,
    context: PDEContext,
    split_name: str,
    trajectory_id: str,
    sample_origin: str,
    seed: int,
) -> list[TransitionRecord]:
    trajectory = result.trajectory
    records: list[TransitionRecord] = []

    for time_index in range(trajectory.shape[0] - 1):
        metadata = TransitionMetadata(
            time_index=time_index,
            dt=context.dt,
            pde_id=context.pde_id,
            pde_params=deepcopy(context.parameters),
            bc_descriptor=deepcopy(context.boundary_conditions),
            forcing_descriptor=deepcopy(context.forcing_descriptor),
            grid_descriptor=deepcopy(context.grid_descriptor),
            trajectory_id=trajectory_id,
            split=split_name,
            sample_origin=sample_origin,
            solver_status=result.status,
            solver_runtime_sec=result.runtime_sec,
            seed=seed,
        )
        records.append(
            TransitionRecord(
                state=trajectory[time_index],
                next_state=trajectory[time_index + 1],
                metadata=metadata,
            )
        )
    return records


def build_manifest(
    *,
    dataset_name: str,
    dataset_version: str,
    solver: BasePDESolver,
    solver_cfg: DictConfig,
    data_cfg: DictConfig,
    samples: list[TransitionRecord],
    trajectory_count: int,
) -> DatasetManifest:
    split_counts: dict[str, int] = {}
    for sample in samples:
        split_counts[sample.metadata.split] = split_counts.get(sample.metadata.split, 0) + 1

    parameter_space_signature = OmegaConf.to_container(solver_cfg.parameter_space, resolve=True)
    seed_policy = {
        "project_seed": int(data_cfg.seed),
        "trajectory_seed_stride": int(data_cfg.generation.seed_stride),
    }
    sample_shape = tuple(int(dim) for dim in samples[0].state.shape)
    extras = {
        "pde_id": solver_cfg.pde_id,
        "dt": float(solver_cfg.dt),
        "num_steps": int(data_cfg.num_steps),
    }
    return DatasetManifest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        generator_git_hash=get_git_commit_hash(),
        solver_version=solver.solver_version,
        parameter_space_signature=parameter_space_signature,
        seed_policy=seed_policy,
        samples=len(samples),
        trajectories=trajectory_count,
        split_counts=split_counts,
        created_at=datetime.now(UTC).isoformat(),
        sample_shape=sample_shape,
        extras=extras,
    )


def _merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _sample_from_spec(spec: Any, rng: np.random.Generator) -> float | int:
    if isinstance(spec, list) and len(spec) == 2:
        lower, upper = spec
        if isinstance(lower, int) and isinstance(upper, int):
            return int(rng.integers(lower, upper + 1))
        return float(rng.uniform(float(lower), float(upper)))

    if isinstance(spec, list):
        return spec[int(rng.integers(0, len(spec)))]

    return spec
