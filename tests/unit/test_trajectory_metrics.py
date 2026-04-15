from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

from pdewm.data.datasets import TrajectorySample, load_trajectory_samples
from pdewm.data.generation import build_manifest
from pdewm.data.schema import TransitionMetadata, TransitionRecord
from pdewm.data.writer import OfflineDatasetWriter
from pdewm.evaluation.trajectory_metrics import (
    evaluate_state_model_trajectories,
    evaluate_world_model_trajectories,
)
from pdewm.solvers.base import BasePDESolver


class _ShiftStateModel(nn.Module):
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return states + 1.0


class _IdentityAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return states

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents


class _ShiftLatentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        latent: torch.Tensor,
        pde_ids: torch.Tensor,
        continuous_context: torch.Tensor,
    ) -> torch.Tensor:
        del pde_ids, continuous_context
        return latent + 1.0


class _DummySolver(BasePDESolver):
    solver_name = "dummy"
    solver_version = "0.0"

    def sample_initial_condition(self, rng, context):  # pragma: no cover - not used
        raise NotImplementedError

    def simulate(self, initial_state, context, num_steps, dt=None, options=None):  # pragma: no cover - not used
        raise NotImplementedError


def test_evaluate_state_model_trajectories_reports_zero_error() -> None:
    sample = _toy_trajectory_sample()
    summary = evaluate_state_model_trajectories(
        _ShiftStateModel(),
        [sample],
        device=torch.device("cpu"),
    )
    assert summary.one_step_rmse.mean == 0.0
    assert summary.one_step_nrmse.mean == 0.0
    assert summary.rollout_rmse.mean == 0.0
    assert summary.rollout_nrmse.mean == 0.0


def test_evaluate_world_model_trajectories_reports_zero_error() -> None:
    sample = _toy_trajectory_sample()
    summary = evaluate_world_model_trajectories(
        _ShiftLatentModel(),
        _IdentityAutoencoder(),
        [sample],
        device=torch.device("cpu"),
    )
    assert summary.one_step_rmse.mean == 0.0
    assert summary.rollout_nrmse.mean == 0.0


def test_load_trajectory_samples_reconstructs_full_trajectory(tmp_path: Path) -> None:
    dataset_root = _write_toy_dataset(tmp_path)
    trajectories = load_trajectory_samples(dataset_root, splits=("val",))
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory.states.shape == (3, 1, 4)
    assert np.allclose(trajectory.states[0], np.zeros((1, 4), dtype=np.float32))
    assert np.allclose(trajectory.states[-1], np.full((1, 4), 2.0, dtype=np.float32))


def _toy_trajectory_sample() -> TrajectorySample:
    states = np.stack(
        [
            np.zeros((1, 4), dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
            np.full((1, 4), 2.0, dtype=np.float32),
        ],
        axis=0,
    )
    return TrajectorySample(
        trajectory_id="toy",
        split="val",
        states=states,
        pde_ids=np.zeros(2, dtype=np.int64),
        continuous_context=np.zeros((2, 2), dtype=np.float32),
        metadatas=[
            {"time_index": 0, "pde_id": "toy", "dt": 1.0},
            {"time_index": 1, "pde_id": "toy", "dt": 1.0},
        ],
    )


def _write_toy_dataset(tmp_path: Path) -> Path:
    samples = []
    for time_index in range(2):
        samples.append(
            TransitionRecord(
                state=np.full((1, 4), float(time_index), dtype=np.float32),
                next_state=np.full((1, 4), float(time_index + 1), dtype=np.float32),
                metadata=TransitionMetadata(
                    time_index=time_index,
                    dt=1.0,
                    pde_id="toy",
                    pde_params={"viscosity": 0.1},
                    bc_descriptor={},
                    forcing_descriptor={},
                    grid_descriptor={"grid_size": 4, "domain_length": 1.0},
                    trajectory_id="traj_0",
                    split="val",
                    sample_origin="offline",
                    solver_status="ok",
                    solver_runtime_sec=0.1,
                    seed=7,
                ),
            )
        )
    manifest = build_manifest(
        dataset_name="toy",
        dataset_version="v0",
        solver=_DummySolver(),
        solver_cfg=_solver_cfg(),
        data_cfg=_data_cfg(),
        samples=samples,
        trajectory_count=1,
    )
    return OfflineDatasetWriter(tmp_path / "toy" / "v0").write(samples, manifest)


def _solver_cfg():
    return OmegaConf.create(
        {
            "pde_id": "toy",
            "dt": 1.0,
            "parameter_space": {},
        }
    )


def _data_cfg():
    return type(
        "DataCfg",
        (),
        {
            "seed": 7,
            "generation": type("GenerationCfg", (), {"seed_stride": 1})(),
            "num_steps": 2,
        },
    )()
