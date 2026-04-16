from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from pdewm.baselines.common import rollout_state_model
from pdewm.data.datasets import TrajectorySample
from pdewm.models.dynamics.transition_1d import rollout_latent_dynamics
from pdewm.models.representations.autoencoder_1d import Autoencoder1D


@dataclass(slots=True)
class MetricDistribution:
    mean: float
    std: float
    min: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "p10": self.p10,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.max,
        }


@dataclass(slots=True)
class TrajectoryMetricSummary:
    trajectory_count: int
    transition_count: int
    one_step_rmse: MetricDistribution
    one_step_nrmse: MetricDistribution
    rollout_rmse: MetricDistribution
    rollout_nrmse: MetricDistribution

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_count": self.trajectory_count,
            "transition_count": self.transition_count,
            "one_step_rmse": self.one_step_rmse.to_dict(),
            "one_step_nrmse": self.one_step_nrmse.to_dict(),
            "rollout_rmse": self.rollout_rmse.to_dict(),
            "rollout_nrmse": self.rollout_nrmse.to_dict(),
        }


def evaluate_state_model_trajectories(
    model: nn.Module,
    trajectories: list[TrajectorySample],
    *,
    device: torch.device,
) -> TrajectoryMetricSummary:
    model.eval()
    one_step_rmse_values: list[float] = []
    one_step_nrmse_values: list[float] = []
    rollout_rmse_values: list[float] = []
    rollout_nrmse_values: list[float] = []
    transition_count = 0

    with torch.no_grad():
        for sample in trajectories:
            states = torch.from_numpy(sample.states).to(device)
            current_states = states[:-1]
            target_states = states[1:]
            one_step_predictions = model(current_states)
            rollout_predictions = rollout_state_model(
                model,
                current_states[:1],
                horizon=target_states.shape[0],
            )[0]
            one_step_rmse, one_step_nrmse = _aggregate_error(one_step_predictions, target_states)
            rollout_rmse, rollout_nrmse = _aggregate_error(rollout_predictions, target_states)
            one_step_rmse_values.append(one_step_rmse)
            one_step_nrmse_values.append(one_step_nrmse)
            rollout_rmse_values.append(rollout_rmse)
            rollout_nrmse_values.append(rollout_nrmse)
            transition_count += int(target_states.shape[0])

    return TrajectoryMetricSummary(
        trajectory_count=len(trajectories),
        transition_count=transition_count,
        one_step_rmse=_summarize_distribution(one_step_rmse_values),
        one_step_nrmse=_summarize_distribution(one_step_nrmse_values),
        rollout_rmse=_summarize_distribution(rollout_rmse_values),
        rollout_nrmse=_summarize_distribution(rollout_nrmse_values),
    )


def evaluate_world_model_trajectories(
    transition_model: nn.Module,
    autoencoder: Autoencoder1D,
    trajectories: list[TrajectorySample],
    *,
    device: torch.device,
) -> TrajectoryMetricSummary:
    transition_model.eval()
    autoencoder.eval()
    one_step_rmse_values: list[float] = []
    one_step_nrmse_values: list[float] = []
    rollout_rmse_values: list[float] = []
    rollout_nrmse_values: list[float] = []
    transition_count = 0

    with torch.no_grad():
        for sample in trajectories:
            states = torch.from_numpy(sample.states).to(device)
            pde_ids = torch.from_numpy(sample.pde_ids).to(device)
            continuous_context = torch.from_numpy(sample.continuous_context).to(device)
            current_states = states[:-1]
            target_states = states[1:]
            one_step_latents = autoencoder.encode(current_states)
            one_step_predictions = autoencoder.decode(
                transition_model(
                    one_step_latents,
                    pde_ids,
                    continuous_context,
                )
            )
            latent0 = autoencoder.encode(current_states[:1])
            rollout_latents = rollout_latent_dynamics(
                transition_model,
                latent0,
                pde_ids.unsqueeze(0),
                continuous_context.unsqueeze(0),
            )[0]
            rollout_predictions = autoencoder.decode(rollout_latents)
            one_step_rmse, one_step_nrmse = _aggregate_error(one_step_predictions, target_states)
            rollout_rmse, rollout_nrmse = _aggregate_error(rollout_predictions, target_states)
            one_step_rmse_values.append(one_step_rmse)
            one_step_nrmse_values.append(one_step_nrmse)
            rollout_rmse_values.append(rollout_rmse)
            rollout_nrmse_values.append(rollout_nrmse)
            transition_count += int(target_states.shape[0])

    return TrajectoryMetricSummary(
        trajectory_count=len(trajectories),
        transition_count=transition_count,
        one_step_rmse=_summarize_distribution(one_step_rmse_values),
        one_step_nrmse=_summarize_distribution(one_step_nrmse_values),
        rollout_rmse=_summarize_distribution(rollout_rmse_values),
        rollout_nrmse=_summarize_distribution(rollout_nrmse_values),
    )


def _aggregate_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    flat_predictions = predictions.reshape(predictions.shape[0], -1)
    flat_targets = targets.reshape(targets.shape[0], -1)
    rmse_per_transition = torch.sqrt(torch.mean((flat_predictions - flat_targets) ** 2, dim=1))
    target_l2 = torch.linalg.vector_norm(flat_targets, ord=2, dim=1)
    nrmse_per_transition = rmse_per_transition / torch.clamp(target_l2, min=1.0e-6)
    return (
        float(rmse_per_transition.mean().cpu()),
        float(nrmse_per_transition.mean().cpu()),
    )


def _summarize_distribution(values: list[float]) -> MetricDistribution:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric distribution.")
    return MetricDistribution(
        mean=float(array.mean()),
        std=float(array.std()),
        min=float(array.min()),
        p10=float(np.quantile(array, 0.10)),
        p25=float(np.quantile(array, 0.25)),
        p50=float(np.quantile(array, 0.50)),
        p75=float(np.quantile(array, 0.75)),
        p90=float(np.quantile(array, 0.90)),
        p95=float(np.quantile(array, 0.95)),
        p99=float(np.quantile(array, 0.99)),
        max=float(array.max()),
    )
