from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pdewm.baselines.common import rollout_state_model
from pdewm.data.datasets import TrajectorySample
from pdewm.models.dynamics.transition_1d import rollout_latent_dynamics
from pdewm.models.representations.autoencoder_1d import Autoencoder1D


@dataclass(slots=True)
class RolloutFigurePayload:
    figure: object
    trajectory_id: str
    split: str
    rank_label: str
    rollout_rmse: float


def build_state_model_rollout_figures(
    model: nn.Module,
    trajectories: list[TrajectorySample],
    *,
    device: torch.device,
    split_name: str,
) -> dict[str, RolloutFigurePayload]:
    if not trajectories:
        return {}

    model.eval()
    scored_rollouts: list[tuple[float, TrajectorySample, np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for sample in trajectories:
            states = torch.from_numpy(sample.states).to(device)
            target_states = states[1:]
            rollout_predictions = rollout_state_model(
                model,
                states[:1],
                horizon=target_states.shape[0],
            )[0]
            scored_rollouts.append(
                (
                    _rollout_rmse(rollout_predictions, target_states),
                    sample,
                    _to_time_grid_numpy(target_states),
                    _to_time_grid_numpy(rollout_predictions),
                )
            )

    return _select_rollout_figures(scored_rollouts, split_name=split_name)


def build_world_model_rollout_figures(
    transition_model: nn.Module,
    autoencoder: Autoencoder1D,
    trajectories: list[TrajectorySample],
    *,
    device: torch.device,
    split_name: str,
) -> dict[str, RolloutFigurePayload]:
    if not trajectories:
        return {}

    transition_model.eval()
    autoencoder.eval()
    scored_rollouts: list[tuple[float, TrajectorySample, np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for sample in trajectories:
            states = torch.from_numpy(sample.states).to(device)
            pde_ids = torch.from_numpy(sample.pde_ids).to(device)
            continuous_context = torch.from_numpy(sample.continuous_context).to(device)
            target_states = states[1:]
            latent0 = autoencoder.encode(states[:1])
            rollout_latents = rollout_latent_dynamics(
                transition_model,
                latent0,
                pde_ids.unsqueeze(0),
                continuous_context.unsqueeze(0),
            )[0]
            rollout_predictions = autoencoder.decode(rollout_latents)
            scored_rollouts.append(
                (
                    _rollout_rmse(rollout_predictions, target_states),
                    sample,
                    _to_time_grid_numpy(target_states),
                    _to_time_grid_numpy(rollout_predictions),
                )
            )

    return _select_rollout_figures(scored_rollouts, split_name=split_name)


def _select_rollout_figures(
    scored_rollouts: list[tuple[float, TrajectorySample, np.ndarray, np.ndarray]],
    *,
    split_name: str,
) -> dict[str, RolloutFigurePayload]:
    ordered = sorted(scored_rollouts, key=lambda item: item[0])
    indices = {
        "best": 0,
        "median": len(ordered) // 2,
        "worst": len(ordered) - 1,
    }
    payloads: dict[str, RolloutFigurePayload] = {}
    for rank_label, index in indices.items():
        rollout_rmse, sample, target, prediction = ordered[index]
        payloads[rank_label] = RolloutFigurePayload(
            figure=_render_rollout_figure(
                target=target,
                prediction=prediction,
                split_name=split_name,
                trajectory_id=sample.trajectory_id,
                rank_label=rank_label,
            ),
            trajectory_id=sample.trajectory_id,
            split=split_name,
            rank_label=rank_label,
            rollout_rmse=rollout_rmse,
        )
    return payloads


def _to_time_grid_numpy(states: torch.Tensor) -> np.ndarray:
    array = states.detach().cpu().numpy().astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected rollout states with shape [time, channels, grid], got {array.shape}")
    return array[:, 0, :]


def _rollout_rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((prediction - target) ** 2)).detach().cpu())


def _render_rollout_figure(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    split_name: str,
    trajectory_id: str,
    rank_label: str,
) -> object:
    import matplotlib.pyplot as plt

    error = prediction - target
    # Keep a consistent, data-driven scale: target defines the reference range.
    # This prevents divergent predictions from collapsing the visible contrast.
    signal_limit = float(max(np.max(np.abs(target)), 1.0e-6))
    error_limit = float(max(np.max(np.abs(error)), 1.0e-6))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    panels = [
        (target.T, "Target", "viridis", -signal_limit, signal_limit),
        (prediction.T, "Prediction", "viridis", -signal_limit, signal_limit),
        (error.T, "Error", "coolwarm", -error_limit, error_limit),
    ]

    for axis, (image, title, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        handle = axis.imshow(
            image,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Grid index")
        fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Rollout {rank_label} | split={split_name} | traj={trajectory_id}"
    )
    return fig
