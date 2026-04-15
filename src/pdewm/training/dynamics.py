from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from pdewm.data.context_features import DEFAULT_CONTEXT_FEATURES
from pdewm.data.datasets import TransitionWindowDataset
from pdewm.models.dynamics.transition_1d import (
    LatentTransitionModel1D,
    TransitionModel1DConfig,
    rollout_latent_dynamics,
)
from pdewm.models.representations.autoencoder_1d import Autoencoder1D, Autoencoder1DConfig


@dataclass(slots=True)
class DynamicsEpochMetrics:
    loss: float
    latent_1step: float
    phys_1step: float
    rollout: float
    consistency: float

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "latent_1step": self.latent_1step,
            "phys_1step": self.phys_1step,
            "rollout": self.rollout,
            "consistency": self.consistency,
        }


def train_latent_dynamics(cfg: DictConfig) -> dict[str, Any]:
    device = torch.device(str(cfg.train.device))
    autoencoder = _load_frozen_autoencoder(cfg.train.ae_checkpoint, device)
    context_features = tuple(cfg.train.context_features) if cfg.train.context_features else DEFAULT_CONTEXT_FEATURES

    train_dataset = TransitionWindowDataset(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.train_splits),
        rollout_horizon=int(cfg.train.rollout_horizon),
        context_feature_keys=context_features,
    )
    val_dataset = TransitionWindowDataset(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.val_splits),
        rollout_horizon=int(cfg.train.rollout_horizon),
        context_feature_keys=context_features,
    )

    transition_model = LatentTransitionModel1D(
        TransitionModel1DConfig(
            latent_channels=int(autoencoder.config.latent_channels),
            num_pdes=int(len(train_dataset.pde_to_index)),
            continuous_context_dim=len(context_features),
            hidden_channels=int(cfg.model.hidden_channels),
            context_hidden_dim=int(cfg.model.context_hidden_dim),
            context_output_dim=int(cfg.model.context_output_dim),
            pde_embedding_dim=int(cfg.model.pde_embedding_dim),
            num_blocks=int(cfg.model.num_blocks),
            kernel_size=int(cfg.model.kernel_size),
        )
    ).to(device)
    optimizer = torch.optim.Adam(
        transition_model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
    )

    output_dir = Path(str(cfg.train.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(cfg.train.epochs) + 1):
        train_metrics = _run_dynamics_epoch(
            transition_model,
            autoencoder,
            train_loader,
            optimizer,
            cfg.train.loss_weights,
            device,
            training=True,
        )
        val_metrics = _run_dynamics_epoch(
            transition_model,
            autoencoder,
            val_loader,
            optimizer,
            cfg.train.loss_weights,
            device,
            training=False,
        )
        history.append({"epoch": epoch, "train": train_metrics.to_dict(), "val": val_metrics.to_dict()})
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": transition_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics.to_dict(),
            "val_metrics": val_metrics.to_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "pde_to_index": train_dataset.pde_to_index,
            "context_features": list(context_features),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics.loss < best_val:
            best_val = val_metrics.loss
            torch.save(checkpoint, output_dir / "best.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"history": history, "best_val_loss": best_val}


def _load_frozen_autoencoder(checkpoint_path: str, device: torch.device) -> Autoencoder1D:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["config"]["model"]
    autoencoder = Autoencoder1D(
        Autoencoder1DConfig(
            in_channels=int(model_cfg["in_channels"]),
            base_channels=int(model_cfg["base_channels"]),
            channel_multipliers=tuple(int(item) for item in model_cfg["channel_multipliers"]),
            latent_channels=int(model_cfg["latent_channels"]),
            kernel_size=int(model_cfg["kernel_size"]),
        )
    ).to(device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    return autoencoder


def _run_dynamics_epoch(
    transition_model: nn.Module,
    autoencoder: Autoencoder1D,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_weights: DictConfig,
    device: torch.device,
    *,
    training: bool,
) -> DynamicsEpochMetrics:
    transition_model.train(training)
    accumulators = {
        "loss": 0.0,
        "latent_1step": 0.0,
        "phys_1step": 0.0,
        "rollout": 0.0,
        "consistency": 0.0,
    }
    num_batches = 0

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch in dataloader:
            state = batch["state"].to(device)
            future_states = batch["future_states"].to(device)
            pde_ids = batch["pde_ids"].to(device)
            continuous_context = batch["continuous_context"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                latent0 = autoencoder.encode(state)
                target_latents = torch.stack(
                    [autoencoder.encode(future_states[:, step]) for step in range(future_states.shape[1])],
                    dim=1,
                )

            predicted_latents = rollout_latent_dynamics(
                transition_model,
                latent0,
                pde_ids,
                continuous_context,
            )
            decoded_predictions = torch.stack(
                [autoencoder.decode(predicted_latents[:, step]) for step in range(predicted_latents.shape[1])],
                dim=1,
            )
            recoded_latents = torch.stack(
                [autoencoder.encode(decoded_predictions[:, step]) for step in range(decoded_predictions.shape[1])],
                dim=1,
            )

            latent_1step = F.mse_loss(predicted_latents[:, 0], target_latents[:, 0])
            phys_1step = F.mse_loss(decoded_predictions[:, 0], future_states[:, 0])
            rollout_loss = F.mse_loss(decoded_predictions, future_states)
            consistency_loss = F.mse_loss(recoded_latents, predicted_latents)
            total_loss = (
                float(loss_weights.latent_1step) * latent_1step
                + float(loss_weights.phys_1step) * phys_1step
                + float(loss_weights.rollout) * rollout_loss
                + float(loss_weights.consistency) * consistency_loss
            )

            if training:
                total_loss.backward()
                optimizer.step()

            accumulators["loss"] += float(total_loss.detach().cpu())
            accumulators["latent_1step"] += float(latent_1step.detach().cpu())
            accumulators["phys_1step"] += float(phys_1step.detach().cpu())
            accumulators["rollout"] += float(rollout_loss.detach().cpu())
            accumulators["consistency"] += float(consistency_loss.detach().cpu())
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Received an empty dataloader for dynamics training.")

    return DynamicsEpochMetrics(
        loss=accumulators["loss"] / num_batches,
        latent_1step=accumulators["latent_1step"] / num_batches,
        phys_1step=accumulators["phys_1step"] / num_batches,
        rollout=accumulators["rollout"] / num_batches,
        consistency=accumulators["consistency"] / num_batches,
    )
