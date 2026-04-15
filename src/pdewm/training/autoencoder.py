from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from pdewm.data.datasets import StateAutoencoderDataset
from pdewm.models.representations.autoencoder_1d import Autoencoder1D, Autoencoder1DConfig
from pdewm.models.representations.losses import AutoencoderLossWeights, compute_autoencoder_losses


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    l1: float
    l2: float
    gradient: float
    spectral: float

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "l1": self.l1,
            "l2": self.l2,
            "gradient": self.gradient,
            "spectral": self.spectral,
        }


def train_autoencoder(cfg: DictConfig) -> dict[str, Any]:
    device = torch.device(str(cfg.train.device))
    model = Autoencoder1D(_build_model_config(cfg.model)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )
    weights = AutoencoderLossWeights(**OmegaConf.to_container(cfg.train.loss_weights, resolve=True))

    train_dataset = StateAutoencoderDataset(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.train_splits),
        include_next_state=bool(cfg.train.include_next_state),
    )
    val_dataset = StateAutoencoderDataset(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.val_splits),
        include_next_state=bool(cfg.train.include_next_state),
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
    history: list[dict[str, Any]] = []
    best_val = float("inf")

    for epoch in range(1, int(cfg.train.epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, weights, device, training=True)
        val_metrics = _run_epoch(model, val_loader, optimizer, weights, device, training=False)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics.to_dict(),
                "val": val_metrics.to_dict(),
            }
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics.to_dict(),
            "val_metrics": val_metrics.to_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics.loss < best_val:
            best_val = val_metrics.loss
            torch.save(checkpoint, output_dir / "best.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"history": history, "best_val_loss": best_val}


def _build_model_config(cfg: DictConfig) -> Autoencoder1DConfig:
    channel_multipliers = tuple(int(value) for value in cfg.channel_multipliers)
    return Autoencoder1DConfig(
        in_channels=int(cfg.in_channels),
        base_channels=int(cfg.base_channels),
        channel_multipliers=channel_multipliers,
        latent_channels=int(cfg.latent_channels),
        kernel_size=int(cfg.kernel_size),
    )


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    weights: AutoencoderLossWeights,
    device: torch.device,
    *,
    training: bool,
) -> EpochMetrics:
    model.train(training)
    accumulators = {"loss": 0.0, "l1": 0.0, "l2": 0.0, "gradient": 0.0, "spectral": 0.0}
    num_batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in dataloader:
            states = batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            reconstructions, _ = model(states)
            total_loss, components = compute_autoencoder_losses(reconstructions, states, weights)

            if training:
                total_loss.backward()
                optimizer.step()

            accumulators["loss"] += float(total_loss.detach().cpu())
            for key, value in components.items():
                accumulators[key] += float(value.detach().cpu())
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Received an empty dataloader.")

    return EpochMetrics(
        loss=accumulators["loss"] / num_batches,
        l1=accumulators["l1"] / num_batches,
        l2=accumulators["l2"] / num_batches,
        gradient=accumulators["gradient"] / num_batches,
        spectral=accumulators["spectral"] / num_batches,
    )
