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

from pdewm.baselines.common import rollout_state_model
from pdewm.baselines.factory import build_baseline_model
from pdewm.baselines.pod_mlp import PODMLPBaseline
from pdewm.data.datasets import TransitionWindowDataset


@dataclass(slots=True)
class BaselineEpochMetrics:
    loss: float
    phys_1step: float
    rollout: float

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "phys_1step": self.phys_1step,
            "rollout": self.rollout,
        }


def train_baseline(cfg: DictConfig) -> dict[str, Any]:
    device = torch.device(str(cfg.train.device))
    train_dataset = TransitionWindowDataset(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.train_splits),
        rollout_horizon=int(cfg.train.rollout_horizon),
    )
    val_dataset = TransitionWindowDataset(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.val_splits),
        rollout_horizon=int(cfg.train.rollout_horizon),
    )
    test_dataset = TransitionWindowDataset(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.test_splits),
        rollout_horizon=int(cfg.train.rollout_horizon),
    )

    sample = train_dataset[0]
    state_size = int(sample["state"].shape[-1])
    model = build_baseline_model(cfg.model, state_size=state_size).to(device)

    if isinstance(model, PODMLPBaseline):
        _fit_pod_basis(model, train_dataset, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
    )

    output_dir = Path(str(cfg.train.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, int(cfg.train.epochs) + 1):
        train_metrics = _run_baseline_epoch(
            model,
            train_loader,
            optimizer,
            cfg.train.loss_weights,
            device,
            training=True,
        )
        val_metrics = _run_baseline_epoch(
            model,
            val_loader,
            optimizer,
            cfg.train.loss_weights,
            device,
            training=False,
        )
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
            best_epoch = epoch
            torch.save(checkpoint, output_dir / "best.pt")

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    val_best_metrics = _run_baseline_epoch(
        model,
        val_loader,
        optimizer,
        cfg.train.loss_weights,
        device,
        training=False,
    )
    test_metrics = _run_baseline_epoch(
        model,
        test_loader,
        optimizer,
        cfg.train.loss_weights,
        device,
        training=False,
    )

    summary = {
        "model_name": str(cfg.model.name),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "val_metrics": val_best_metrics.to_dict(),
        "test_metrics": test_metrics.to_dict(),
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _fit_pod_basis(model: PODMLPBaseline, dataset: TransitionWindowDataset, device: torch.device) -> None:
    collected_states = []
    for index in range(len(dataset)):
        sample = dataset[index]
        collected_states.append(sample["state"])
        future_states = sample["future_states"]
        for step in range(future_states.shape[0]):
            collected_states.append(future_states[step])
    stacked = torch.stack(collected_states, dim=0).to(device)
    model.fit_basis(stacked)


def _run_baseline_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_weights: DictConfig,
    device: torch.device,
    *,
    training: bool,
) -> BaselineEpochMetrics:
    model.train(training)
    accumulators = {"loss": 0.0, "phys_1step": 0.0, "rollout": 0.0}
    num_batches = 0

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch in dataloader:
            state = batch["state"].to(device)
            future_states = batch["future_states"].to(device)
            horizon = future_states.shape[1]

            if training:
                optimizer.zero_grad(set_to_none=True)

            predictions = rollout_state_model(model, state, horizon)
            phys_1step = F.mse_loss(predictions[:, 0], future_states[:, 0])
            rollout_loss = F.mse_loss(predictions, future_states)
            total_loss = (
                float(loss_weights.phys_1step) * phys_1step
                + float(loss_weights.rollout) * rollout_loss
            )

            if training:
                total_loss.backward()
                optimizer.step()

            accumulators["loss"] += float(total_loss.detach().cpu())
            accumulators["phys_1step"] += float(phys_1step.detach().cpu())
            accumulators["rollout"] += float(rollout_loss.detach().cpu())
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Received an empty dataloader for baseline training.")

    return BaselineEpochMetrics(
        loss=accumulators["loss"] / num_batches,
        phys_1step=accumulators["phys_1step"] / num_batches,
        rollout=accumulators["rollout"] / num_batches,
    )
