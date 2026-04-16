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
from pdewm.data.datasets import TransitionWindowDataset, load_trajectory_samples
from pdewm.evaluation.rollout_figures import build_state_model_rollout_figures
from pdewm.evaluation.trajectory_metrics import evaluate_state_model_trajectories
from pdewm.utils.device import resolve_device
from pdewm.utils.wandb import (
    compose_wandb_group,
    compose_wandb_name,
    flatten_metrics,
    init_wandb_run,
)


@dataclass(slots=True)
class BaselineEpochMetrics:
    loss: float
    loss_std: float
    loss_min: float
    loss_max: float
    phys_1step: float
    rollout: float

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "loss_std": self.loss_std,
            "loss_min": self.loss_min,
            "loss_max": self.loss_max,
            "phys_1step": self.phys_1step,
            "rollout": self.rollout,
        }


def train_baseline(cfg: DictConfig) -> dict[str, Any]:
    device = resolve_device(str(cfg.train.device))
    train_dataset = TransitionWindowDataset(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.train_splits),
        rollout_horizon=int(cfg.train.rollout_horizon),
    )
    eval_dataset = TransitionWindowDataset(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.eval_splits),
        rollout_horizon=int(cfg.train.rollout_horizon),
    )
    eval_trajectories = load_trajectory_samples(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.eval_splits),
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
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
    )

    output_dir = Path(str(cfg.train.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    wandb_run = init_wandb_run(
        cfg,
        default_name=compose_wandb_name(
            "baseline",
            cfg.model.name,
            Path(str(cfg.train.dataset_root)).name,
            output_dir.name,
            f"seed {int(cfg.project.seed)}",
        ),
        default_group=compose_wandb_group("baseline", cfg.model.name, Path(str(cfg.train.dataset_root)).name),
        default_job_type="train_baseline",
        extra_tags=[str(cfg.model.name), "baseline_1d", str(OmegaConf.select(cfg, "project.phase") or "unknown_phase")],
    )

    try:
        for epoch in range(1, int(cfg.train.epochs) + 1):
            train_metrics = _run_baseline_epoch(
                model,
                train_loader,
                optimizer,
                cfg.train.loss_weights,
                device,
                training=True,
            )
            eval_metrics = _run_baseline_epoch(
                model,
                eval_loader,
                optimizer,
                cfg.train.loss_weights,
                device,
                training=False,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics.to_dict(),
                    "eval": eval_metrics.to_dict(),
                }
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics.to_dict(),
                "eval_metrics": eval_metrics.to_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            torch.save(checkpoint, output_dir / "last.pt")

            wandb_run.log(
                {
                    "epoch": epoch,
                    **flatten_metrics("train", train_metrics.to_dict()),
                    **flatten_metrics("eval", eval_metrics.to_dict()),
                },
                step=epoch,
            )
            last_epoch = epoch
            last_train_metrics = train_metrics
            last_eval_metrics = eval_metrics

        eval_final_metrics = _run_baseline_epoch(
            model,
            eval_loader,
            optimizer,
            cfg.train.loss_weights,
            device,
            training=False,
        )
        trajectory_eval_metrics = evaluate_state_model_trajectories(
            model,
            eval_trajectories,
            device=device,
        )

        rollout_eval_figures = build_state_model_rollout_figures(
            model,
            eval_trajectories,
            device=device,
            split_name="eval",
        )

        summary = {
            "model_name": str(cfg.model.name),
            "final_epoch": int(last_epoch),
            "final_eval_loss": float(eval_final_metrics.loss),
            "train_metrics": last_train_metrics.to_dict(),
            "eval_metrics": eval_final_metrics.to_dict(),
            "trajectory_eval_metrics": trajectory_eval_metrics.to_dict(),
            "selected_checkpoint": str(output_dir / "last.pt"),
        }
    finally:
        history_path = output_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        summary = locals().get(
            "summary",
            {
                "model_name": str(cfg.model.name),
                "final_epoch": int(locals().get("last_epoch", 0)),
                "final_eval_loss": float(
                    getattr(locals().get("last_eval_metrics"), "loss", float("inf"))
                ),
                "selected_checkpoint": str(output_dir / "last.pt"),
            },
        )
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for rank_label, payload in locals().get("rollout_eval_figures", {}).items():
            figure_path = output_dir / f"rollout_eval_{rank_label}.png"
            payload.figure.savefig(figure_path, dpi=160, bbox_inches="tight")
            wandb_run.log(
                {
                    f"rollout_figures/eval_{rank_label}": payload.figure,
                    f"rollout_figures/eval_{rank_label}_rmse": payload.rollout_rmse,
                    f"rollout_figures/eval_{rank_label}_trajectory_id": payload.trajectory_id,
                }
            )
            wandb_run.save_file(figure_path)
            try:
                import matplotlib.pyplot as plt

                plt.close(payload.figure)
            except Exception:
                pass

        wandb_run.update_summary(flatten_metrics("summary", summary))
        wandb_run.update_summary({"output_dir": str(output_dir)})
        wandb_run.save_file(history_path)
        wandb_run.save_file(summary_path)
        wandb_run.finish()

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
    batch_losses: list[float] = []
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

            detached_loss = float(total_loss.detach().cpu())
            accumulators["loss"] += detached_loss
            batch_losses.append(detached_loss)
            accumulators["phys_1step"] += float(phys_1step.detach().cpu())
            accumulators["rollout"] += float(rollout_loss.detach().cpu())
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Received an empty dataloader for baseline training.")

    return BaselineEpochMetrics(
        loss=accumulators["loss"] / num_batches,
        loss_std=float(torch.tensor(batch_losses, dtype=torch.float64).std(unbiased=False).item()),
        loss_min=min(batch_losses),
        loss_max=max(batch_losses),
        phys_1step=accumulators["phys_1step"] / num_batches,
        rollout=accumulators["rollout"] / num_batches,
    )
