from __future__ import annotations

import copy
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
from pdewm.data.datasets import TransitionWindowDataset, load_trajectory_samples
from pdewm.evaluation.rollout_figures import build_world_model_rollout_figures
from pdewm.evaluation.trajectory_metrics import evaluate_world_model_trajectories
from pdewm.models.dynamics.transition_1d import (
    LatentTransitionModel1D,
    TransitionModel1DConfig,
    rollout_latent_dynamics,
)
from pdewm.models.representations.autoencoder_1d import Autoencoder1D, Autoencoder1DConfig
from pdewm.models.representations.losses import AutoencoderLossWeights, compute_autoencoder_losses
from pdewm.utils.device import resolve_device
from pdewm.utils.wandb import (
    compose_wandb_group,
    compose_wandb_name,
    flatten_metrics,
    init_wandb_run,
)


@dataclass(slots=True)
class ReconstructionMetrics:
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


@dataclass(slots=True)
class WorldModelEpochMetrics:
    loss: float
    loss_std: float
    loss_min: float
    loss_max: float
    dynamics_loss: float
    ae_loss: float
    latent_1step: float
    phys_1step: float
    rollout: float
    consistency: float
    ae_l1: float
    ae_l2: float
    ae_gradient: float
    ae_spectral: float

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "loss_std": self.loss_std,
            "loss_min": self.loss_min,
            "loss_max": self.loss_max,
            "dynamics_loss": self.dynamics_loss,
            "ae_loss": self.ae_loss,
            "latent_1step": self.latent_1step,
            "phys_1step": self.phys_1step,
            "rollout": self.rollout,
            "consistency": self.consistency,
            "ae_l1": self.ae_l1,
            "ae_l2": self.ae_l2,
            "ae_gradient": self.ae_gradient,
            "ae_spectral": self.ae_spectral,
        }


def train_latent_dynamics(cfg: DictConfig) -> dict[str, Any]:
    device = resolve_device(str(cfg.train.device))
    regime = str(OmegaConf.select(cfg, "train.regime") or "frozen")
    if regime not in {"frozen", "joint_no_ema", "joint_ema"}:
        raise ValueError(f"Unsupported training regime: {regime}")

    context_features = tuple(cfg.train.context_features) if cfg.train.context_features else DEFAULT_CONTEXT_FEATURES
    ae_loss_weights = _build_ae_loss_weights(cfg)
    ae_loss_scale = float(OmegaConf.select(cfg, "train.ae_loss_scale") or 1.0)
    resume_checkpoint = OmegaConf.select(cfg, "train.resume_checkpoint")

    train_dataset = TransitionWindowDataset(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.train_splits),
        rollout_horizon=int(cfg.train.rollout_horizon),
        context_feature_keys=context_features,
    )
    eval_dataset = TransitionWindowDataset(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.eval_splits),
        context_feature_keys=context_features,
        rollout_horizon=int(cfg.train.rollout_horizon),
    )
    eval_trajectories = load_trajectory_samples(
        cfg.train.dataset_root,
        splits=tuple(cfg.train.eval_splits),
        context_feature_keys=context_features,
    )

    autoencoder = _initialize_autoencoder(cfg, device, trainable=regime != "frozen")
    transition_model = _build_transition_model(cfg, autoencoder, train_dataset, context_features).to(device)
    ema_autoencoder = _initialize_ema_autoencoder(autoencoder, regime=regime)

    optimizer = _build_optimizer(cfg, transition_model, autoencoder, regime=regime)

    if resume_checkpoint:
        _load_training_state(
            Path(str(resume_checkpoint)),
            transition_model,
            autoencoder,
            ema_autoencoder,
            optimizer,
            device=device,
            regime=regime,
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
            "dynamics",
            regime,
            Path(str(cfg.train.dataset_root)).name,
            output_dir.name,
            f"seed {int(cfg.project.seed)}",
        ),
        default_group=compose_wandb_group("dynamics", regime, Path(str(cfg.train.dataset_root)).name),
        default_job_type="train_dynamics",
        extra_tags=[
            "dynamics_1d",
            regime,
            str(OmegaConf.select(cfg, "project.phase") or "unknown_phase"),
        ],
    )
    epoch_offset = int(OmegaConf.select(cfg, "train.epoch_offset") or 0)

    try:
        for epoch in range(1, int(cfg.train.epochs) + 1):
            train_metrics = _run_world_model_epoch(
                transition_model=transition_model,
                autoencoder=autoencoder,
                ema_autoencoder=ema_autoencoder,
                dataloader=train_loader,
                optimizer=optimizer,
                dynamics_loss_weights=cfg.train.loss_weights,
                ae_loss_weights=ae_loss_weights,
                device=device,
                regime=regime,
                ae_loss_scale=ae_loss_scale,
                ema_decay=float(OmegaConf.select(cfg, "train.ema.decay") or 0.995),
                training=True,
            )
            eval_metrics = _run_world_model_epoch(
                transition_model=transition_model,
                autoencoder=autoencoder,
                ema_autoencoder=ema_autoencoder,
                dataloader=eval_loader,
                optimizer=optimizer,
                dynamics_loss_weights=cfg.train.loss_weights,
                ae_loss_weights=ae_loss_weights,
                device=device,
                regime=regime,
                ae_loss_scale=ae_loss_scale,
                ema_decay=float(OmegaConf.select(cfg, "train.ema.decay") or 0.995),
                training=False,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics.to_dict(),
                    "eval": eval_metrics.to_dict(),
                }
            )
            checkpoint = _build_checkpoint(
                cfg=cfg,
                epoch=epoch,
                transition_model=transition_model,
                autoencoder=autoencoder,
                ema_autoencoder=ema_autoencoder,
                optimizer=optimizer,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
                train_dataset=train_dataset,
                context_features=context_features,
                regime=regime,
            )
            torch.save(checkpoint, output_dir / "last.pt")

            wandb_run.log(
                {
                    "epoch": epoch,
                    **flatten_metrics("train", train_metrics.to_dict()),
                    **flatten_metrics("eval", eval_metrics.to_dict()),
                },
                step=epoch_offset + epoch,
            )
            last_epoch = epoch
            last_train_metrics = train_metrics
            last_eval_metrics = eval_metrics

        eval_final_metrics = _run_world_model_epoch(
            transition_model=transition_model,
            autoencoder=autoencoder,
            ema_autoencoder=ema_autoencoder,
            dataloader=eval_loader,
            optimizer=optimizer,
            dynamics_loss_weights=cfg.train.loss_weights,
            ae_loss_weights=ae_loss_weights,
            device=device,
            regime=regime,
            ae_loss_scale=ae_loss_scale,
            ema_decay=float(OmegaConf.select(cfg, "train.ema.decay") or 0.995),
            training=False,
        )
        ae_eval_metrics = _evaluate_autoencoder_trajectories(
            autoencoder,
            eval_trajectories,
            weights=ae_loss_weights,
            device=device,
        )
        trajectory_eval_metrics = evaluate_world_model_trajectories(
            transition_model,
            autoencoder,
            eval_trajectories,
            device=device,
        )

        rollout_eval_figures = build_world_model_rollout_figures(
            transition_model,
            autoencoder,
            eval_trajectories,
            device=device,
            split_name="eval",
        )

        summary = {
            "regime": regime,
            "final_epoch": int(epoch_offset + last_epoch),
            "final_eval_loss": float(eval_final_metrics.loss),
            "acquisition_autoencoder_source": _acquisition_autoencoder_source(regime),
            "train_metrics": last_train_metrics.to_dict(),
            "eval_metrics": eval_final_metrics.to_dict(),
            "ae_eval_metrics": ae_eval_metrics.to_dict(),
            "trajectory_eval_metrics": trajectory_eval_metrics.to_dict(),
            "selected_checkpoint": str(output_dir / "last.pt"),
        }
    finally:
        history_path = output_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        summary = locals().get(
            "summary",
            {
                "regime": regime,
                "final_epoch": int(epoch_offset + int(locals().get("last_epoch", 0))),
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
                    f"simulations/eval_{rank_label}_txg": payload.figure,
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

    return {"history": history, **summary}


def load_trained_dynamics(
    checkpoint_path: str | Path,
    ae_checkpoint_path: str | Path | None = None,
    *,
    device: str | torch.device = "auto",
    autoencoder_role: str = "acquisition",
) -> tuple[LatentTransitionModel1D, Autoencoder1D, dict[str, Any]]:
    device = resolve_device(str(device))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["config"]["model"]
    train_cfg = checkpoint["config"]["train"]
    ae_config = _build_autoencoder_config_from_checkpoint(checkpoint, ae_checkpoint_path, device)
    autoencoder = Autoencoder1D(ae_config).to(device)
    autoencoder_state = _resolve_autoencoder_state_dict(
        checkpoint,
        ae_checkpoint_path=ae_checkpoint_path,
        autoencoder_role=autoencoder_role,
        device=device,
    )
    autoencoder.load_state_dict(autoencoder_state)
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    model = LatentTransitionModel1D(
        TransitionModel1DConfig(
            latent_channels=int(ae_config.latent_channels),
            num_pdes=int(len(checkpoint["pde_to_index"])),
            continuous_context_dim=len(checkpoint["context_features"]),
            hidden_channels=int(model_cfg["hidden_channels"]),
            context_hidden_dim=int(model_cfg["context_hidden_dim"]),
            context_output_dim=int(model_cfg["context_output_dim"]),
            pde_embedding_dim=int(model_cfg["pde_embedding_dim"]),
            num_blocks=int(model_cfg["num_blocks"]),
            kernel_size=int(model_cfg["kernel_size"]),
        )
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metadata = {
        "pde_to_index": checkpoint["pde_to_index"],
        "context_features": tuple(checkpoint["context_features"]),
        "rollout_horizon": int(train_cfg["rollout_horizon"]),
        "regime": str(checkpoint.get("regime", train_cfg.get("regime", "frozen"))),
        "acquisition_autoencoder_source": str(
            checkpoint.get("acquisition_autoencoder_source", "student")
        ),
    }
    return model, autoencoder, metadata


def _run_world_model_epoch(
    *,
    transition_model: nn.Module,
    autoencoder: Autoencoder1D,
    ema_autoencoder: Autoencoder1D | None,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    dynamics_loss_weights: DictConfig,
    ae_loss_weights: AutoencoderLossWeights,
    device: torch.device,
    regime: str,
    ae_loss_scale: float,
    ema_decay: float,
    training: bool,
) -> WorldModelEpochMetrics:
    transition_model.train(training)
    autoencoder.train(training and regime != "frozen")
    if ema_autoencoder is not None:
        ema_autoencoder.eval()

    accumulators = {
        "loss": 0.0,
        "dynamics_loss": 0.0,
        "ae_loss": 0.0,
        "latent_1step": 0.0,
        "phys_1step": 0.0,
        "rollout": 0.0,
        "consistency": 0.0,
        "ae_l1": 0.0,
        "ae_l2": 0.0,
        "ae_gradient": 0.0,
        "ae_spectral": 0.0,
    }
    batch_losses: list[float] = []
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

            flat_states = torch.cat([state.unsqueeze(1), future_states], dim=1)
            flat_states = flat_states.reshape(-1, *state.shape[1:])
            if regime == "frozen":
                with torch.no_grad():
                    reconstructions, _ = autoencoder(flat_states)
                    ae_total_loss, ae_components = compute_autoencoder_losses(
                        reconstructions,
                        flat_states,
                        ae_loss_weights,
                    )
                    latent0 = autoencoder.encode(state)
                    target_latents = _encode_future_latents(autoencoder, future_states)
            else:
                reconstructions, _ = autoencoder(flat_states)
                ae_total_loss, ae_components = compute_autoencoder_losses(
                    reconstructions,
                    flat_states,
                    ae_loss_weights,
                )
                latent0 = autoencoder.encode(state)
                teacher_autoencoder = ema_autoencoder if ema_autoencoder is not None else autoencoder
                with torch.no_grad():
                    target_latents = _encode_future_latents(teacher_autoencoder, future_states)

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
            dynamics_loss = (
                float(dynamics_loss_weights.latent_1step) * latent_1step
                + float(dynamics_loss_weights.phys_1step) * phys_1step
                + float(dynamics_loss_weights.rollout) * rollout_loss
                + float(dynamics_loss_weights.consistency) * consistency_loss
            )
            total_loss = dynamics_loss
            if regime != "frozen":
                total_loss = total_loss + ae_loss_scale * ae_total_loss

            if training:
                total_loss.backward()
                optimizer.step()
                if ema_autoencoder is not None:
                    _update_ema(autoencoder, ema_autoencoder, decay=ema_decay)

            detached_loss = float(total_loss.detach().cpu())
            accumulators["loss"] += detached_loss
            batch_losses.append(detached_loss)
            accumulators["dynamics_loss"] += float(dynamics_loss.detach().cpu())
            accumulators["ae_loss"] += float(ae_total_loss.detach().cpu())
            accumulators["latent_1step"] += float(latent_1step.detach().cpu())
            accumulators["phys_1step"] += float(phys_1step.detach().cpu())
            accumulators["rollout"] += float(rollout_loss.detach().cpu())
            accumulators["consistency"] += float(consistency_loss.detach().cpu())
            accumulators["ae_l1"] += float(ae_components["l1"].detach().cpu())
            accumulators["ae_l2"] += float(ae_components["l2"].detach().cpu())
            accumulators["ae_gradient"] += float(ae_components["gradient"].detach().cpu())
            accumulators["ae_spectral"] += float(ae_components["spectral"].detach().cpu())
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Received an empty dataloader for world-model training.")

    return WorldModelEpochMetrics(
        loss=accumulators["loss"] / num_batches,
        loss_std=float(torch.tensor(batch_losses, dtype=torch.float64).std(unbiased=False).item()),
        loss_min=min(batch_losses),
        loss_max=max(batch_losses),
        dynamics_loss=accumulators["dynamics_loss"] / num_batches,
        ae_loss=accumulators["ae_loss"] / num_batches,
        latent_1step=accumulators["latent_1step"] / num_batches,
        phys_1step=accumulators["phys_1step"] / num_batches,
        rollout=accumulators["rollout"] / num_batches,
        consistency=accumulators["consistency"] / num_batches,
        ae_l1=accumulators["ae_l1"] / num_batches,
        ae_l2=accumulators["ae_l2"] / num_batches,
        ae_gradient=accumulators["ae_gradient"] / num_batches,
        ae_spectral=accumulators["ae_spectral"] / num_batches,
    )


def _evaluate_autoencoder_trajectories(
    autoencoder: Autoencoder1D,
    trajectories,
    *,
    weights: AutoencoderLossWeights,
    device: torch.device,
) -> ReconstructionMetrics:
    autoencoder.eval()
    accumulators = {"loss": 0.0, "l1": 0.0, "l2": 0.0, "gradient": 0.0, "spectral": 0.0}

    with torch.no_grad():
        for sample in trajectories:
            states = torch.from_numpy(sample.states).to(device)
            reconstructions, _ = autoencoder(states)
            total_loss, components = compute_autoencoder_losses(reconstructions, states, weights)
            accumulators["loss"] += float(total_loss.cpu())
            for key, value in components.items():
                accumulators[key] += float(value.cpu())

    if not trajectories:
        raise ValueError("Cannot evaluate autoencoder reconstruction on an empty trajectory set.")

    count = float(len(trajectories))
    return ReconstructionMetrics(
        loss=accumulators["loss"] / count,
        l1=accumulators["l1"] / count,
        l2=accumulators["l2"] / count,
        gradient=accumulators["gradient"] / count,
        spectral=accumulators["spectral"] / count,
    )


def _build_transition_model(
    cfg: DictConfig,
    autoencoder: Autoencoder1D,
    dataset: TransitionWindowDataset,
    context_features: tuple[str, ...],
) -> LatentTransitionModel1D:
    return LatentTransitionModel1D(
        TransitionModel1DConfig(
            latent_channels=int(autoencoder.config.latent_channels),
            num_pdes=int(len(dataset.pde_to_index)),
            continuous_context_dim=len(context_features),
            hidden_channels=int(cfg.model.hidden_channels),
            context_hidden_dim=int(cfg.model.context_hidden_dim),
            context_output_dim=int(cfg.model.context_output_dim),
            pde_embedding_dim=int(cfg.model.pde_embedding_dim),
            num_blocks=int(cfg.model.num_blocks),
            kernel_size=int(cfg.model.kernel_size),
        )
    )


def _initialize_autoencoder(
    cfg: DictConfig,
    device: torch.device,
    *,
    trainable: bool,
) -> Autoencoder1D:
    checkpoint = torch.load(str(cfg.train.ae_checkpoint), map_location=device)
    autoencoder = Autoencoder1D(_build_autoencoder_config_from_model_cfg(checkpoint["config"]["model"])).to(device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.train(trainable)
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(trainable)
    return autoencoder


def _initialize_ema_autoencoder(
    autoencoder: Autoencoder1D,
    *,
    regime: str,
) -> Autoencoder1D | None:
    if regime != "joint_ema":
        return None
    ema_autoencoder = copy.deepcopy(autoencoder)
    ema_autoencoder.eval()
    for parameter in ema_autoencoder.parameters():
        parameter.requires_grad_(False)
    return ema_autoencoder


def _build_optimizer(
    cfg: DictConfig,
    transition_model: LatentTransitionModel1D,
    autoencoder: Autoencoder1D,
    *,
    regime: str,
) -> torch.optim.Optimizer:
    learning_rate = float(cfg.train.learning_rate)
    weight_decay = float(cfg.train.weight_decay)
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": list(transition_model.parameters()),
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
    ]
    if regime != "frozen":
        ae_learning_rate = float(OmegaConf.select(cfg, "train.ae_learning_rate") or learning_rate)
        ae_weight_decay = float(OmegaConf.select(cfg, "train.ae_weight_decay") or weight_decay)
        parameter_groups.append(
            {
                "params": [parameter for parameter in autoencoder.parameters() if parameter.requires_grad],
                "lr": ae_learning_rate,
                "weight_decay": ae_weight_decay,
            }
        )
    return torch.optim.Adam(parameter_groups)


def _build_checkpoint(
    *,
    cfg: DictConfig,
    epoch: int,
    transition_model: LatentTransitionModel1D,
    autoencoder: Autoencoder1D,
    ema_autoencoder: Autoencoder1D | None,
    optimizer: torch.optim.Optimizer,
    train_metrics: WorldModelEpochMetrics,
    eval_metrics: WorldModelEpochMetrics,
    train_dataset: TransitionWindowDataset,
    context_features: tuple[str, ...],
    regime: str,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": transition_model.state_dict(),
        "ae_state_dict": autoencoder.state_dict(),
        "ema_ae_state_dict": ema_autoencoder.state_dict() if ema_autoencoder is not None else None,
        "ae_config": {
            "in_channels": int(autoencoder.config.in_channels),
            "base_channels": int(autoencoder.config.base_channels),
            "channel_multipliers": list(autoencoder.config.channel_multipliers),
            "latent_channels": int(autoencoder.config.latent_channels),
            "kernel_size": int(autoencoder.config.kernel_size),
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics.to_dict(),
        "eval_metrics": eval_metrics.to_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "pde_to_index": train_dataset.pde_to_index,
        "context_features": list(context_features),
        "regime": regime,
        "acquisition_autoencoder_source": _acquisition_autoencoder_source(regime),
    }


def _load_training_state(
    checkpoint_path: Path,
    transition_model: LatentTransitionModel1D,
    autoencoder: Autoencoder1D,
    ema_autoencoder: Autoencoder1D | None,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    regime: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_regime = str(checkpoint.get("regime", "frozen"))
    if checkpoint_regime != regime:
        raise ValueError(
            f"Resume checkpoint regime {checkpoint_regime!r} does not match requested regime {regime!r}."
        )
    _load_checkpoint_models(checkpoint, transition_model, autoencoder, ema_autoencoder, device=device)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def _load_checkpoint_models(
    checkpoint: dict[str, Any],
    transition_model: LatentTransitionModel1D,
    autoencoder: Autoencoder1D,
    ema_autoencoder: Autoencoder1D | None,
    *,
    device: torch.device,
) -> None:
    del device
    transition_model.load_state_dict(checkpoint["model_state_dict"])
    if "ae_state_dict" in checkpoint:
        autoencoder.load_state_dict(checkpoint["ae_state_dict"])
    if ema_autoencoder is not None and checkpoint.get("ema_ae_state_dict") is not None:
        ema_autoencoder.load_state_dict(checkpoint["ema_ae_state_dict"])


def _build_autoencoder_config_from_checkpoint(
    checkpoint: dict[str, Any],
    ae_checkpoint_path: str | Path | None,
    device: torch.device,
) -> Autoencoder1DConfig:
    if "ae_config" in checkpoint:
        return Autoencoder1DConfig(
            in_channels=int(checkpoint["ae_config"]["in_channels"]),
            base_channels=int(checkpoint["ae_config"]["base_channels"]),
            channel_multipliers=tuple(int(item) for item in checkpoint["ae_config"]["channel_multipliers"]),
            latent_channels=int(checkpoint["ae_config"]["latent_channels"]),
            kernel_size=int(checkpoint["ae_config"]["kernel_size"]),
        )
    if ae_checkpoint_path is None:
        raise ValueError("No embedded AE config found and no fallback AE checkpoint was provided.")
    ae_checkpoint = torch.load(str(ae_checkpoint_path), map_location=device)
    return _build_autoencoder_config_from_model_cfg(ae_checkpoint["config"]["model"])


def _resolve_autoencoder_state_dict(
    checkpoint: dict[str, Any],
    *,
    ae_checkpoint_path: str | Path | None,
    autoencoder_role: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if autoencoder_role not in {"acquisition", "student"}:
        raise ValueError(f"Unsupported autoencoder role: {autoencoder_role}")
    if autoencoder_role == "acquisition":
        preferred_source = str(checkpoint.get("acquisition_autoencoder_source", "student"))
        if preferred_source == "ema" and checkpoint.get("ema_ae_state_dict") is not None:
            return checkpoint["ema_ae_state_dict"]
    if checkpoint.get("ae_state_dict") is not None:
        return checkpoint["ae_state_dict"]
    if ae_checkpoint_path is None:
        raise ValueError("No AE state found in checkpoint and no fallback AE checkpoint was provided.")
    ae_checkpoint = torch.load(str(ae_checkpoint_path), map_location=device)
    return ae_checkpoint["model_state_dict"]


def _build_autoencoder_config_from_model_cfg(model_cfg: dict[str, Any]) -> Autoencoder1DConfig:
    return Autoencoder1DConfig(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        channel_multipliers=tuple(int(item) for item in model_cfg["channel_multipliers"]),
        latent_channels=int(model_cfg["latent_channels"]),
        kernel_size=int(model_cfg["kernel_size"]),
    )


def _build_ae_loss_weights(cfg: DictConfig) -> AutoencoderLossWeights:
    weights_cfg = OmegaConf.select(cfg, "train.ae_loss_weights")
    if weights_cfg is None:
        return AutoencoderLossWeights()
    return AutoencoderLossWeights(**OmegaConf.to_container(weights_cfg, resolve=True))


def _encode_future_latents(autoencoder: Autoencoder1D, future_states: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [autoencoder.encode(future_states[:, step]) for step in range(future_states.shape[1])],
        dim=1,
    )


def _update_ema(
    autoencoder: Autoencoder1D,
    ema_autoencoder: Autoencoder1D,
    *,
    decay: float,
) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(
            ema_autoencoder.parameters(),
            autoencoder.parameters(),
            strict=True,
        ):
            ema_parameter.data.mul_(decay).add_(parameter.data, alpha=1.0 - decay)


def _acquisition_autoencoder_source(regime: str) -> str:
    return "ema" if regime == "joint_ema" else "student"
