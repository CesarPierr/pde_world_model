from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class AutoencoderLossWeights:
    l1: float = 1.0
    l2: float = 1.0
    gradient: float = 0.1
    spectral: float = 0.1


def reconstruction_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target)


def reconstruction_l2_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_grad = torch.diff(prediction, dim=-1)
    target_grad = torch.diff(target, dim=-1)
    return F.l1_loss(pred_grad, target_grad)


def spectral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft(prediction, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


def compute_autoencoder_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: AutoencoderLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components = {
        "l1": reconstruction_l1_loss(prediction, target),
        "l2": reconstruction_l2_loss(prediction, target),
        "gradient": gradient_loss(prediction, target),
        "spectral": spectral_loss(prediction, target),
    }
    total = (
        weights.l1 * components["l1"]
        + weights.l2 * components["l2"]
        + weights.gradient * components["gradient"]
        + weights.spectral * components["spectral"]
    )
    return total, components

