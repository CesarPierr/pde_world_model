from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pdewm.models.dynamics.context import PhysicsContextEncoder, PhysicsContextEncoderConfig


class FiLMResidualBlock1D(nn.Module):
    def __init__(self, channels: int, context_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(1, channels)
        self.film1 = nn.Linear(context_dim, channels * 2)
        self.film2 = nn.Linear(context_dim, channels * 2)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(inputs)
        hidden = self.norm1(hidden)
        hidden = self._apply_film(hidden, context, self.film1)
        hidden = self.activation(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        hidden = self._apply_film(hidden, context, self.film2)
        return self.activation(inputs + hidden)

    def _apply_film(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        projector: nn.Linear,
    ) -> torch.Tensor:
        gamma, beta = projector(context).chunk(2, dim=-1)
        return hidden * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


@dataclass(slots=True)
class TransitionModel1DConfig:
    latent_channels: int
    num_pdes: int
    continuous_context_dim: int
    hidden_channels: int = 64
    context_hidden_dim: int = 64
    context_output_dim: int = 64
    pde_embedding_dim: int = 8
    num_blocks: int = 4
    kernel_size: int = 5


class LatentTransitionModel1D(nn.Module):
    def __init__(self, config: TransitionModel1DConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = PhysicsContextEncoder(
            PhysicsContextEncoderConfig(
                num_pdes=config.num_pdes,
                continuous_dim=config.continuous_context_dim,
                pde_embedding_dim=config.pde_embedding_dim,
                hidden_dim=config.context_hidden_dim,
                output_dim=config.context_output_dim,
            )
        )
        self.input_projection = nn.Conv1d(config.latent_channels, config.hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock1D(
                    channels=config.hidden_channels,
                    context_dim=config.context_output_dim,
                    kernel_size=config.kernel_size,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.output_projection = nn.Conv1d(config.hidden_channels, config.latent_channels, kernel_size=1)

    def forward(
        self,
        latent: torch.Tensor,
        pde_ids: torch.Tensor,
        continuous_context: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context_encoder(pde_ids, continuous_context)
        hidden = self.input_projection(latent)
        for block in self.blocks:
            hidden = block(hidden, context)
        delta = self.output_projection(hidden)
        return latent + delta


def rollout_latent_dynamics(
    model: LatentTransitionModel1D,
    latent0: torch.Tensor,
    pde_ids: torch.Tensor,
    continuous_context: torch.Tensor,
) -> torch.Tensor:
    predictions = []
    current = latent0
    horizon = pde_ids.shape[1]

    for step in range(horizon):
        current = model(current, pde_ids[:, step], continuous_context[:, step])
        predictions.append(current)

    return torch.stack(predictions, dim=1)
