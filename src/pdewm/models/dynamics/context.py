from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class PhysicsContextEncoderConfig:
    num_pdes: int
    continuous_dim: int
    pde_embedding_dim: int = 8
    hidden_dim: int = 64
    output_dim: int = 64


class PhysicsContextEncoder(nn.Module):
    def __init__(self, config: PhysicsContextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.pde_embedding = nn.Embedding(config.num_pdes, config.pde_embedding_dim)
        self.continuous_encoder = nn.Sequential(
            nn.Linear(config.continuous_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim + config.pde_embedding_dim, config.output_dim),
            nn.SiLU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

    def forward(self, pde_ids: torch.Tensor, continuous_context: torch.Tensor) -> torch.Tensor:
        pde_embedding = self.pde_embedding(pde_ids)
        continuous_embedding = self.continuous_encoder(continuous_context)
        return self.output_layer(torch.cat([pde_embedding, continuous_embedding], dim=-1))

