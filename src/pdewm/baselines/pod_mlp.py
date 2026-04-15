from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class PODMLPConfig:
    state_size: int
    rank: int = 16
    hidden_dim: int = 128
    num_layers: int = 2


class PODMLPBaseline(nn.Module):
    def __init__(self, config: PODMLPConfig) -> None:
        super().__init__()
        self.config = config
        hidden_layers: list[nn.Module] = []
        input_dim = config.rank
        for _ in range(config.num_layers):
            hidden_layers.extend([nn.Linear(input_dim, config.hidden_dim), nn.SiLU()])
            input_dim = config.hidden_dim
        hidden_layers.append(nn.Linear(input_dim, config.rank))
        self.mlp = nn.Sequential(*hidden_layers)
        self.register_buffer("mean", torch.zeros(config.state_size))
        self.register_buffer("basis", torch.zeros(config.rank, config.state_size))
        self._basis_fitted = False

    def fit_basis(self, states: torch.Tensor) -> None:
        flattened = states.reshape(states.shape[0], -1)
        mean = flattened.mean(dim=0)
        centered = flattened - mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        rank = min(self.config.rank, vh.shape[0])
        self.mean.copy_(mean)
        self.basis.zero_()
        self.basis[:rank].copy_(vh[:rank])
        self._basis_fitted = True

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        if not self._basis_fitted:
            raise RuntimeError("POD basis must be fitted before encoding states.")
        flattened = states.reshape(states.shape[0], -1)
        return (flattened - self.mean) @ self.basis.T

    def decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        reconstructed = coefficients @ self.basis + self.mean
        return reconstructed.view(coefficients.shape[0], 1, self.config.state_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        coefficients = self.encode(states)
        next_coefficients = coefficients + self.mlp(coefficients)
        return self.decode(next_coefficients)

