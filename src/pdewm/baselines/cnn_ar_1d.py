from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class CNNResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


@dataclass(slots=True)
class CNNAutoregressive1DConfig:
    in_channels: int = 1
    hidden_channels: int = 64
    num_blocks: int = 4
    kernel_size: int = 5


class CNNAutoregressive1D(nn.Module):
    def __init__(self, config: CNNAutoregressive1DConfig) -> None:
        super().__init__()
        padding = config.kernel_size // 2
        self.stem = nn.Conv1d(
            config.in_channels,
            config.hidden_channels,
            kernel_size=config.kernel_size,
            padding=padding,
        )
        self.blocks = nn.ModuleList(
            [CNNResidualBlock1D(config.hidden_channels, config.kernel_size) for _ in range(config.num_blocks)]
        )
        self.output_layer = nn.Conv1d(
            config.hidden_channels,
            config.in_channels,
            kernel_size=config.kernel_size,
            padding=padding,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(states)
        for block in self.blocks:
            hidden = block(hidden)
        return states + self.output_layer(hidden)

