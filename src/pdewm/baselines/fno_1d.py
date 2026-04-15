from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, signal_length = inputs.shape
        inputs_fft = torch.fft.rfft(inputs, dim=-1)
        out_fft = torch.zeros(
            batch_size,
            self.out_channels,
            signal_length // 2 + 1,
            device=inputs.device,
            dtype=torch.cfloat,
        )
        modes = min(self.modes, inputs_fft.shape[-1], self.weight.shape[-1])
        out_fft[..., :modes] = torch.einsum(
            "bim,iom->bom",
            inputs_fft[..., :modes],
            self.weight[..., :modes],
        )
        return torch.fft.irfft(out_fft, n=signal_length, dim=-1)


@dataclass(slots=True)
class FNO1DConfig:
    in_channels: int = 1
    hidden_channels: int = 64
    modes: int = 16
    num_layers: int = 4


class FNO1D(nn.Module):
    def __init__(self, config: FNO1DConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.in_channels + 1, config.hidden_channels)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(config.hidden_channels, config.hidden_channels, config.modes) for _ in range(config.num_layers)]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv1d(config.hidden_channels, config.hidden_channels, kernel_size=1) for _ in range(config.num_layers)]
        )
        self.output_projection = nn.Sequential(
            nn.Conv1d(config.hidden_channels, config.hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(config.hidden_channels, config.in_channels, kernel_size=1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, _, signal_length = states.shape
        grid = torch.linspace(0.0, 1.0, signal_length, device=states.device, dtype=states.dtype)
        grid = grid.view(1, 1, signal_length).expand(batch_size, -1, -1)
        lifted = torch.cat([states, grid], dim=1).permute(0, 2, 1)
        hidden = self.input_projection(lifted).permute(0, 2, 1)

        for spectral_layer, pointwise_layer in zip(self.spectral_layers, self.pointwise_layers, strict=True):
            hidden = torch.nn.functional.silu(spectral_layer(hidden) + pointwise_layer(hidden))

        return states + self.output_projection(hidden)

