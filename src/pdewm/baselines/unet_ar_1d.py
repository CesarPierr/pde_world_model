from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int, kernel_size: int = 5) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.GroupNorm(1, out_channels),
        nn.SiLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.GroupNorm(1, out_channels),
        nn.SiLU(),
    )


@dataclass(slots=True)
class UNetAutoregressive1DConfig:
    in_channels: int = 1
    base_channels: int = 32
    depth: int = 2
    kernel_size: int = 5


class UNetAutoregressive1D(nn.Module):
    def __init__(self, config: UNetAutoregressive1DConfig) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        channels = config.base_channels
        in_channels = config.in_channels

        for _ in range(config.depth):
            self.encoders.append(_conv_block(in_channels, channels, config.kernel_size))
            self.downsamplers.append(nn.Conv1d(channels, channels * 2, kernel_size=4, stride=2, padding=1))
            in_channels = channels * 2
            channels *= 2

        self.bottleneck = _conv_block(in_channels, channels, config.kernel_size)
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for _ in range(config.depth):
            self.upsamplers.append(nn.ConvTranspose1d(channels, channels // 2, kernel_size=4, stride=2, padding=1))
            self.decoders.append(_conv_block(channels, channels // 2, config.kernel_size))
            channels //= 2

        padding = config.kernel_size // 2
        self.output_layer = nn.Conv1d(channels, config.in_channels, kernel_size=config.kernel_size, padding=padding)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        skips = []
        hidden = states
        for encoder, downsampler in zip(self.encoders, self.downsamplers, strict=True):
            hidden = encoder(hidden)
            skips.append(hidden)
            hidden = downsampler(hidden)

        hidden = self.bottleneck(hidden)

        for upsampler, decoder in zip(self.upsamplers, self.decoders, strict=True):
            skip = skips.pop()
            hidden = upsampler(hidden)
            if hidden.shape[-1] != skip.shape[-1]:
                hidden = hidden[..., : skip.shape[-1]]
            hidden = torch.cat([hidden, skip], dim=1)
            hidden = decoder(hidden)

        return states + self.output_layer(hidden)

