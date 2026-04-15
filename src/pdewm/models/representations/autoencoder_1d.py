from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
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
class Autoencoder1DConfig:
    in_channels: int = 1
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    latent_channels: int = 64
    kernel_size: int = 5


class Encoder1D(nn.Module):
    def __init__(self, config: Autoencoder1DConfig) -> None:
        super().__init__()
        channels = [config.base_channels * mult for mult in config.channel_multipliers]
        padding = config.kernel_size // 2
        self.stem = nn.Conv1d(
            config.in_channels,
            channels[0],
            kernel_size=config.kernel_size,
            padding=padding,
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock1D(channel, kernel_size=config.kernel_size) for channel in channels]
        )
        self.downsamplers = nn.ModuleList(
            [
                nn.Conv1d(channels[idx], channels[idx + 1], kernel_size=4, stride=2, padding=1)
                for idx in range(len(channels) - 1)
            ]
        )
        self.latent_projection = nn.Conv1d(channels[-1], config.latent_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(inputs)
        for idx, block in enumerate(self.blocks):
            hidden = block(hidden)
            if idx < len(self.downsamplers):
                hidden = self.downsamplers[idx](hidden)
        return self.latent_projection(hidden)


class Decoder1D(nn.Module):
    def __init__(self, config: Autoencoder1DConfig) -> None:
        super().__init__()
        channels = [config.base_channels * mult for mult in config.channel_multipliers]
        padding = config.kernel_size // 2
        self.latent_projection = nn.Conv1d(config.latent_channels, channels[-1], kernel_size=1)
        reversed_channels = list(reversed(channels))
        self.upsamplers = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    reversed_channels[idx],
                    reversed_channels[idx + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
                for idx in range(len(reversed_channels) - 1)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock1D(channel, kernel_size=config.kernel_size)
                for channel in reversed_channels[1:]
            ]
        )
        self.output_layer = nn.Conv1d(
            channels[0],
            config.in_channels,
            kernel_size=config.kernel_size,
            padding=padding,
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.latent_projection(latents)
        for upsampler, block in zip(self.upsamplers, self.blocks, strict=True):
            hidden = upsampler(hidden)
            hidden = block(hidden)
        return self.output_layer(hidden)


class Autoencoder1D(nn.Module):
    def __init__(self, config: Autoencoder1DConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder1D(config)
        self.decoder = Decoder1D(config)

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(states)
        reconstructions = self.decode(latents)
        return reconstructions, latents
