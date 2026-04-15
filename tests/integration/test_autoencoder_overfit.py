from __future__ import annotations

import math

import torch

from pdewm.models.representations.autoencoder_1d import Autoencoder1D, Autoencoder1DConfig
from pdewm.models.representations.losses import AutoencoderLossWeights, compute_autoencoder_losses


def test_autoencoder_can_overfit_small_sine_batch() -> None:
    torch.manual_seed(7)
    grid = torch.linspace(0.0, 2.0 * math.pi, steps=64)
    batch = torch.stack(
        [
            torch.sin(grid)[None, :],
            torch.cos(grid)[None, :],
            torch.sin(2.0 * grid)[None, :],
            torch.cos(2.0 * grid)[None, :],
        ],
        dim=0,
    )
    model = Autoencoder1D(
        Autoencoder1DConfig(
            in_channels=1,
            base_channels=8,
            channel_multipliers=(1, 2, 4),
            latent_channels=16,
            kernel_size=5,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    weights = AutoencoderLossWeights(l1=1.0, l2=1.0, gradient=0.05, spectral=0.05)

    reconstruction, _ = model(batch)
    initial_loss, _ = compute_autoencoder_losses(reconstruction, batch, weights)

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, _ = model(batch)
        loss, _ = compute_autoencoder_losses(reconstruction, batch, weights)
        loss.backward()
        optimizer.step()

    final_reconstruction, _ = model(batch)
    final_loss, _ = compute_autoencoder_losses(final_reconstruction, batch, weights)

    assert final_loss.item() < initial_loss.item() * 0.3
