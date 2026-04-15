from __future__ import annotations

import torch

from pdewm.models.representations.autoencoder_1d import Autoencoder1D, Autoencoder1DConfig


def test_autoencoder_1d_preserves_shape_and_latent_grid() -> None:
    model = Autoencoder1D(
        Autoencoder1DConfig(
            in_channels=1,
            base_channels=8,
            channel_multipliers=(1, 2, 4),
            latent_channels=16,
            kernel_size=5,
        )
    )
    batch = torch.randn(4, 1, 64)

    reconstruction, latent = model(batch)

    assert reconstruction.shape == batch.shape
    assert latent.shape == (4, 16, 16)

