from __future__ import annotations

import torch

from pdewm.models.representations.losses import AutoencoderLossWeights, compute_autoencoder_losses


def test_autoencoder_losses_are_zero_for_perfect_reconstruction() -> None:
    target = torch.randn(2, 1, 32)
    total, components = compute_autoencoder_losses(target, target, AutoencoderLossWeights())

    assert torch.isclose(total, torch.tensor(0.0))
    for value in components.values():
        assert torch.isclose(value, torch.tensor(0.0))

