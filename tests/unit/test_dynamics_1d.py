from __future__ import annotations

import torch

from pdewm.models.dynamics.transition_1d import (
    LatentTransitionModel1D,
    TransitionModel1DConfig,
    rollout_latent_dynamics,
)


def test_transition_model_1d_preserves_latent_shape() -> None:
    model = LatentTransitionModel1D(
        TransitionModel1DConfig(
            latent_channels=16,
            num_pdes=2,
            continuous_context_dim=8,
            hidden_channels=32,
            context_hidden_dim=32,
            context_output_dim=32,
            num_blocks=2,
        )
    )
    latent = torch.randn(4, 16, 16)
    pde_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    context = torch.randn(4, 8)

    prediction = model(latent, pde_ids, context)

    assert prediction.shape == latent.shape


def test_rollout_latent_dynamics_returns_sequence() -> None:
    model = LatentTransitionModel1D(
        TransitionModel1DConfig(
            latent_channels=8,
            num_pdes=1,
            continuous_context_dim=4,
            hidden_channels=16,
            context_hidden_dim=16,
            context_output_dim=16,
            num_blocks=2,
        )
    )
    latent0 = torch.randn(3, 8, 12)
    pde_ids = torch.zeros(3, 2, dtype=torch.long)
    context = torch.randn(3, 2, 4)

    rollout = rollout_latent_dynamics(model, latent0, pde_ids, context)

    assert rollout.shape == (3, 2, 8, 12)

