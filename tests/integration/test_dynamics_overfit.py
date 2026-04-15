from __future__ import annotations

import torch
import torch.nn.functional as F

from pdewm.models.dynamics.transition_1d import LatentTransitionModel1D, TransitionModel1DConfig


def test_transition_model_can_fit_small_latent_mapping() -> None:
    torch.manual_seed(7)
    model = LatentTransitionModel1D(
        TransitionModel1DConfig(
            latent_channels=4,
            num_pdes=1,
            continuous_context_dim=3,
            hidden_channels=16,
            context_hidden_dim=16,
            context_output_dim=16,
            num_blocks=2,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    latent = torch.randn(6, 4, 10)
    context = torch.randn(6, 3)
    pde_ids = torch.zeros(6, dtype=torch.long)
    target = latent + 0.25

    initial = F.mse_loss(model(latent, pde_ids, context), target)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(latent, pde_ids, context)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()

    final = F.mse_loss(model(latent, pde_ids, context), target)
    assert final.item() < initial.item() * 0.2
