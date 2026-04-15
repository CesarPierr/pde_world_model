from __future__ import annotations

import torch
from torch import nn


def rollout_state_model(model: nn.Module, state0: torch.Tensor, horizon: int) -> torch.Tensor:
    predictions = []
    current = state0
    for _ in range(horizon):
        current = model(current)
        predictions.append(current)
    return torch.stack(predictions, dim=1)

