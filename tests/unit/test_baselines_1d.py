from __future__ import annotations

import torch

from pdewm.baselines.cnn_ar_1d import CNNAutoregressive1D, CNNAutoregressive1DConfig
from pdewm.baselines.common import rollout_state_model
from pdewm.baselines.fno_1d import FNO1D, FNO1DConfig
from pdewm.baselines.pod_mlp import PODMLPBaseline, PODMLPConfig
from pdewm.baselines.unet_ar_1d import UNetAutoregressive1D, UNetAutoregressive1DConfig


def test_cnn_autoregressive_1d_preserves_shape() -> None:
    model = CNNAutoregressive1D(CNNAutoregressive1DConfig(in_channels=1, hidden_channels=16, num_blocks=2))
    states = torch.randn(4, 1, 64)
    prediction = model(states)
    assert prediction.shape == states.shape


def test_unet_autoregressive_1d_preserves_shape() -> None:
    model = UNetAutoregressive1D(UNetAutoregressive1DConfig(in_channels=1, base_channels=8, depth=2))
    states = torch.randn(4, 1, 64)
    prediction = model(states)
    assert prediction.shape == states.shape


def test_fno_1d_preserves_shape() -> None:
    model = FNO1D(FNO1DConfig(in_channels=1, hidden_channels=16, modes=8, num_layers=2))
    states = torch.randn(4, 1, 64)
    prediction = model(states)
    assert prediction.shape == states.shape


def test_pod_mlp_baseline_preserves_shape_after_fit() -> None:
    model = PODMLPBaseline(PODMLPConfig(state_size=64, rank=8, hidden_dim=32, num_layers=2))
    states = torch.randn(16, 1, 64)
    model.fit_basis(states)
    prediction = model(states[:4])
    assert prediction.shape == (4, 1, 64)


def test_rollout_state_model_returns_sequence() -> None:
    model = CNNAutoregressive1D(CNNAutoregressive1DConfig(in_channels=1, hidden_channels=8, num_blocks=1))
    states = torch.randn(3, 1, 64)
    rollout = rollout_state_model(model, states, horizon=3)
    assert rollout.shape == (3, 3, 1, 64)
