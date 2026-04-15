from __future__ import annotations

from omegaconf import DictConfig

from pdewm.baselines.cnn_ar_1d import CNNAutoregressive1D, CNNAutoregressive1DConfig
from pdewm.baselines.fno_1d import FNO1D, FNO1DConfig
from pdewm.baselines.pod_mlp import PODMLPBaseline, PODMLPConfig
from pdewm.baselines.unet_ar_1d import UNetAutoregressive1D, UNetAutoregressive1DConfig


def build_baseline_model(cfg: DictConfig, *, state_size: int):
    name = str(cfg.name)
    if name == "cnn_ar_1d":
        return CNNAutoregressive1D(
            CNNAutoregressive1DConfig(
                in_channels=int(cfg.in_channels),
                hidden_channels=int(cfg.hidden_channels),
                num_blocks=int(cfg.num_blocks),
                kernel_size=int(cfg.kernel_size),
            )
        )
    if name == "unet_ar_1d":
        return UNetAutoregressive1D(
            UNetAutoregressive1DConfig(
                in_channels=int(cfg.in_channels),
                base_channels=int(cfg.base_channels),
                depth=int(cfg.depth),
                kernel_size=int(cfg.kernel_size),
            )
        )
    if name == "fno_1d":
        return FNO1D(
            FNO1DConfig(
                in_channels=int(cfg.in_channels),
                hidden_channels=int(cfg.hidden_channels),
                modes=int(cfg.modes),
                num_layers=int(cfg.num_layers),
            )
        )
    if name == "pod_mlp_1d":
        return PODMLPBaseline(
            PODMLPConfig(
                state_size=state_size,
                rank=int(cfg.rank),
                hidden_dim=int(cfg.hidden_dim),
                num_layers=int(cfg.num_layers),
            )
        )
    raise ValueError(f"Unsupported baseline model: {name}")
