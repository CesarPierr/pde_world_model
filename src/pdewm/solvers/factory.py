from __future__ import annotations

from omegaconf import DictConfig

from pdewm.solvers.base import BasePDESolver
from pdewm.solvers.burgers_1d import Burgers1DSolver
from pdewm.solvers.kuramoto_sivashinsky_1d import KuramotoSivashinsky1DSolver


def build_solver(cfg: DictConfig) -> BasePDESolver:
    name = cfg.name
    if name == "burgers_1d":
        return Burgers1DSolver(
            grid_size=int(cfg.grid_size),
            domain_length=float(cfg.domain_length),
            dealias=bool(cfg.dealias),
        )
    if name == "ks_1d":
        return KuramotoSivashinsky1DSolver(
            grid_size=int(cfg.grid_size),
            domain_length=float(cfg.domain_length),
            contour_points=int(cfg.contour_points),
            dealias=bool(cfg.dealias),
        )
    raise ValueError(f"Unsupported solver: {name}")
