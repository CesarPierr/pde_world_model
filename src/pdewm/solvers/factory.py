from __future__ import annotations

from omegaconf import DictConfig

from pdewm.solvers.base import BasePDESolver
from pdewm.solvers.burgers_1d import Burgers1DSolver
from pdewm.solvers.contexts import PDEContext
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


def build_solver_from_context(context: PDEContext) -> BasePDESolver:
    grid_size = int(context.grid_descriptor.get("grid_size", 256))
    domain_length = float(context.grid_descriptor.get("domain_length", 1.0))
    if context.pde_id == "burgers_1d":
        return Burgers1DSolver(
            grid_size=grid_size,
            domain_length=domain_length,
            dealias=True,
        )
    if context.pde_id == "ks_1d":
        return KuramotoSivashinsky1DSolver(
            grid_size=grid_size,
            domain_length=domain_length,
            contour_points=16,
            dealias=True,
        )
    raise ValueError(f"Unsupported PDE context: {context.pde_id}")
