from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PDEContext:
    pde_id: str
    parameters: dict[str, float]
    boundary_conditions: dict[str, Any] = field(default_factory=lambda: {"type": "periodic"})
    forcing_descriptor: dict[str, Any] = field(default_factory=dict)
    grid_descriptor: dict[str, Any] = field(default_factory=dict)
    dt: float = 0.01
    dimension: int = 1

    def to_metadata(self) -> dict[str, Any]:
        return {
            "pde_id": self.pde_id,
            "parameters": dict(self.parameters),
            "bc_descriptor": dict(self.boundary_conditions),
            "forcing_descriptor": dict(self.forcing_descriptor),
            "grid_descriptor": dict(self.grid_descriptor),
            "dt": self.dt,
            "dimension": self.dimension,
        }

