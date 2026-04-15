from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pdewm.solvers.contexts import PDEContext


@dataclass(slots=True)
class SimulationResult:
    trajectory: np.ndarray
    status: str
    runtime_sec: float
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class BasePDESolver(ABC):
    solver_name: str
    solver_version: str = "0.1.0"

    @abstractmethod
    def sample_initial_condition(self, rng: np.random.Generator, context: PDEContext) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def simulate(
        self,
        initial_state: np.ndarray,
        context: PDEContext,
        num_steps: int,
        dt: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> SimulationResult:
        raise NotImplementedError

