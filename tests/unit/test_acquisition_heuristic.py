from __future__ import annotations

import numpy as np

from pdewm.acquisition.heuristic import CandidateState, select_diverse_candidates
from pdewm.solvers.contexts import PDEContext
from pdewm.solvers.factory import build_solver_from_context


def test_select_diverse_candidates_returns_requested_batch() -> None:
    candidates = [
        CandidateState(
            state=np.zeros((1, 8), dtype=np.float32),
            metadata={"pde_id": "burgers_1d"},
            latent_summary=np.asarray([float(index), 0.0], dtype=np.float32),
            uncertainty=1.0 - 0.1 * index,
            novelty=0.2 * index,
            risk=0.0,
            score=1.0 - 0.05 * index,
        )
        for index in range(6)
    ]

    selected = select_diverse_candidates(candidates, top_m=5, batch_size=3, diversity_lambda=0.5)

    assert len(selected) == 3


def test_build_solver_from_context_uses_context_grid() -> None:
    context = PDEContext(
        pde_id="burgers_1d",
        parameters={"viscosity": 0.05},
        grid_descriptor={"grid_size": 64, "domain_length": 2.0 * np.pi},
        dt=0.01,
        dimension=1,
    )

    solver = build_solver_from_context(context)

    assert solver.grid_size == 64
