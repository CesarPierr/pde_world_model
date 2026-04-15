from __future__ import annotations

import numpy as np

from pdewm.solvers.contexts import PDEContext
from pdewm.solvers.kuramoto_sivashinsky_1d import KuramotoSivashinsky1DSolver


def test_ks_solver_runs_without_nan_for_short_rollout() -> None:
    solver = KuramotoSivashinsky1DSolver(grid_size=64, domain_length=22.0, contour_points=16)
    context = PDEContext(
        pde_id="ks_1d",
        parameters={"domain_length": 22.0, "ic_amplitude": 0.8, "ic_bandwidth": 5},
        grid_descriptor={"grid_size": 64, "domain_length": 22.0},
        dt=0.05,
        dimension=1,
    )
    rng = np.random.default_rng(11)
    initial_state = solver.sample_initial_condition(rng, context)

    result = solver.simulate(initial_state, context, num_steps=10)

    assert result.status == "ok"
    assert result.trajectory.shape == (11, 1, 64)
    assert np.isfinite(result.trajectory).all()
    assert result.diagnostics["steps_completed"] == 10

