from __future__ import annotations

import numpy as np

from pdewm.solvers.contexts import PDEContext
from pdewm.solvers.kdv_1d import KdV1DSolver


def test_kdv_solver_runs_without_nan_for_short_rollout() -> None:
    solver = KdV1DSolver(grid_size=64, domain_length=32.0)
    context = PDEContext(
        pde_id="kdv_1d",
        parameters={
            "domain_length": 32.0,
            "dispersion": 1.0,
            "ic_amplitude": 0.8,
            "ic_bandwidth": 4,
        },
        forcing_descriptor={"type": "sinusoidal", "amplitude": 0.02, "mode": 2, "phase": 0.0},
        grid_descriptor={"grid_size": 64, "domain_length": 32.0},
        dt=0.01,
        dimension=1,
    )
    rng = np.random.default_rng(17)
    initial_state = solver.sample_initial_condition(rng, context)

    result = solver.simulate(initial_state, context, num_steps=20)

    assert result.status == "ok"
    assert result.trajectory.shape == (21, 1, 64)
    assert np.isfinite(result.trajectory).all()
    assert result.diagnostics["steps_completed"] == 20
    assert result.diagnostics["total_substeps"] >= 20
