from __future__ import annotations

import numpy as np

from pdewm.solvers.burgers_1d import Burgers1DSolver
from pdewm.solvers.contexts import PDEContext


def test_burgers_solver_is_deterministic_for_fixed_input() -> None:
    solver = Burgers1DSolver(grid_size=64, domain_length=2.0 * np.pi)
    context = PDEContext(
        pde_id="burgers_1d",
        parameters={"viscosity": 0.05, "ic_amplitude": 1.0, "ic_bandwidth": 4},
        forcing_descriptor={"type": "sinusoidal", "amplitude": 0.0, "mode": 1},
        grid_descriptor={"grid_size": 64, "domain_length": 2.0 * np.pi},
        dt=0.002,
        dimension=1,
    )
    rng = np.random.default_rng(3)
    initial_state = solver.sample_initial_condition(rng, context)

    result_a = solver.simulate(initial_state, context, num_steps=12)
    result_b = solver.simulate(initial_state, context, num_steps=12)

    assert result_a.status == "ok"
    assert result_b.status == "ok"
    assert result_a.trajectory.shape == (13, 1, 64)
    assert np.allclose(result_a.trajectory, result_b.trajectory)
    assert np.isfinite(result_a.trajectory).all()

