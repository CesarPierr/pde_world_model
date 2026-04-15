from __future__ import annotations

import numpy as np
import torch

from pdewm.acquisition.heuristic import CandidateState
from pdewm.acquisition.online import acquire_transition_budget
from pdewm.training.dynamics import _update_ema


def test_acquire_transition_budget_respects_requested_cap() -> None:
    candidates = [
        CandidateState(
            state=np.zeros((1, 32), dtype=np.float32),
            metadata={
                "time_index": 0,
                "dt": 0.0025,
                "pde_id": "burgers_1d",
                "pde_params": {
                    "viscosity": 0.02,
                    "ic_amplitude": 1.0,
                    "ic_bandwidth": 4.0,
                },
                "bc_descriptor": {},
                "forcing_descriptor": {
                    "type": "sinusoidal",
                    "amplitude": 0.0,
                    "mode": 1,
                    "phase": 0.0,
                },
                "grid_descriptor": {"grid_size": 32, "domain_length": float(2.0 * np.pi)},
                "trajectory_id": "seed",
                "split": "train",
                "sample_origin": "offline",
                "solver_status": "ok",
                "solver_runtime_sec": 0.0,
                "seed": 7,
            },
            latent_summary=np.zeros(8, dtype=np.float32),
            uncertainty=1.0,
            novelty=1.0,
            risk=0.0,
            score=1.0,
        )
        for _ in range(3)
    ]
    result = acquire_transition_budget(
        candidates,
        rollout_horizon=3,
        transition_budget=5,
        round_index=1,
    )
    assert result.transitions_acquired == 5
    assert len(result.new_records) == 5
    assert result.transitions_requested == 5


def test_update_ema_moves_teacher_toward_student() -> None:
    student = torch.nn.Linear(2, 2, bias=False)
    teacher = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
        teacher.weight.fill_(0.0)
    _update_ema(student, teacher, decay=0.5)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 1.0))
