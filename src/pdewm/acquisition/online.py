from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pdewm.acquisition.heuristic import CandidateState
from pdewm.data.generation import simulation_to_records
from pdewm.data.schema import TransitionRecord
from pdewm.solvers.contexts import context_from_metadata
from pdewm.solvers.factory import build_solver_from_context


@dataclass(slots=True)
class AcquisitionRoundResult:
    new_records: list[TransitionRecord]
    attempted_candidates: int
    accepted_candidates: int
    transitions_requested: int
    transitions_acquired: int
    transitions_lost: int
    crash_count: int
    statuses: dict[str, int]
    batch_diversity_mean: float
    batch_diversity_min: float
    mean_uncertainty: float
    mean_novelty: float
    mean_risk: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted_candidates": self.attempted_candidates,
            "accepted_candidates": self.accepted_candidates,
            "transitions_requested": self.transitions_requested,
            "transitions_acquired": self.transitions_acquired,
            "transitions_lost": self.transitions_lost,
            "crash_count": self.crash_count,
            "statuses": dict(self.statuses),
            "batch_diversity_mean": self.batch_diversity_mean,
            "batch_diversity_min": self.batch_diversity_min,
            "mean_uncertainty": self.mean_uncertainty,
            "mean_novelty": self.mean_novelty,
            "mean_risk": self.mean_risk,
        }


def acquire_transition_budget(
    ordered_candidates: list[CandidateState],
    *,
    rollout_horizon: int,
    transition_budget: int,
    round_index: int,
    sample_origin: str = "online_active",
) -> AcquisitionRoundResult:
    all_records: list[TransitionRecord] = []
    attempted_candidates = 0
    accepted_candidates = 0
    transitions_requested = 0
    transitions_acquired = 0
    crash_count = 0
    statuses: dict[str, int] = {}
    selected_candidates: list[CandidateState] = []

    for candidate_index, candidate in enumerate(ordered_candidates):
        if transitions_acquired >= transition_budget:
            break
        remaining_budget = transition_budget - transitions_acquired
        rollout_steps = min(int(rollout_horizon), int(remaining_budget))
        if rollout_steps <= 0:
            break

        attempted_candidates += 1
        transitions_requested += rollout_steps
        context = context_from_metadata(candidate.metadata)
        solver = build_solver_from_context(context)
        result = solver.simulate(candidate.state, context, num_steps=rollout_steps)
        statuses[result.status] = statuses.get(result.status, 0) + 1
        if result.status != "ok":
            crash_count += 1

        records = simulation_to_records(
            result=result,
            context=context,
            split_name="train",
            trajectory_id=f"online_round_{round_index:02d}_{candidate_index:04d}",
            sample_origin=sample_origin,
            seed=int(candidate.metadata["seed"]),
        )
        if records:
            accepted_candidates += 1
            selected_candidates.append(candidate)
            all_records.extend(records)
            transitions_acquired += len(records)

    transitions_lost = transitions_requested - transitions_acquired
    diversity_mean, diversity_min = _candidate_diversity(selected_candidates)
    uncertainty_mean = _safe_mean([candidate.uncertainty for candidate in selected_candidates])
    novelty_mean = _safe_mean([candidate.novelty for candidate in selected_candidates])
    risk_mean = _safe_mean([candidate.risk for candidate in selected_candidates])

    return AcquisitionRoundResult(
        new_records=all_records,
        attempted_candidates=attempted_candidates,
        accepted_candidates=accepted_candidates,
        transitions_requested=transitions_requested,
        transitions_acquired=transitions_acquired,
        transitions_lost=transitions_lost,
        crash_count=crash_count,
        statuses=statuses,
        batch_diversity_mean=diversity_mean,
        batch_diversity_min=diversity_min,
        mean_uncertainty=uncertainty_mean,
        mean_novelty=novelty_mean,
        mean_risk=risk_mean,
    )


def _candidate_diversity(candidates: list[CandidateState]) -> tuple[float, float]:
    if len(candidates) < 2:
        return 0.0, 0.0
    distances = []
    for index, candidate in enumerate(candidates):
        for other in candidates[index + 1 :]:
            distances.append(float(np.linalg.norm(candidate.latent_summary - other.latent_summary)))
    if not distances:
        return 0.0, 0.0
    array = np.asarray(distances, dtype=np.float64)
    return float(array.mean()), float(array.min())


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))
