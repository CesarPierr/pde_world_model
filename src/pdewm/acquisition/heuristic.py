from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pdewm.data.context_features import metadata_to_context_vector
from pdewm.training.dynamics import load_trained_dynamics


@dataclass(slots=True)
class CandidateState:
    state: np.ndarray
    metadata: dict[str, Any]
    latent_summary: np.ndarray
    uncertainty: float
    novelty: float
    risk: float
    score: float


def load_world_model_committee(
    checkpoint_paths: list[str],
    ae_checkpoint_path: str | None = None,
    *,
    device: str = "auto",
):
    members = []
    for checkpoint_path in checkpoint_paths:
        model, autoencoder, metadata = load_trained_dynamics(
            checkpoint_path,
            ae_checkpoint_path,
            device=device,
            autoencoder_role="acquisition",
        )
        members.append((model, autoencoder, metadata))
    return members


def build_memory_bank(
    records: list[Any],
    members,
    *,
    max_items: int | None = None,
) -> tuple[list[Any], np.ndarray]:
    selected_records = records[: max_items or len(records)]
    latents = []
    model, autoencoder, metadata = members[0]
    device = next(model.parameters()).device
    del metadata

    for record in selected_records:
        state = torch.from_numpy(record.state).unsqueeze(0).to(device)
        with torch.no_grad():
            latent = autoencoder.encode(state)
        latents.append(_latent_summary(latent[0]))
    return selected_records, np.stack(latents).astype(np.float32)


def propose_candidates(
    memory_records: list[Any],
    memory_latents: np.ndarray,
    members,
    *,
    pool_size: int,
    noise_std: float,
    alpha: float,
    beta: float,
    gamma: float,
    max_abs_factor: float,
    seed: int,
) -> list[CandidateState]:
    rng = np.random.default_rng(seed)
    state_stack = np.stack([record.state for record in memory_records], axis=0)
    global_scale = float(np.std(state_stack))
    amplitude_reference = float(np.max(np.abs(state_stack))) + 1.0e-6
    candidates: list[CandidateState] = []

    for _ in range(pool_size):
        index = int(rng.integers(0, len(memory_records)))
        base_record = memory_records[index]
        candidate_state = np.array(base_record.state, copy=True)
        noise = rng.normal(scale=noise_std * global_scale, size=candidate_state.shape)
        candidate_state = (candidate_state + noise).astype(np.float32)
        risk = max(0.0, float(np.max(np.abs(candidate_state))) - max_abs_factor * amplitude_reference)
        latent_summary = _encode_candidate_summary(candidate_state, members)
        metadata_dict = base_record.metadata.to_dict()
        uncertainty = _committee_uncertainty(candidate_state, metadata_dict, members)
        novelty = _novelty_score(latent_summary, memory_latents)
        score = alpha * uncertainty + beta * novelty - gamma * risk
        candidates.append(
            CandidateState(
                state=candidate_state,
                metadata=metadata_dict,
                latent_summary=latent_summary,
                uncertainty=uncertainty,
                novelty=novelty,
                risk=risk,
                score=score,
            )
        )
    return candidates


def select_diverse_candidates(
    candidates: list[CandidateState],
    *,
    top_m: int,
    batch_size: int,
    diversity_lambda: float,
) -> list[CandidateState]:
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)[:top_m]
    if not ranked:
        return []

    selected = [ranked.pop(0)]
    while ranked and len(selected) < batch_size:
        best_index = 0
        best_value = float("-inf")
        for index, candidate in enumerate(ranked):
            distance = min(
                float(np.linalg.norm(candidate.latent_summary - existing.latent_summary))
                for existing in selected
            )
            value = candidate.score + diversity_lambda * distance
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(ranked.pop(best_index))
    return selected


def rank_candidates(
    candidates: list[CandidateState],
    *,
    strategy: str,
    top_m: int,
    diversity_lambda: float,
    seed: int,
) -> list[CandidateState]:
    if strategy == "offline_only":
        return []
    if strategy == "random_states":
        rng = np.random.default_rng(seed)
        ranked = list(candidates)
        rng.shuffle(ranked)
        return ranked
    if strategy == "uncertainty_only":
        return sorted(candidates, key=lambda item: (item.uncertainty, -item.risk), reverse=True)
    if strategy == "diversity_only":
        return _greedy_diverse_ranking(
            candidates,
            base_score=lambda candidate: candidate.novelty - candidate.risk,
            diversity_lambda=diversity_lambda,
        )
    if strategy == "uncertainty_diversity":
        ranked = sorted(candidates, key=lambda item: item.uncertainty, reverse=True)[:top_m]
        return _greedy_diverse_ranking(
            ranked,
            base_score=lambda candidate: candidate.uncertainty,
            diversity_lambda=diversity_lambda,
        )
    if strategy == "ours":
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)[:top_m]
        return _greedy_diverse_ranking(
            ranked,
            base_score=lambda candidate: candidate.score,
            diversity_lambda=diversity_lambda,
        )
    raise ValueError(f"Unsupported acquisition strategy: {strategy}")


def _encode_candidate_summary(candidate_state: np.ndarray, members) -> np.ndarray:
    model, autoencoder, metadata = members[0]
    del model, metadata
    device = next(autoencoder.parameters()).device
    with torch.no_grad():
        latent = autoencoder.encode(torch.from_numpy(candidate_state).unsqueeze(0).to(device))
    return _latent_summary(latent[0])


def _committee_uncertainty(candidate_state: np.ndarray, metadata: dict[str, Any], members) -> float:
    predictions = []
    for model, autoencoder, model_metadata in members:
        device = next(model.parameters()).device
        state = torch.from_numpy(candidate_state).unsqueeze(0).to(device)
        context_features = metadata_to_context_vector(metadata, model_metadata["context_features"])
        context = torch.from_numpy(context_features).unsqueeze(0).to(device)
        pde_index = model_metadata["pde_to_index"][metadata["pde_id"]]
        pde_ids = torch.tensor([pde_index], device=device, dtype=torch.long)
        with torch.no_grad():
            latent = autoencoder.encode(state)
            next_latent = model(latent, pde_ids, context)
            prediction = autoencoder.decode(next_latent)[0].cpu().numpy()
        predictions.append(prediction)
    stacked = np.stack(predictions, axis=0)
    return float(np.mean(np.var(stacked, axis=0)))


def _novelty_score(latent_summary: np.ndarray, memory_latents: np.ndarray) -> float:
    distances = np.linalg.norm(memory_latents - latent_summary[None, :], axis=1)
    return float(np.min(distances))


def _latent_summary(latent: torch.Tensor) -> np.ndarray:
    return latent.detach().cpu().reshape(-1).numpy().astype(np.float32)


def _greedy_diverse_ranking(
    candidates: list[CandidateState],
    *,
    base_score,
    diversity_lambda: float,
) -> list[CandidateState]:
    pool = list(candidates)
    if not pool:
        return []

    pool.sort(key=base_score, reverse=True)
    ordered = [pool.pop(0)]
    while pool:
        best_index = 0
        best_value = float("-inf")
        for index, candidate in enumerate(pool):
            distance = min(
                float(np.linalg.norm(candidate.latent_summary - existing.latent_summary))
                for existing in ordered
            )
            value = float(base_score(candidate)) + diversity_lambda * distance
            if value > best_value:
                best_value = value
                best_index = index
        ordered.append(pool.pop(best_index))
    return ordered
