from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from pdewm.data.context_features import DEFAULT_CONTEXT_FEATURES, build_pde_index, metadata_to_context_vector
from pdewm.data.schema import TransitionMetadata, TransitionRecord


@dataclass(slots=True)
class TransitionArrayBundle:
    state: Any
    next_state: Any
    time_index: Any
    solver_runtime_sec: Any
    metadata: list[dict[str, Any]]
    manifest: dict[str, Any]


@dataclass(slots=True)
class TrajectorySample:
    trajectory_id: str
    split: str
    states: np.ndarray
    pde_ids: np.ndarray
    continuous_context: np.ndarray
    metadatas: list[dict[str, Any]]


def load_transition_bundle(dataset_root: str | Path) -> TransitionArrayBundle:
    dataset_root = Path(dataset_root)
    root = zarr.open_group(str(dataset_root / "transitions.zarr"), mode="r")
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    metadata = [
        json.loads(line)
        for line in (dataset_root / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return TransitionArrayBundle(
        state=root["state"],
        next_state=root["next_state"],
        time_index=root["time_index"],
        solver_runtime_sec=root["solver_runtime_sec"],
        metadata=metadata,
        manifest=manifest,
    )


def load_transition_records(dataset_root: str | Path) -> tuple[list[TransitionRecord], dict[str, Any]]:
    bundle = load_transition_bundle(dataset_root)
    records: list[TransitionRecord] = []
    for index, metadata in enumerate(bundle.metadata):
        record = TransitionRecord(
            state=np.asarray(bundle.state[index], dtype=np.float32),
            next_state=np.asarray(bundle.next_state[index], dtype=np.float32),
            metadata=TransitionMetadata(
                time_index=int(metadata["time_index"]),
                dt=float(metadata["dt"]),
                pde_id=str(metadata["pde_id"]),
                pde_params={str(key): float(value) for key, value in dict(metadata["pde_params"]).items()},
                bc_descriptor=dict(metadata["bc_descriptor"]),
                forcing_descriptor=dict(metadata["forcing_descriptor"]),
                grid_descriptor=dict(metadata["grid_descriptor"]),
                trajectory_id=str(metadata["trajectory_id"]),
                split=str(metadata["split"]),
                sample_origin=str(metadata["sample_origin"]),
                solver_status=str(metadata["solver_status"]),
                solver_runtime_sec=float(metadata["solver_runtime_sec"]),
                seed=int(metadata["seed"]),
            ),
        )
        records.append(record)
    return records, bundle.manifest


def load_trajectory_samples(
    dataset_root: str | Path,
    *,
    splits: tuple[str, ...] = ("val",),
    context_feature_keys: tuple[str, ...] = DEFAULT_CONTEXT_FEATURES,
) -> list[TrajectorySample]:
    bundle = load_transition_bundle(dataset_root)
    allowed = set(splits)
    pde_to_index = build_pde_index(metadata["pde_id"] for metadata in bundle.metadata)
    trajectory_to_entries: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for index, metadata in enumerate(bundle.metadata):
        if metadata["split"] in allowed:
            trajectory_to_entries[metadata["trajectory_id"]].append((index, metadata))

    samples: list[TrajectorySample] = []
    for trajectory_id, entries in trajectory_to_entries.items():
        ordered = sorted(entries, key=lambda item: item[1]["time_index"])
        indices = [index for index, _ in ordered]
        metadatas = [metadata for _, metadata in ordered]
        initial_state = np.asarray(bundle.state[indices[0]], dtype=np.float32)
        future_states = np.asarray(bundle.next_state[indices], dtype=np.float32)
        states = np.concatenate([initial_state[None, ...], future_states], axis=0)
        pde_ids = np.asarray(
            [pde_to_index[metadata["pde_id"]] for metadata in metadatas],
            dtype=np.int64,
        )
        continuous_context = np.stack(
            [metadata_to_context_vector(metadata, context_feature_keys) for metadata in metadatas],
            axis=0,
        ).astype(np.float32)
        samples.append(
            TrajectorySample(
                trajectory_id=trajectory_id,
                split=str(metadatas[0]["split"]),
                states=states,
                pde_ids=pde_ids,
                continuous_context=continuous_context,
                metadatas=metadatas,
            )
        )
    return sorted(samples, key=lambda sample: sample.trajectory_id)


class StateAutoencoderDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        dataset_root: str | Path,
        splits: tuple[str, ...] = ("train",),
        include_next_state: bool = True,
    ) -> None:
        bundle = load_transition_bundle(dataset_root)
        allowed = set(splits)
        indices = [idx for idx, meta in enumerate(bundle.metadata) if meta["split"] in allowed]
        states = np.asarray(bundle.state[indices], dtype=np.float32)

        if include_next_state:
            next_states = np.asarray(bundle.next_state[indices], dtype=np.float32)
            self.samples = np.concatenate([states, next_states], axis=0)
        else:
            self.samples = states

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.samples[index])


class TransitionWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        dataset_root: str | Path,
        *,
        splits: tuple[str, ...] = ("train",),
        rollout_horizon: int = 1,
        context_feature_keys: tuple[str, ...] = DEFAULT_CONTEXT_FEATURES,
    ) -> None:
        bundle = load_transition_bundle(dataset_root)
        allowed = set(splits)
        self.state_array = bundle.state
        self.next_state_array = bundle.next_state
        self.context_feature_keys = context_feature_keys
        self.rollout_horizon = int(rollout_horizon)
        self.pde_to_index = build_pde_index(metadata["pde_id"] for metadata in bundle.metadata)
        self.windows: list[list[int]] = []
        self.window_contexts: list[list[dict[str, Any]]] = []

        trajectory_to_entries: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, metadata in enumerate(bundle.metadata):
            if metadata["split"] in allowed:
                trajectory_to_entries[metadata["trajectory_id"]].append((index, metadata))

        for entries in trajectory_to_entries.values():
            ordered = sorted(entries, key=lambda item: item[1]["time_index"])
            indices = [index for index, _ in ordered]
            metadatas = [metadata for _, metadata in ordered]
            max_start = len(indices) - self.rollout_horizon + 1
            for start in range(max_start):
                self.windows.append(indices[start : start + self.rollout_horizon])
                self.window_contexts.append(metadatas[start : start + self.rollout_horizon])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window_indices = self.windows[index]
        window_metadata = self.window_contexts[index]
        initial_state = np.asarray(self.state_array[window_indices[0]], dtype=np.float32)
        future_states = np.asarray(self.next_state_array[window_indices], dtype=np.float32)
        pde_ids = np.asarray(
            [self.pde_to_index[metadata["pde_id"]] for metadata in window_metadata],
            dtype=np.int64,
        )
        continuous_context = np.stack(
            [metadata_to_context_vector(metadata, self.context_feature_keys) for metadata in window_metadata]
        ).astype(np.float32)
        return {
            "state": torch.from_numpy(initial_state),
            "future_states": torch.from_numpy(future_states),
            "pde_ids": torch.from_numpy(pde_ids),
            "continuous_context": torch.from_numpy(continuous_context),
        }
