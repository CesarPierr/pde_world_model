from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


@dataclass(slots=True)
class TransitionArrayBundle:
    state: Any
    next_state: Any
    time_index: Any
    solver_runtime_sec: Any
    metadata: list[dict[str, Any]]
    manifest: dict[str, Any]


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
