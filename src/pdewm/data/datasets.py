from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zarr


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

