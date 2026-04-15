from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr

from pdewm.data.schema import DatasetManifest, TransitionRecord


class OfflineDatasetWriter:
    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)

    def write(self, samples: Sequence[TransitionRecord], manifest: DatasetManifest) -> Path:
        if not samples:
            raise ValueError("Cannot write an empty dataset.")

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        zarr_path = self.dataset_root / "transitions.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")

        state_array = np.stack([sample.state for sample in samples]).astype(np.float32)
        next_state_array = np.stack([sample.next_state for sample in samples]).astype(np.float32)
        time_index_array = np.asarray([sample.metadata.time_index for sample in samples], dtype=np.int32)
        runtime_array = np.asarray(
            [sample.metadata.solver_runtime_sec for sample in samples], dtype=np.float32
        )

        chunk_size = min(256, max(1, len(samples)))
        root.create_array("state", data=state_array, chunks=(chunk_size, *state_array.shape[1:]))
        root.create_array(
            "next_state",
            data=next_state_array,
            chunks=(chunk_size, *next_state_array.shape[1:]),
        )
        root.create_array("time_index", data=time_index_array, chunks=(chunk_size,))
        root.create_array("solver_runtime_sec", data=runtime_array, chunks=(chunk_size,))

        manifest_path = self.dataset_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        index_path = self.dataset_root / "index.jsonl"
        with index_path.open("w", encoding="utf-8") as handle:
            for record in samples:
                handle.write(json.dumps(record.metadata.to_dict()))
                handle.write("\n")

        return self.dataset_root
