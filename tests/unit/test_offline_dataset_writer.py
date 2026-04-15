from __future__ import annotations

from pathlib import Path

import numpy as np

from pdewm.data.datasets import load_transition_bundle
from pdewm.data.schema import DatasetManifest, TransitionMetadata, TransitionRecord
from pdewm.data.writer import OfflineDatasetWriter


def test_offline_dataset_writer_persists_arrays_and_metadata(tmp_path: Path) -> None:
    samples = [
        TransitionRecord(
            state=np.zeros((1, 8), dtype=np.float32),
            next_state=np.ones((1, 8), dtype=np.float32),
            metadata=TransitionMetadata(
                time_index=0,
                dt=0.01,
                pde_id="burgers_1d",
                pde_params={"viscosity": 0.05},
                bc_descriptor={"type": "periodic"},
                forcing_descriptor={},
                grid_descriptor={"grid_size": 8},
                trajectory_id="traj_0001",
                split="train",
                sample_origin="offline",
                solver_status="ok",
                solver_runtime_sec=0.1,
                seed=7,
            ),
        )
    ]
    manifest = DatasetManifest(
        dataset_name="toy",
        dataset_version="v0",
        generator_git_hash="deadbeef",
        solver_version="0.1.0",
        parameter_space_signature={"viscosity_range": [0.01, 0.08]},
        seed_policy={"project_seed": 7},
        samples=1,
        trajectories=1,
        split_counts={"train": 1},
        created_at="2026-04-15T00:00:00+00:00",
        sample_shape=(1, 8),
    )

    writer = OfflineDatasetWriter(tmp_path / "toy" / "v0")
    dataset_root = writer.write(samples, manifest)
    bundle = load_transition_bundle(dataset_root)

    assert bundle.state.shape == (1, 1, 8)
    assert bundle.next_state.shape == (1, 1, 8)
    assert bundle.metadata[0]["trajectory_id"] == "traj_0001"
    assert bundle.manifest["dataset_name"] == "toy"
