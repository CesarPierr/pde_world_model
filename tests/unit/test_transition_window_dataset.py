from __future__ import annotations

from pathlib import Path

import numpy as np

from pdewm.data.datasets import TransitionWindowDataset
from pdewm.data.schema import DatasetManifest, TransitionMetadata, TransitionRecord
from pdewm.data.writer import OfflineDatasetWriter


def test_transition_window_dataset_builds_rollout_windows(tmp_path: Path) -> None:
    samples: list[TransitionRecord] = []
    for time_index in range(3):
        samples.append(
            TransitionRecord(
                state=np.full((1, 8), fill_value=time_index, dtype=np.float32),
                next_state=np.full((1, 8), fill_value=time_index + 1, dtype=np.float32),
                metadata=TransitionMetadata(
                    time_index=time_index,
                    dt=0.01,
                    pde_id="burgers_1d",
                    pde_params={"viscosity": 0.05, "ic_amplitude": 1.0, "ic_bandwidth": 3},
                    bc_descriptor={"type": "periodic"},
                    forcing_descriptor={"amplitude": 0.1, "mode": 2},
                    grid_descriptor={"grid_size": 8, "domain_length": 6.28},
                    trajectory_id="traj_0001",
                    split="train",
                    sample_origin="offline",
                    solver_status="ok",
                    solver_runtime_sec=0.1,
                    seed=7,
                ),
            )
        )

    manifest = DatasetManifest(
        dataset_name="toy_rollout",
        dataset_version="v0",
        generator_git_hash="deadbeef",
        solver_version="0.1.0",
        parameter_space_signature={"viscosity_range": [0.01, 0.08]},
        seed_policy={"project_seed": 7},
        samples=3,
        trajectories=1,
        split_counts={"train": 3},
        created_at="2026-04-15T00:00:00+00:00",
        sample_shape=(1, 8),
    )

    dataset_root = OfflineDatasetWriter(tmp_path / "toy_rollout" / "v0").write(samples, manifest)
    dataset = TransitionWindowDataset(dataset_root, rollout_horizon=2)
    item = dataset[0]

    assert len(dataset) == 2
    assert item["state"].shape == (1, 8)
    assert item["future_states"].shape == (2, 1, 8)
    assert item["pde_ids"].shape == (2,)
    assert item["continuous_context"].shape[-1] == 8

