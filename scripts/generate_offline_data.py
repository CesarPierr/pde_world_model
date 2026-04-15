from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from pdewm.data.generation import build_manifest, sample_context, simulation_to_records
from pdewm.data.writer import OfflineDatasetWriter
from pdewm.solvers.factory import build_solver
from pdewm.utils.config import load_named_config
from pdewm.utils.git import repo_root
from pdewm.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="burgers_1d")
    parser.add_argument("--solver-config", default="burgers_1d")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf overrides in key=value form, e.g. solver.grid_size=64",
    )
    args = parser.parse_args()
    cfg = load_named_config(
        "generate_offline_data",
        overrides=list(args.overrides),
        defaults_overrides={"data": args.data_config, "solver": args.solver_config},
    )
    seed_everything(int(cfg.project.seed))
    solver = build_solver(cfg.solver)
    all_records = []
    trajectory_count = 0

    for split_name, split_cfg in cfg.data.splits.items():
        num_trajectories = int(split_cfg.num_trajectories)
        for local_index in range(num_trajectories):
            trajectory_seed = int(cfg.project.seed + trajectory_count * int(cfg.data.generation.seed_stride))
            trajectory_rng = np.random.default_rng(trajectory_seed)
            context = sample_context(cfg.solver, split_cfg, trajectory_rng)
            initial_state = solver.sample_initial_condition(trajectory_rng, context)
            result = solver.simulate(initial_state, context, num_steps=int(cfg.data.num_steps))
            trajectory_id = f"{cfg.data.dataset_name}_{split_name}_{trajectory_count:05d}"
            all_records.extend(
                simulation_to_records(
                    result=result,
                    context=context,
                    split_name=split_name,
                    trajectory_id=trajectory_id,
                    sample_origin=str(cfg.data.sample_origin),
                    seed=trajectory_seed,
                )
            )
            trajectory_count += 1

    manifest = build_manifest(
        dataset_name=str(cfg.data.dataset_name),
        dataset_version=str(cfg.data.dataset_version),
        solver=solver,
        solver_cfg=cfg.solver,
        data_cfg=cfg.data,
        samples=all_records,
        trajectory_count=trajectory_count,
    )

    dataset_root = (
        Path(repo_root())
        / str(cfg.data.output_dir)
        / str(cfg.data.dataset_name)
        / str(cfg.data.dataset_version)
    )
    writer = OfflineDatasetWriter(dataset_root)
    writer.write(all_records, manifest)

    print(OmegaConf.to_yaml(cfg))
    print(f"dataset_root={dataset_root}")
    print(f"samples={manifest.samples}")
    print(f"trajectories={manifest.trajectories}")


if __name__ == "__main__":
    main()
