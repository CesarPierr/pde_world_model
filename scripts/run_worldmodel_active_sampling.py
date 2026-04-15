from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from pdewm.acquisition.heuristic import (
    build_memory_bank,
    load_world_model_committee,
    propose_candidates,
    select_diverse_candidates,
)
from pdewm.data.datasets import load_transition_records
from pdewm.data.schema import DatasetManifest
from pdewm.data.writer import OfflineDatasetWriter
from pdewm.solvers.contexts import context_from_metadata
from pdewm.solvers.factory import build_solver_from_context
from pdewm.utils.git import get_git_commit_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="burgers_1d")
    parser.add_argument("--solver-config", default="burgers_1d")
    parser.add_argument("--dataset-root", default="data/generated_worldmodel/burgers_1d_offline/worldmodel")
    parser.add_argument("--output-root", default="artifacts/runs/worldmodel_active")
    parser.add_argument("--data-version", default="worldmodel")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--prepare-ae", action="store_true")
    parser.add_argument("--ae-epochs", type=int, default=120)
    parser.add_argument("--dynamics-epochs", type=int, default=120)
    parser.add_argument("--fine-tune-epochs", type=int, default=100)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--online-iters", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--acquisition-rollout-steps", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = dataset_root.parents[1] if len(dataset_root.parts) >= 3 else Path("data/generated_worldmodel")

    if args.prepare_data or not dataset_root.exists():
        _run(
            [
                sys.executable,
                "scripts/generate_offline_data.py",
                "--data-config",
                args.data_config,
                "--solver-config",
                args.solver_config,
                f"data.dataset_version={args.data_version}",
                f"data.output_dir={data_root.as_posix()}",
                "data.splits.train.num_trajectories=16",
                "data.splits.val.num_trajectories=4",
                "data.splits.test.num_trajectories=4",
                "data.splits.parameter_ood.num_trajectories=4",
                f"data.num_steps={args.num_steps}",
                f"solver.grid_size={args.grid_size}",
            ]
        )

    ae_dir = output_root / "autoencoder"
    ae_checkpoint = ae_dir / "best.pt"
    if args.prepare_ae or not ae_checkpoint.exists():
        _run(
            [
                sys.executable,
                "scripts/train_autoencoder.py",
                f"train.dataset_root={dataset_root}",
                f"train.output_dir={ae_dir}",
                f"train.epochs={args.ae_epochs}",
                "train.batch_size=16",
                "model.base_channels=16",
                "model.latent_channels=32",
                "model.channel_multipliers=[1,2,4]",
            ]
        )

    current_dataset_root = dataset_root
    campaign_summary: list[dict[str, object]] = []

    for iteration in range(args.online_iters + 1):
        iteration_dir = output_root / f"iter_{iteration:02d}"
        committee_dir = iteration_dir / "dynamics_committee"
        committee_dir.mkdir(parents=True, exist_ok=True)
        committee_checkpoints = []
        ensemble_summaries = []
        epochs = args.dynamics_epochs if iteration == 0 else args.fine_tune_epochs

        for member_index in range(args.ensemble_size):
            member_dir = committee_dir / f"member_{member_index:02d}"
            summary_path = member_dir / "summary.json"
            if not summary_path.exists():
                _run(
                    [
                        sys.executable,
                        "scripts/train_dynamics.py",
                        f"train.dataset_root={current_dataset_root}",
                        f"train.ae_checkpoint={ae_checkpoint}",
                        f"train.output_dir={member_dir}",
                        f"train.epochs={epochs}",
                        f"project.seed={args.seed + member_index}",
                        "train.batch_size=8",
                        "model.hidden_channels=32",
                        "model.context_hidden_dim=32",
                        "model.context_output_dim=32",
                        "model.num_blocks=2",
                    ]
                )
            committee_checkpoints.append(str(member_dir / "best.pt"))
            ensemble_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

        iteration_summary = {
            "iteration": iteration,
            "dataset_root": str(current_dataset_root),
            "ensemble": ensemble_summaries,
        }
        campaign_summary.append(iteration_summary)
        _write_campaign_summary(output_root, campaign_summary)

        if iteration == args.online_iters:
            break

        next_dataset_root = output_root / "datasets" / f"online_iter_{iteration + 1:02d}"
        if not next_dataset_root.exists():
            members = load_world_model_committee(
                committee_checkpoints,
                str(ae_checkpoint),
                device="cpu",
            )
            records, manifest_dict = load_transition_records(current_dataset_root)
            train_records = [record for record in records if record.metadata.split == "train"]
            memory_records, memory_latents = build_memory_bank(train_records, members)
            candidates = propose_candidates(
                memory_records,
                memory_latents,
                members,
                pool_size=args.pool_size,
                noise_std=args.noise_std,
                alpha=1.0,
                beta=0.2,
                gamma=0.5,
                max_abs_factor=2.0,
                seed=args.seed + iteration,
            )
            selected = select_diverse_candidates(
                candidates,
                top_m=args.top_m,
                batch_size=args.batch_size,
                diversity_lambda=0.2,
            )
            new_records = _simulate_selected_candidates(
                selected,
                rollout_steps=args.acquisition_rollout_steps,
                iteration=iteration + 1,
            )
            _write_extended_dataset(
                previous_records=records,
                previous_manifest=manifest_dict,
                new_records=new_records,
                dataset_root=next_dataset_root,
                dataset_version=f"{args.data_version}_online_iter_{iteration + 1:02d}",
            )
            iteration_summary["acquisition"] = {
                "selected_candidates": len(selected),
                "new_samples": len(new_records),
                "next_dataset_root": str(next_dataset_root),
            }
            _write_campaign_summary(output_root, campaign_summary)

        current_dataset_root = next_dataset_root


def _simulate_selected_candidates(
    selected,
    *,
    rollout_steps: int,
    iteration: int,
):
    from pdewm.data.generation import simulation_to_records

    all_records = []
    for candidate_index, candidate in enumerate(selected):
        context = context_from_metadata(candidate.metadata)
        solver = build_solver_from_context(context)
        result = solver.simulate(candidate.state, context, num_steps=rollout_steps)
        trajectory_id = f"online_iter_{iteration:02d}_{candidate_index:04d}"
        all_records.extend(
            simulation_to_records(
                result=result,
                context=context,
                split_name="train",
                trajectory_id=trajectory_id,
                sample_origin="online_active",
                seed=int(candidate.metadata["seed"]),
            )
        )
    return all_records


def _write_extended_dataset(
    *,
    previous_records,
    previous_manifest: dict[str, object],
    new_records,
    dataset_root: Path,
    dataset_version: str,
) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    all_records = [*previous_records, *new_records]
    split_counts: dict[str, int] = {}
    trajectory_ids = set()
    for record in all_records:
        split_counts[record.metadata.split] = split_counts.get(record.metadata.split, 0) + 1
        trajectory_ids.add(record.metadata.trajectory_id)

    manifest = DatasetManifest(
        dataset_name=str(previous_manifest["dataset_name"]),
        dataset_version=dataset_version,
        generator_git_hash=get_git_commit_hash(),
        solver_version=str(previous_manifest["solver_version"]),
        parameter_space_signature=dict(previous_manifest["parameter_space_signature"]),
        seed_policy=dict(previous_manifest["seed_policy"]),
        samples=len(all_records),
        trajectories=len(trajectory_ids),
        split_counts=split_counts,
        created_at=datetime.now(UTC).isoformat(),
        sample_shape=tuple(previous_manifest["sample_shape"]),
        extras=dict(previous_manifest.get("extras", {})),
    )
    OfflineDatasetWriter(dataset_root).write(all_records, manifest)


def _write_campaign_summary(output_root: Path, summary: list[dict[str, object]]) -> None:
    (output_root / "campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
