from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-epochs", type=int, default=120)
    parser.add_argument("--ae-epochs", type=int, default=120)
    parser.add_argument("--dynamics-epochs", type=int, default=120)
    parser.add_argument("--fine-tune-epochs", type=int, default=100)
    parser.add_argument("--online-solver-transitions", type=int, default=192)
    parser.add_argument("--transitions-per-round", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--run-burgers-baselines", action="store_true")
    parser.add_argument("--run-ks-baselines", action="store_true")
    parser.add_argument("--run-worldmodel-benchmark", action="store_true")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="pde-world-model")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-group", default="")
    args = parser.parse_args()

    run_all = not (
        args.run_burgers_baselines or args.run_ks_baselines or args.run_worldmodel_benchmark
    )
    wandb_args = _wandb_args(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        group=args.wandb_group or "long_campaign",
    )
    if args.run_burgers_baselines or run_all:
        _run(
            [
                sys.executable,
                "scripts/run_sprint4_experiments.py",
                "--prepare-data" if args.prepare_data else "",
                "--epochs",
                str(args.baseline_epochs),
                "--data-config",
                "burgers_1d",
                "--solver-config",
                "burgers_1d",
                "--dataset-root",
                "data/generated_long/burgers_1d_offline/long_burgers",
                "--data-output-dir",
                "data/generated_long",
                "--output-root",
                "artifacts/runs/long_burgers_baselines",
                "--data-version",
                "long_burgers",
                *wandb_args,
            ]
        )
    if args.run_ks_baselines or run_all:
        _run(
            [
                sys.executable,
                "scripts/run_sprint4_experiments.py",
                "--prepare-data" if args.prepare_data else "",
                "--epochs",
                str(args.baseline_epochs),
                "--data-config",
                "ks_1d",
                "--solver-config",
                "ks_1d",
                "--dataset-root",
                "data/generated_long/ks_1d_offline/long_ks",
                "--data-output-dir",
                "data/generated_long",
                "--output-root",
                "artifacts/runs/long_ks_baselines",
                "--data-version",
                "long_ks",
                *wandb_args,
            ]
        )
    if args.run_worldmodel_benchmark or run_all:
        _run(
            [
                sys.executable,
                "scripts/run_worldmodel_benchmark.py",
                "--prepare-data" if args.prepare_data else "",
                "--data-config",
                "burgers_1d",
                "--solver-config",
                "burgers_1d",
                "--dataset-root",
                "data/generated_long_worldmodel/burgers_1d_offline/long_worldmodel",
                "--output-root",
                "artifacts/runs/long_worldmodel_benchmark",
                "--data-version",
                "long_worldmodel",
                "--ae-epochs",
                str(args.ae_epochs),
                "--dynamics-epochs",
                str(args.dynamics_epochs),
                "--fine-tune-epochs",
                str(args.fine_tune_epochs),
                "--online-solver-transitions",
                str(args.online_solver_transitions),
                "--transitions-per-round",
                str(args.transitions_per_round),
                "--rollout-horizon",
                str(args.rollout_horizon),
                "--ensemble-size",
                str(args.ensemble_size),
                *wandb_args,
            ]
        )


def _run(command: list[str]) -> None:
    filtered = [part for part in command if part]
    subprocess.run(filtered, check=True)


def _wandb_args(
    *,
    enabled: bool,
    project: str,
    entity: str,
    mode: str,
    group: str,
) -> list[str]:
    if not enabled:
        return []
    args = [
        "--wandb",
        "--wandb-project",
        project,
        "--wandb-mode",
        mode,
        "--wandb-group",
        group,
    ]
    if entity:
        args.extend(["--wandb-entity", entity])
    return args


if __name__ == "__main__":
    main()
