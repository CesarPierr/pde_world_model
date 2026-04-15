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
    parser.add_argument("--online-iters", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--run-burgers-baselines", action="store_true")
    parser.add_argument("--run-ks-baselines", action="store_true")
    parser.add_argument("--run-worldmodel-active", action="store_true")
    parser.add_argument("--prepare-data", action="store_true")
    args = parser.parse_args()

    run_all = not (args.run_burgers_baselines or args.run_ks_baselines or args.run_worldmodel_active)
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
            ]
        )
    if args.run_worldmodel_active or run_all:
        _run(
            [
                sys.executable,
                "scripts/run_worldmodel_active_sampling.py",
                "--prepare-data" if args.prepare_data else "",
                "--prepare-ae",
                "--data-config",
                "burgers_1d",
                "--solver-config",
                "burgers_1d",
                "--dataset-root",
                "data/generated_long_worldmodel/burgers_1d_offline/long_worldmodel",
                "--output-root",
                "artifacts/runs/long_worldmodel_active",
                "--data-version",
                "long_worldmodel",
                "--ae-epochs",
                str(args.ae_epochs),
                "--dynamics-epochs",
                str(args.dynamics_epochs),
                "--fine-tune-epochs",
                str(args.fine_tune_epochs),
                "--online-iters",
                str(args.online_iters),
                "--ensemble-size",
                str(args.ensemble_size),
            ]
        )


def _run(command: list[str]) -> None:
    filtered = [part for part in command if part]
    subprocess.run(filtered, check=True)


if __name__ == "__main__":
    main()
