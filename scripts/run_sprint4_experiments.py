from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_BASELINES = ("cnn_ar_1d", "unet_ar_1d", "fno_1d", "pod_mlp_1d")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="burgers_1d")
    parser.add_argument("--solver-config", default="burgers_1d")
    parser.add_argument("--dataset-root", default="data/generated_sprint4/burgers_1d_offline/sprint4")
    parser.add_argument("--data-output-dir", default="data/generated_sprint4")
    parser.add_argument("--output-root", default="artifacts/runs/sprint4_seq")
    parser.add_argument("--data-version", default="sprint4")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--baselines", nargs="+", default=list(DEFAULT_BASELINES))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="pde-world-model")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-group", default="")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

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
                f"data.output_dir={args.data_output_dir}",
                "data.splits.train.num_trajectories=12",
                "data.splits.val.num_trajectories=4",
                "data.splits.test.num_trajectories=4",
                "data.splits.parameter_ood.num_trajectories=4",
                f"data.num_steps={args.num_steps}",
                f"solver.grid_size={args.grid_size}",
            ]
        )

    results = []
    for baseline in args.baselines:
        baseline_output = output_root / baseline
        summary_path = baseline_output / "summary.json"
        if not summary_path.exists():
            command = [
                sys.executable,
                "scripts/train_baseline.py",
                "--model-config",
                baseline,
                f"train.dataset_root={dataset_root}",
                f"train.output_dir={baseline_output}",
                f"train.epochs={args.epochs}",
            ]
            command.extend(
                _wandb_overrides(
                    enabled=args.wandb,
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    mode=args.wandb_mode,
                    group=args.wandb_group or f"sprint4_{args.data_version}",
                    name=f"{args.data_version}_{baseline}",
                    tags=["sprint4_seq", args.data_config, baseline],
                )
            )
            _run(
                command
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results.append(summary)
        (output_root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    (output_root / "summary.md").write_text(_format_markdown_summary(results), encoding="utf-8")
    print(f"summary={output_root / 'summary.json'}")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _wandb_overrides(
    *,
    enabled: bool,
    project: str,
    entity: str,
    mode: str,
    group: str,
    name: str,
    tags: list[str],
) -> list[str]:
    if not enabled:
        return []
    overrides = [
        "logging.wandb.enabled=true",
        f"logging.wandb.project={project}",
        f"logging.wandb.mode={mode}",
        f"logging.wandb.group={group}",
        f"logging.wandb.name={name}",
        f"logging.wandb.tags=[{','.join(tags)}]",
    ]
    if entity:
        overrides.append(f"logging.wandb.entity={entity}")
    return overrides


def _format_markdown_summary(results: list[dict[str, object]]) -> str:
    lines = [
        "# Sprint 4 Sequential Experiments",
        "",
        "| Model | Best Val Loss | Val Rollout NRMSE | Test Rollout NRMSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results:
        trajectory_val_metrics = result["trajectory_val_metrics"]
        trajectory_test_metrics = result["trajectory_test_metrics"]
        assert isinstance(trajectory_val_metrics, dict)
        assert isinstance(trajectory_test_metrics, dict)
        lines.append(
            "| "
            f"{result['model_name']} | "
            f"{float(result['best_val_loss']):.6f} | "
            f"{float(trajectory_val_metrics['rollout_nrmse']['mean']):.6f} | "
            f"{float(trajectory_test_metrics['rollout_nrmse']['mean']):.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
