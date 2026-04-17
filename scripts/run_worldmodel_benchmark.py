from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from pdewm.acquisition.heuristic import (
    build_memory_bank,
    load_world_model_committee,
    propose_candidates,
    rank_candidates,
)
from pdewm.acquisition.online import acquire_transition_budget
from pdewm.data.datasets import load_transition_records
from pdewm.data.schema import DatasetManifest, TransitionRecord
from pdewm.data.writer import OfflineDatasetWriter
from pdewm.utils.git import get_git_commit_hash
from pdewm.utils.device import resolve_device
from pdewm.utils.wandb import compose_wandb_group, compose_wandb_name


DEFAULT_REGIMES = ("frozen", "joint_no_ema", "joint_ema")
DEFAULT_STRATEGIES = (
    "offline_only",
    "random_states",
    "uncertainty_only",
    "diversity_only",
    "uncertainty_diversity",
    "ours",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="burgers_1d")
    parser.add_argument("--solver-config", default="burgers_1d")
    parser.add_argument("--dataset-root", default="data/generated_worldmodel/burgers_1d_offline/worldmodel")
    parser.add_argument("--output-root", default="artifacts/runs/worldmodel_benchmark")
    parser.add_argument("--data-version", default="worldmodel_protocol")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--prepare-ae", action="store_true")
    parser.add_argument("--ae-epochs", type=int, default=120)
    parser.add_argument("--dynamics-epochs", type=int, default=120)
    parser.add_argument("--fine-tune-epochs", type=int, default=100)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument("--online-solver-transitions", type=int, default=128)
    parser.add_argument("--transitions-per-round", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--diversity-lambda", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--selected-regime", default="auto")
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="pde-world-model")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-group", default="")
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
                "data.splits.test.num_trajectories=4",
                "data.splits.parameter_ood.num_trajectories=4",
                f"data.num_steps={args.num_steps}",
                f"solver.grid_size={args.grid_size}",
            ]
        )

    ae_dir = output_root / "warmup_autoencoder"
    ae_checkpoint = ae_dir / "last.pt"
    if args.prepare_ae or not ae_checkpoint.exists():
        ae_command = [
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
        ae_command.extend(
            _wandb_overrides(
                enabled=args.wandb,
                project=args.wandb_project,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                group=_resolve_wandb_group(
                    args.wandb_group,
                    "worldmodel-benchmark",
                    args.data_version,
                    "warmup-autoencoder",
                ),
                name=compose_wandb_name("worldmodel-benchmark", args.data_version, "warmup autoencoder"),
                tags=["worldmodel_benchmark", args.data_config, "warmup_autoencoder"],
            )
        )
        _run(ae_command)

    _, manifest = load_transition_records(dataset_root)
    offline_transitions = int(manifest["samples"])

    regime_root = output_root / "regime_ablation"
    regime_root.mkdir(parents=True, exist_ok=True)
    regime_results = {}
    for regime in args.regimes:
        regime_results[regime] = _run_regime_ablation(
            dataset_root=dataset_root,
            output_dir=regime_root / regime,
            ae_checkpoint=ae_checkpoint,
            regime=regime,
            epochs=args.dynamics_epochs,
            ensemble_size=args.ensemble_size,
            seed=args.seed,
            wandb_args=args,
        )
    regime_summary = {
        "regimes": regime_results,
        "selected_regime": _select_regime(regime_results, preferred=args.selected_regime),
    }
    (regime_root / "summary.json").write_text(json.dumps(regime_summary, indent=2), encoding="utf-8")
    (regime_root / "summary.md").write_text(_format_regime_summary(regime_summary), encoding="utf-8")

    selected_regime = str(regime_summary["selected_regime"])
    selected_regime_result = regime_results[selected_regime]

    acquisition_root = output_root / "acquisition_benchmark"
    acquisition_root.mkdir(parents=True, exist_ok=True)
    strategy_results = {}
    for strategy in args.strategies:
        strategy_results[strategy] = _run_strategy_benchmark(
            strategy=strategy,
            dataset_root=dataset_root,
            output_dir=acquisition_root / strategy,
            ae_checkpoint=ae_checkpoint,
            offline_transitions=offline_transitions,
            selected_regime=selected_regime,
            selected_regime_result=selected_regime_result,
            online_solver_transitions=args.online_solver_transitions,
            transitions_per_round=args.transitions_per_round,
            rollout_horizon=args.rollout_horizon,
            pool_size=args.pool_size,
            top_m=args.top_m,
            noise_std=args.noise_std,
            diversity_lambda=args.diversity_lambda,
            fine_tune_epochs=args.fine_tune_epochs,
            seed=args.seed,
            wandb_args=args,
        )

    benchmark_summary = {
        "dataset_root": str(dataset_root),
        "offline_transitions": offline_transitions,
        "online_solver_transition_budget": int(args.online_solver_transitions),
        "selected_regime": selected_regime,
        "regime_summary": regime_summary,
        "strategy_results": strategy_results,
    }
    (output_root / "benchmark_summary.json").write_text(
        json.dumps(benchmark_summary, indent=2),
        encoding="utf-8",
    )
    (output_root / "benchmark_summary.md").write_text(
        _format_benchmark_summary(benchmark_summary),
        encoding="utf-8",
    )
    _plot_strategy_curves(strategy_results, output_root)


def _run_regime_ablation(
    *,
    dataset_root: Path,
    output_dir: Path,
    ae_checkpoint: Path,
    regime: str,
    epochs: int,
    ensemble_size: int,
    seed: int,
    wandb_args,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    member_summaries = []
    member_checkpoints = []
    for member_index in range(ensemble_size):
        member_dir = output_dir / f"member_{member_index:02d}"
        member_summary_path = member_dir / "summary.json"
        if not member_summary_path.exists():
            command = [
                sys.executable,
                "scripts/train_dynamics.py",
                f"train.dataset_root={dataset_root}",
                f"train.ae_checkpoint={ae_checkpoint}",
                f"train.output_dir={member_dir}",
                f"train.epochs={epochs}",
                f"project.seed={seed + member_index}",
                f"train.regime={regime}",
                "train.batch_size=8",
                "train.ae_loss_scale=1.0",
                "train.ema.decay=0.995",
                "model.hidden_channels=32",
                "model.context_hidden_dim=32",
                "model.context_output_dim=32",
                "model.num_blocks=2",
            ]
            command.extend(
                _wandb_overrides(
                    enabled=wandb_args.wandb,
                    project=wandb_args.wandb_project,
                    entity=wandb_args.wandb_entity,
                    mode=wandb_args.wandb_mode,
                    group=_resolve_wandb_group(
                        wandb_args.wandb_group,
                        "worldmodel-benchmark",
                        wandb_args.data_version,
                        "regime-ablation",
                        regime,
                    ),
                    name=compose_wandb_name(
                        "worldmodel-benchmark",
                        "regime ablation",
                        regime,
                        f"member {member_index:02d}",
                        f"seed {seed + member_index}",
                    ),
                    tags=["worldmodel_benchmark", "regime_ablation", regime],
                )
            )
            _run(command)
        member_summaries.append(json.loads(member_summary_path.read_text(encoding="utf-8")))
        member_checkpoints.append(str(member_dir / "last.pt"))

    result = {
        "regime": regime,
        "ensemble_size": ensemble_size,
        "aggregate": _aggregate_ensemble_summaries(member_summaries),
        "members": member_summaries,
        "member_checkpoints": member_checkpoints,
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _run_strategy_benchmark(
    *,
    strategy: str,
    dataset_root: Path,
    output_dir: Path,
    ae_checkpoint: Path,
    offline_transitions: int,
    selected_regime: str,
    selected_regime_result: dict[str, Any],
    online_solver_transitions: int,
    transitions_per_round: int,
    rollout_horizon: int,
    pool_size: int,
    top_m: int,
    noise_std: float,
    diversity_lambda: float,
    fine_tune_epochs: int,
    seed: int,
    wandb_args,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    curve = [
        _curve_point(
            aggregate=selected_regime_result["aggregate"],
            online_solver_transitions=0,
            offline_transitions=offline_transitions,
            round_index=0,
        )
    ]
    result = {
        "strategy": strategy,
        "selected_regime": selected_regime,
        "offline_transitions": offline_transitions,
        "online_solver_transitions": 0,
        "curve": curve,
        "rounds": [],
    }

    if strategy == "offline_only":
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    current_dataset_root = dataset_root
    current_committee_checkpoints = list(selected_regime_result["member_checkpoints"])
    cumulative_online_transitions = 0
    round_index = 0

    while cumulative_online_transitions < online_solver_transitions:
        remaining_budget = online_solver_transitions - cumulative_online_transitions
        round_budget = min(transitions_per_round, remaining_budget)
        round_dir = output_dir / f"round_{round_index + 1:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        members = load_world_model_committee(
            current_committee_checkpoints,
            str(ae_checkpoint),
            device=str(resolve_device("auto")),
        )
        records, manifest = load_transition_records(current_dataset_root)
        train_records = [record for record in records if record.metadata.split == "train"]
        memory_records, memory_latents = build_memory_bank(train_records, members)
        candidates = propose_candidates(
            memory_records,
            memory_latents,
            members,
            pool_size=pool_size,
            noise_std=noise_std,
            alpha=1.0,
            beta=0.2,
            gamma=0.5,
            max_abs_factor=2.0,
            seed=seed + round_index,
        )
        ordered_candidates = rank_candidates(
            candidates,
            strategy=strategy,
            top_m=top_m,
            diversity_lambda=diversity_lambda,
            seed=seed + round_index,
        )
        acquisition_result = acquire_transition_budget(
            ordered_candidates,
            rollout_horizon=rollout_horizon,
            transition_budget=round_budget,
            round_index=round_index + 1,
            sample_origin=f"online_{strategy}",
        )
        if not acquisition_result.new_records:
            break

        next_dataset_root = output_dir / "datasets" / f"round_{round_index + 1:02d}"
        _write_extended_dataset(
            previous_records=records,
            previous_manifest=manifest,
            new_records=acquisition_result.new_records,
            dataset_root=next_dataset_root,
            dataset_version=f"{dataset_root.name}_{strategy}_round_{round_index + 1:02d}",
            offline_transitions=offline_transitions,
            online_solver_transitions=cumulative_online_transitions + acquisition_result.transitions_acquired,
        )

        committee_result = _fine_tune_committee(
            dataset_root=next_dataset_root,
            output_dir=round_dir / "dynamics_committee",
            ae_checkpoint=ae_checkpoint,
            regime=selected_regime,
            epochs=fine_tune_epochs,
            ensemble_size=len(current_committee_checkpoints),
            seed=seed + 1000 * (round_index + 1),
            resume_checkpoints=current_committee_checkpoints,
            wandb_args=wandb_args,
            strategy=strategy,
            round_index=round_index + 1,
        )

        cumulative_online_transitions += acquisition_result.transitions_acquired
        round_summary = {
            "round_index": round_index + 1,
            "dataset_root": str(next_dataset_root),
            "acquisition": acquisition_result.to_dict(),
            "aggregate": committee_result["aggregate"],
            "members": committee_result["members"],
            "online_solver_transitions": cumulative_online_transitions,
            "total_transitions": offline_transitions + cumulative_online_transitions,
        }
        result["rounds"].append(round_summary)
        result["curve"].append(
            _curve_point(
                aggregate=committee_result["aggregate"],
                online_solver_transitions=cumulative_online_transitions,
                offline_transitions=offline_transitions,
                round_index=round_index + 1,
            )
        )
        result["online_solver_transitions"] = cumulative_online_transitions
        current_dataset_root = next_dataset_root
        current_committee_checkpoints = list(committee_result["member_checkpoints"])
        round_index += 1

    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _fine_tune_committee(
    *,
    dataset_root: Path,
    output_dir: Path,
    ae_checkpoint: Path,
    regime: str,
    epochs: int,
    ensemble_size: int,
    seed: int,
    resume_checkpoints: list[str],
    wandb_args,
    strategy: str,
    round_index: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    member_summaries = []
    member_checkpoints = []

    for member_index in range(ensemble_size):
        member_dir = output_dir / f"member_{member_index:02d}"
        member_summary_path = member_dir / "summary.json"
        if not member_summary_path.exists():
            command = [
                sys.executable,
                "scripts/train_dynamics.py",
                f"train.dataset_root={dataset_root}",
                f"train.ae_checkpoint={ae_checkpoint}",
                f"train.output_dir={member_dir}",
                f"train.epochs={epochs}",
                f"project.seed={seed + member_index}",
                f"train.regime={regime}",
                f"train.resume_checkpoint={resume_checkpoints[member_index]}",
                "train.batch_size=8",
                "train.ae_loss_scale=1.0",
                "train.ema.decay=0.995",
                "model.hidden_channels=32",
                "model.context_hidden_dim=32",
                "model.context_output_dim=32",
                "model.num_blocks=2",
            ]
            command.extend(
                _wandb_overrides(
                    enabled=wandb_args.wandb,
                    project=wandb_args.wandb_project,
                    entity=wandb_args.wandb_entity,
                    mode=wandb_args.wandb_mode,
                    group=_resolve_wandb_group(
                        wandb_args.wandb_group,
                        "worldmodel-benchmark",
                        wandb_args.data_version,
                        "acquisition",
                        strategy,
                        regime,
                    ),
                    name=compose_wandb_name(
                        "worldmodel-benchmark",
                        "acquisition",
                        strategy,
                        f"round {round_index:02d}",
                        f"member {member_index:02d}",
                        f"seed {seed + member_index}",
                    ),
                    tags=[
                        "worldmodel_benchmark",
                        "acquisition_benchmark",
                        strategy,
                        regime,
                    ],
                )
            )
            _run(command)
        member_summaries.append(json.loads(member_summary_path.read_text(encoding="utf-8")))
        member_checkpoints.append(str(member_dir / "last.pt"))

    return {
        "aggregate": _aggregate_ensemble_summaries(member_summaries),
        "members": member_summaries,
        "member_checkpoints": member_checkpoints,
    }


def _aggregate_ensemble_summaries(member_summaries: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "final_eval_loss_mean": _mean_nested(member_summaries, ("final_eval_loss",)),
        "ae_eval_loss_mean": _mean_nested(member_summaries, ("ae_eval_metrics", "loss")),
        "trajectory_eval_one_step_rmse_mean": _mean_nested(
            member_summaries,
            ("trajectory_eval_metrics", "one_step_rmse", "mean"),
        ),
        "trajectory_eval_one_step_nrmse_mean": _mean_nested(
            member_summaries,
            ("trajectory_eval_metrics", "one_step_nrmse", "mean"),
        ),
        "trajectory_eval_rollout_rmse_mean": _mean_nested(
            member_summaries,
            ("trajectory_eval_metrics", "rollout_rmse", "mean"),
        ),
        "trajectory_eval_rollout_nrmse_mean": _mean_nested(
            member_summaries,
            ("trajectory_eval_metrics", "rollout_nrmse", "mean"),
        ),
    }


def _select_regime(
    regime_results: dict[str, dict[str, Any]],
    *,
    preferred: str,
) -> str:
    if preferred != "auto":
        if preferred not in regime_results:
            raise ValueError(f"Requested regime {preferred!r} is not available.")
        return preferred

    ae_floor = min(
        result["aggregate"]["ae_eval_loss_mean"]
        for result in regime_results.values()
    )
    rollout_floor = min(
        result["aggregate"]["trajectory_eval_rollout_nrmse_mean"]
        for result in regime_results.values()
    )
    best_regime = ""
    best_score = float("inf")
    for regime, result in regime_results.items():
        ae_ratio = result["aggregate"]["ae_eval_loss_mean"] / max(ae_floor, 1.0e-8)
        rollout_ratio = result["aggregate"]["trajectory_eval_rollout_nrmse_mean"] / max(
            rollout_floor,
            1.0e-8,
        )
        selection_score = ae_ratio * rollout_ratio
        result["aggregate"]["selection_score"] = selection_score
        if selection_score < best_score:
            best_score = selection_score
            best_regime = regime
    if not best_regime:
        raise ValueError("Failed to select a training regime.")
    return best_regime


def _curve_point(
    *,
    aggregate: dict[str, Any],
    online_solver_transitions: int,
    offline_transitions: int,
    round_index: int,
) -> dict[str, Any]:
    return {
        "round_index": round_index,
        "online_solver_transitions": int(online_solver_transitions),
        "total_transitions": int(offline_transitions + online_solver_transitions),
        "trajectory_eval_one_step_rmse_mean": float(aggregate["trajectory_eval_one_step_rmse_mean"]),
        "trajectory_eval_one_step_nrmse_mean": float(aggregate["trajectory_eval_one_step_nrmse_mean"]),
        "trajectory_eval_rollout_rmse_mean": float(aggregate["trajectory_eval_rollout_rmse_mean"]),
        "trajectory_eval_rollout_nrmse_mean": float(aggregate["trajectory_eval_rollout_nrmse_mean"]),
    }


def _mean_nested(items: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for item in items:
        current: Any = item
        for key in path:
            current = current[key]
        values.append(float(current))
    return float(statistics.mean(values))


def _write_extended_dataset(
    *,
    previous_records: list[TransitionRecord],
    previous_manifest: dict[str, Any],
    new_records: list[TransitionRecord],
    dataset_root: Path,
    dataset_version: str,
    offline_transitions: int,
    online_solver_transitions: int,
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
        extras={
            **dict(previous_manifest.get("extras", {})),
            "offline_transitions": int(offline_transitions),
            "online_solver_transitions": int(online_solver_transitions),
        },
    )
    OfflineDatasetWriter(dataset_root).write(all_records, manifest)


def _plot_strategy_curves(strategy_results: dict[str, dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _plot_metric_group(
        strategy_results,
        output_root / "eval_nrmse_vs_online_solver_transitions.png",
        x_key="online_solver_transitions",
        one_step_key="trajectory_eval_one_step_nrmse_mean",
        rollout_key="trajectory_eval_rollout_nrmse_mean",
        title_prefix="Eval NRMSE",
    )
    _plot_metric_group(
        strategy_results,
        output_root / "eval_nrmse_vs_total_transitions.png",
        x_key="total_transitions",
        one_step_key="trajectory_eval_one_step_nrmse_mean",
        rollout_key="trajectory_eval_rollout_nrmse_mean",
        title_prefix="Eval NRMSE",
    )


def _plot_metric_group(
    strategy_results: dict[str, dict[str, Any]],
    figure_path: Path,
    *,
    x_key: str,
    one_step_key: str,
    rollout_key: str,
    title_prefix: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for strategy, result in strategy_results.items():
        xs = [point[x_key] for point in result["curve"]]
        one_step_values = [point[one_step_key] for point in result["curve"]]
        rollout_values = [point[rollout_key] for point in result["curve"]]
        axes[0].plot(xs, one_step_values, marker="o", label=strategy)
        axes[1].plot(xs, rollout_values, marker="o", label=strategy)
    axes[0].set_title(f"{title_prefix} One-Step")
    axes[1].set_title(f"{title_prefix} Rollout")
    axes[0].set_xlabel(x_key)
    axes[1].set_xlabel(x_key)
    axes[0].set_ylabel("NRMSE")
    axes[1].set_ylabel("NRMSE")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    figure.tight_layout()
    figure.savefig(figure_path)
    plt.close(figure)


def _format_regime_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Regime Ablation",
        "",
        "| Regime | Selection Score | AE Eval Loss | Eval Rollout NRMSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    selected_regime = str(summary["selected_regime"])
    for regime, result in summary["regimes"].items():
        aggregate = result["aggregate"]
        marker = " <- selected" if regime == selected_regime else ""
        lines.append(
            "| "
            f"{regime}{marker} | "
            f"{float(aggregate.get('selection_score', 0.0)):.6f} | "
            f"{float(aggregate['ae_eval_loss_mean']):.6f} | "
            f"{float(aggregate['trajectory_eval_rollout_nrmse_mean']):.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_benchmark_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# World Model Benchmark",
        "",
        f"- selected regime: `{summary['selected_regime']}`",
        f"- offline transitions: `{summary['offline_transitions']}`",
        f"- online transition budget: `{summary['online_solver_transition_budget']}`",
        "",
        "| Strategy | Online Transitions | Eval Rollout NRMSE |",
        "| --- | ---: | ---: |",
    ]
    for strategy, result in summary["strategy_results"].items():
        final_point = result["curve"][-1]
        lines.append(
            "| "
            f"{strategy} | "
            f"{int(result['online_solver_transitions'])} | "
            f"{float(final_point['trajectory_eval_rollout_nrmse_mean']):.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


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
        return ["logging.wandb.enabled=false"]
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


def _resolve_wandb_group(group_override: str | None, *group_parts: Any) -> str:
    prefix = str(group_override).strip() if group_override is not None else ""
    if prefix:
        return compose_wandb_group(prefix, *group_parts)
    return compose_wandb_group(*group_parts)


if __name__ == "__main__":
    main()
