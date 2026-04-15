"""Generative acquisition benchmark — compares heuristic strategies with
flow-matching-based generative sampling.

This benchmark extends the existing world-model benchmark with three new
strategies that use a Conditional Flow Matching generative model trained
in the compressed latent space:

    generative_loss_weighted  – flow matching with loss-proportional weights
    generative_uniform        – flow matching with uniform weights (ablation)
    generative_combined       – merge generative + heuristic candidate pools

All 9 strategies share the same offline warm-up, regime ablation, and
evaluation protocol for a fair comparison.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pdewm.acquisition.generative import (
    FlowMatchingSamplerConfig,
    LatentFlowMatchingSampler,
    build_generative_candidates,
    compute_transition_losses,
)
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
from pdewm.utils.device import resolve_device
from pdewm.utils.git import get_git_commit_hash


# --------------------------------------------------------------------------
# Default strategies
# --------------------------------------------------------------------------

HEURISTIC_STRATEGIES = (
    "offline_only",
    "random_states",
    "uncertainty_only",
    "diversity_only",
    "uncertainty_diversity",
    "ours",
)
GENERATIVE_STRATEGIES = (
    "generative_loss_weighted",
    "generative_uniform",
    "generative_combined",
)
ALL_STRATEGIES = (*HEURISTIC_STRATEGIES, *GENERATIVE_STRATEGIES)
DEFAULT_REGIMES = ("frozen", "joint_no_ema", "joint_ema")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark comparing heuristic and generative acquisition strategies."
    )
    # Data
    parser.add_argument("--data-config", default="burgers_1d")
    parser.add_argument("--solver-config", default="burgers_1d")
    parser.add_argument("--dataset-root", default="data/generated_genbench/burgers_1d_offline/genbench")
    parser.add_argument("--output-root", default="artifacts/runs/generative_benchmark")
    parser.add_argument("--data-version", default="genbench")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--prepare-ae", action="store_true")

    # Dataset
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--train-trajectories", type=int, default=32)
    parser.add_argument("--val-trajectories", type=int, default=8)
    parser.add_argument("--test-trajectories", type=int, default=8)
    parser.add_argument("--ood-trajectories", type=int, default=8)

    # AE architecture
    parser.add_argument("--ae-base-channels", type=int, default=16)
    parser.add_argument("--ae-latent-channels", type=int, default=32)
    parser.add_argument("--ae-channel-mults", nargs="+", type=int, default=[1, 2, 4])

    # Dynamics architecture
    parser.add_argument("--dyn-hidden-channels", type=int, default=64)
    parser.add_argument("--dyn-context-hidden", type=int, default=64)
    parser.add_argument("--dyn-context-output", type=int, default=64)
    parser.add_argument("--dyn-num-blocks", type=int, default=4)

    # Training
    parser.add_argument("--ae-epochs", type=int, default=200)
    parser.add_argument("--dynamics-epochs", type=int, default=200)
    parser.add_argument("--fine-tune-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)

    # Ensemble
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)

    # Acquisition
    parser.add_argument("--online-solver-transitions", type=int, default=256)
    parser.add_argument("--transitions-per-round", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=192)
    parser.add_argument("--top-m", type=int, default=48)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--diversity-lambda", type=float, default=0.2)

    # Flow matching config
    parser.add_argument("--flow-hidden-channels", type=int, default=64)
    parser.add_argument("--flow-num-blocks", type=int, default=4)
    parser.add_argument("--flow-epochs", type=int, default=200)
    parser.add_argument("--flow-lr", type=float, default=1e-3)
    parser.add_argument("--flow-ode-steps", type=int, default=30)
    parser.add_argument("--flow-temperature", type=float, default=1.0)
    parser.add_argument("--flow-candidates", type=int, default=192,
                        help="Number of candidates to generate from flow model per round")

    # Regime & strategy selection
    parser.add_argument("--selected-regime", default="auto")
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
    parser.add_argument("--strategies", nargs="+", default=list(ALL_STRATEGIES))

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="pde-world-model")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-group", default="")

    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    # --- Data generation ---
    if args.prepare_data or not dataset_root.exists():
        _run([
            sys.executable, "scripts/generate_offline_data.py",
            "--data-config", args.data_config,
            "--solver-config", args.solver_config,
            f"data.dataset_version={args.data_version}",
            f"data.output_dir={Path(args.dataset_root).parents[1]}",
            f"data.splits.train.num_trajectories={args.train_trajectories}",
            f"data.splits.val.num_trajectories={args.val_trajectories}",
            f"data.splits.test.num_trajectories={args.test_trajectories}",
            f"data.splits.parameter_ood.num_trajectories={args.ood_trajectories}",
            f"data.num_steps={args.num_steps}",
            f"solver.grid_size={args.grid_size}",
            f"project.seed={args.seed}",
        ])

    # --- AE warm-up ---
    ae_dir = output_root / "warmup_autoencoder"
    ae_checkpoint = ae_dir / "best.pt"
    if args.prepare_ae or not ae_checkpoint.exists():
        ae_cmd = [
            sys.executable, "scripts/train_autoencoder.py",
            f"train.dataset_root={dataset_root}",
            f"train.output_dir={ae_dir}",
            f"train.epochs={args.ae_epochs}",
            f"train.batch_size={args.batch_size}",
            f"project.seed={args.seed}",
            f"model.base_channels={args.ae_base_channels}",
            f"model.latent_channels={args.ae_latent_channels}",
            f"model.channel_multipliers=[{','.join(str(m) for m in args.ae_channel_mults)}]",
        ]
        ae_cmd.extend(_wandb_overrides(
            enabled=args.wandb, project=args.wandb_project,
            entity=args.wandb_entity, mode=args.wandb_mode,
            group=args.wandb_group or f"genbench_{args.data_version}",
            name=f"{args.data_version}_warmup_ae",
            tags=["generative_benchmark", args.data_config, "warmup_ae"],
        ))
        _run(ae_cmd)

    _, manifest = load_transition_records(dataset_root)
    offline_transitions = int(manifest["samples"])

    # --- Regime ablation ---
    regime_root = output_root / "regime_ablation"
    regime_root.mkdir(parents=True, exist_ok=True)
    regime_results: dict[str, dict[str, Any]] = {}
    for regime in args.regimes:
        regime_results[regime] = _run_regime_ablation(
            dataset_root=dataset_root, output_dir=regime_root / regime,
            ae_checkpoint=ae_checkpoint, regime=regime,
            epochs=args.dynamics_epochs, ensemble_size=args.ensemble_size,
            seed=args.seed, batch_size=args.batch_size,
            dyn_hidden=args.dyn_hidden_channels,
            dyn_ctx_hidden=args.dyn_context_hidden,
            dyn_ctx_output=args.dyn_context_output,
            dyn_num_blocks=args.dyn_num_blocks,
            wandb_args=args,
        )
    regime_summary = {
        "regimes": regime_results,
        "selected_regime": _select_regime(regime_results, preferred=args.selected_regime),
    }
    (regime_root / "summary.json").write_text(
        json.dumps(regime_summary, indent=2), encoding="utf-8"
    )

    selected_regime = str(regime_summary["selected_regime"])
    selected_regime_result = regime_results[selected_regime]

    # --- Acquisition benchmark ---
    acquisition_root = output_root / "acquisition_benchmark"
    acquisition_root.mkdir(parents=True, exist_ok=True)
    strategy_results: dict[str, dict[str, Any]] = {}

    for strategy in args.strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")
        strategy_results[strategy] = _run_strategy_benchmark(
            strategy=strategy,
            dataset_root=dataset_root,
            output_dir=acquisition_root / strategy,
            ae_checkpoint=ae_checkpoint,
            offline_transitions=offline_transitions,
            selected_regime=selected_regime,
            selected_regime_result=selected_regime_result,
            args=args,
        )

    # --- Summary ---
    benchmark_summary = {
        "dataset_root": str(dataset_root),
        "offline_transitions": offline_transitions,
        "selected_regime": selected_regime,
        "regime_summary": regime_summary,
        "strategy_results": strategy_results,
    }
    (output_root / "benchmark_summary.json").write_text(
        json.dumps(benchmark_summary, indent=2), encoding="utf-8"
    )
    (output_root / "benchmark_summary.md").write_text(
        _format_benchmark_markdown(benchmark_summary), encoding="utf-8"
    )
    _plot_all_curves(strategy_results, output_root)
    print(f"\nResults: {output_root / 'benchmark_summary.md'}")


# --------------------------------------------------------------------------
# Strategy execution
# --------------------------------------------------------------------------

def _run_strategy_benchmark(
    *, strategy: str, dataset_root: Path, output_dir: Path,
    ae_checkpoint: Path, offline_transitions: int,
    selected_regime: str, selected_regime_result: dict[str, Any],
    args,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    curve = [_curve_point(
        aggregate=selected_regime_result["aggregate"],
        online_solver_transitions=0,
        offline_transitions=offline_transitions,
        round_index=0,
    )]
    result: dict[str, Any] = {
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

    is_generative = strategy.startswith("generative_")
    current_dataset_root = dataset_root
    current_committee_checkpoints = list(selected_regime_result["member_checkpoints"])
    cumulative_online = 0
    round_index = 0
    device = resolve_device("auto")
    device_str = str(device)

    while cumulative_online < args.online_solver_transitions:
        remaining = args.online_solver_transitions - cumulative_online
        round_budget = min(args.transitions_per_round, remaining)
        round_dir = output_dir / f"round_{round_index + 1:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Load committee
        members = load_world_model_committee(
            current_committee_checkpoints, str(ae_checkpoint),
            device=device_str,
        )
        records, manifest = load_transition_records(current_dataset_root)
        train_records = [r for r in records if r.metadata.split == "train"]

        # Generate candidates
        if is_generative:
            candidates = _generate_flow_candidates(
                strategy=strategy, train_records=train_records,
                members=members, device=device, args=args,
                memory_latents_flat=None,  # computed inside
                round_dir=round_dir,
            )
        else:
            memory_records, memory_latents = build_memory_bank(train_records, members)
            candidates = propose_candidates(
                memory_records, memory_latents, members,
                pool_size=args.pool_size, noise_std=args.noise_std,
                alpha=1.0, beta=0.2, gamma=0.5,
                max_abs_factor=2.0, seed=args.seed + round_index,
            )

        # Rank and acquire
        ordered = rank_candidates(
            candidates, strategy=strategy,
            top_m=args.top_m, diversity_lambda=args.diversity_lambda,
            seed=args.seed + round_index,
        )
        acquisition = acquire_transition_budget(
            ordered, rollout_horizon=args.rollout_horizon,
            transition_budget=round_budget,
            round_index=round_index + 1,
            sample_origin=f"online_{strategy}",
        )
        if not acquisition.new_records:
            print(f"  No records acquired in round {round_index + 1}, stopping.")
            break

        # Extend dataset
        next_dataset_root = output_dir / "datasets" / f"round_{round_index + 1:02d}"
        _write_extended_dataset(
            previous_records=records, previous_manifest=manifest,
            new_records=acquisition.new_records,
            dataset_root=next_dataset_root,
            dataset_version=f"{dataset_root.name}_{strategy}_r{round_index+1:02d}",
        )

        # Fine-tune committee
        committee_result = _fine_tune_committee(
            dataset_root=next_dataset_root,
            output_dir=round_dir / "dynamics_committee",
            ae_checkpoint=ae_checkpoint, regime=selected_regime,
            epochs=args.fine_tune_epochs,
            ensemble_size=len(current_committee_checkpoints),
            seed=args.seed + 1000 * (round_index + 1),
            resume_checkpoints=current_committee_checkpoints,
            batch_size=args.batch_size,
            dyn_hidden=args.dyn_hidden_channels,
            dyn_ctx_hidden=args.dyn_context_hidden,
            dyn_ctx_output=args.dyn_context_output,
            dyn_num_blocks=args.dyn_num_blocks,
            wandb_args=args, strategy=strategy,
            round_index=round_index + 1,
        )

        cumulative_online += acquisition.transitions_acquired
        round_summary = {
            "round": round_index + 1,
            "acquisition": acquisition.to_dict(),
            "aggregate": committee_result["aggregate"],
            "online_solver_transitions": cumulative_online,
        }
        result["rounds"].append(round_summary)
        result["curve"].append(_curve_point(
            aggregate=committee_result["aggregate"],
            online_solver_transitions=cumulative_online,
            offline_transitions=offline_transitions,
            round_index=round_index + 1,
        ))
        result["online_solver_transitions"] = cumulative_online
        current_dataset_root = next_dataset_root
        current_committee_checkpoints = list(committee_result["member_checkpoints"])
        round_index += 1
        print(f"  Round {round_index}: +{acquisition.transitions_acquired} transitions "
              f"(total online: {cumulative_online})")

    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# --------------------------------------------------------------------------
# Generative candidate generation
# --------------------------------------------------------------------------

def _generate_flow_candidates(
    *, strategy: str, train_records: list[TransitionRecord],
    members, device: torch.device, args,
    memory_latents_flat: np.ndarray | None,
    round_dir: Path,
) -> list[Any]:
    """Generate candidates using the flow matching model.

    For ``generative_combined``, merges flow candidates with heuristic
    candidates for the best of both worlds.
    """
    import torch

    # Compute transition losses and get 2D latents
    print("  Computing transition losses...")
    latents_2d, losses = compute_transition_losses(
        train_records, members, device=device,
    )
    latent_channels = latents_2d.shape[1]
    latent_length = latents_2d.shape[2]

    # Build memory bank for scoring (flat latents)
    memory_records, mem_latents_flat = build_memory_bank(train_records, members)

    # Choose temperature based on strategy
    temperature = args.flow_temperature if strategy != "generative_uniform" else 0.0

    # Create and train flow model
    config = FlowMatchingSamplerConfig(
        hidden_channels=args.flow_hidden_channels,
        num_conv_blocks=args.flow_num_blocks,
        training_epochs=args.flow_epochs,
        learning_rate=args.flow_lr,
        ode_steps=args.flow_ode_steps,
        temperature=temperature,
    )
    sampler = LatentFlowMatchingSampler(
        config, latent_channels, latent_length, device,
    )
    print(f"  Training flow model (temperature={temperature})...")
    stats = sampler.fit(latents_2d, losses, verbose=True)
    print(f"  Flow training done: {stats}")

    # Save flow model stats
    (round_dir / "flow_training_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    # Generate candidates
    print(f"  Generating {args.flow_candidates} flow candidates...")
    gen_candidates = build_generative_candidates(
        sampler, members, mem_latents_flat,
        n_candidates=args.flow_candidates,
        device=device,
    )

    # For combined strategy, merge with heuristic candidates
    if strategy == "generative_combined":
        heuristic_candidates = propose_candidates(
            memory_records, mem_latents_flat, members,
            pool_size=args.pool_size, noise_std=args.noise_std,
            alpha=1.0, beta=0.2, gamma=0.5,
            max_abs_factor=2.0, seed=args.seed,
        )
        gen_candidates.extend(heuristic_candidates)
        print(f"  Combined: {args.flow_candidates} generative + "
              f"{len(heuristic_candidates)} heuristic candidates")

    return gen_candidates


# --------------------------------------------------------------------------
# Regime ablation
# --------------------------------------------------------------------------

def _run_regime_ablation(
    *, dataset_root: Path, output_dir: Path, ae_checkpoint: Path,
    regime: str, epochs: int, ensemble_size: int, seed: int,
    batch_size: int, dyn_hidden: int, dyn_ctx_hidden: int,
    dyn_ctx_output: int, dyn_num_blocks: int, wandb_args,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    member_summaries = []
    member_checkpoints = []
    for mi in range(ensemble_size):
        member_dir = output_dir / f"member_{mi:02d}"
        member_summary = member_dir / "summary.json"
        if not member_summary.exists():
            cmd = [
                sys.executable, "scripts/train_dynamics.py",
                f"train.dataset_root={dataset_root}",
                f"train.ae_checkpoint={ae_checkpoint}",
                f"train.output_dir={member_dir}",
                f"train.epochs={epochs}",
                f"project.seed={seed + mi}",
                f"train.regime={regime}",
                f"train.batch_size={batch_size}",
                "train.ae_loss_scale=1.0", "train.ema.decay=0.995",
                f"model.hidden_channels={dyn_hidden}",
                f"model.context_hidden_dim={dyn_ctx_hidden}",
                f"model.context_output_dim={dyn_ctx_output}",
                f"model.num_blocks={dyn_num_blocks}",
            ]
            cmd.extend(_wandb_overrides(
                enabled=wandb_args.wandb,
                project=wandb_args.wandb_project,
                entity=wandb_args.wandb_entity,
                mode=wandb_args.wandb_mode,
                group=wandb_args.wandb_group or "genbench_regime",
                name=f"{regime}_m{mi:02d}",
                tags=["generative_benchmark", "regime_ablation", regime],
            ))
            _run(cmd)
        member_summaries.append(json.loads(member_summary.read_text(encoding="utf-8")))
        member_checkpoints.append(str(member_dir / "best.pt"))

    result = {
        "regime": regime,
        "ensemble_size": ensemble_size,
        "aggregate": _aggregate_ensemble(member_summaries),
        "members": member_summaries,
        "member_checkpoints": member_checkpoints,
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _fine_tune_committee(
    *, dataset_root: Path, output_dir: Path, ae_checkpoint: Path,
    regime: str, epochs: int, ensemble_size: int, seed: int,
    resume_checkpoints: list[str], batch_size: int,
    dyn_hidden: int, dyn_ctx_hidden: int, dyn_ctx_output: int,
    dyn_num_blocks: int, wandb_args, strategy: str, round_index: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    member_summaries = []
    member_checkpoints = []
    for mi in range(ensemble_size):
        member_dir = output_dir / f"member_{mi:02d}"
        member_summary = member_dir / "summary.json"
        if not member_summary.exists():
            cmd = [
                sys.executable, "scripts/train_dynamics.py",
                f"train.dataset_root={dataset_root}",
                f"train.ae_checkpoint={ae_checkpoint}",
                f"train.output_dir={member_dir}",
                f"train.epochs={epochs}",
                f"project.seed={seed + mi}",
                f"train.regime={regime}",
                f"train.resume_checkpoint={resume_checkpoints[mi]}",
                f"train.batch_size={batch_size}",
                "train.ae_loss_scale=1.0", "train.ema.decay=0.995",
                f"model.hidden_channels={dyn_hidden}",
                f"model.context_hidden_dim={dyn_ctx_hidden}",
                f"model.context_output_dim={dyn_ctx_output}",
                f"model.num_blocks={dyn_num_blocks}",
            ]
            cmd.extend(_wandb_overrides(
                enabled=wandb_args.wandb,
                project=wandb_args.wandb_project,
                entity=wandb_args.wandb_entity,
                mode=wandb_args.wandb_mode,
                group=wandb_args.wandb_group or f"genbench_{strategy}",
                name=f"{strategy}_r{round_index:02d}_m{mi:02d}",
                tags=["generative_benchmark", "acquisition", strategy, regime],
            ))
            _run(cmd)
        member_summaries.append(json.loads(member_summary.read_text(encoding="utf-8")))
        member_checkpoints.append(str(member_dir / "best.pt"))
    return {
        "aggregate": _aggregate_ensemble(member_summaries),
        "members": member_summaries,
        "member_checkpoints": member_checkpoints,
    }


# --------------------------------------------------------------------------
# Aggregation & selection
# --------------------------------------------------------------------------

def _aggregate_ensemble(summaries: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "best_val_loss_mean": _mean_nested(summaries, ("best_val_loss",)),
        "ae_val_loss_mean": _mean_nested(summaries, ("ae_val_metrics", "loss")),
        "ae_test_loss_mean": _mean_nested(summaries, ("ae_test_metrics", "loss")),
        "trajectory_val_one_step_rmse_mean": _mean_nested(
            summaries, ("trajectory_val_metrics", "one_step_rmse", "mean")),
        "trajectory_val_one_step_nrmse_mean": _mean_nested(
            summaries, ("trajectory_val_metrics", "one_step_nrmse", "mean")),
        "trajectory_val_rollout_rmse_mean": _mean_nested(
            summaries, ("trajectory_val_metrics", "rollout_rmse", "mean")),
        "trajectory_val_rollout_nrmse_mean": _mean_nested(
            summaries, ("trajectory_val_metrics", "rollout_nrmse", "mean")),
        "trajectory_test_one_step_rmse_mean": _mean_nested(
            summaries, ("trajectory_test_metrics", "one_step_rmse", "mean")),
        "trajectory_test_one_step_nrmse_mean": _mean_nested(
            summaries, ("trajectory_test_metrics", "one_step_nrmse", "mean")),
        "trajectory_test_rollout_rmse_mean": _mean_nested(
            summaries, ("trajectory_test_metrics", "rollout_rmse", "mean")),
        "trajectory_test_rollout_nrmse_mean": _mean_nested(
            summaries, ("trajectory_test_metrics", "rollout_nrmse", "mean")),
    }


def _select_regime(results: dict[str, dict[str, Any]], *, preferred: str) -> str:
    if preferred != "auto":
        return preferred
    ae_floor = min(r["aggregate"]["ae_val_loss_mean"] for r in results.values())
    rollout_floor = min(
        r["aggregate"]["trajectory_val_rollout_nrmse_mean"] for r in results.values()
    )
    best, best_score = "", float("inf")
    for regime, result in results.items():
        a = result["aggregate"]
        score = (
            a["ae_val_loss_mean"] / max(ae_floor, 1e-8)
            * a["trajectory_val_rollout_nrmse_mean"] / max(rollout_floor, 1e-8)
        )
        a["selection_score"] = score
        if score < best_score:
            best, best_score = regime, score
    return best


def _curve_point(*, aggregate, online_solver_transitions, offline_transitions, round_index):
    return {
        "round_index": round_index,
        "online_solver_transitions": int(online_solver_transitions),
        "total_transitions": int(offline_transitions + online_solver_transitions),
        "trajectory_val_rollout_nrmse_mean": float(aggregate["trajectory_val_rollout_nrmse_mean"]),
        "trajectory_val_one_step_nrmse_mean": float(aggregate["trajectory_val_one_step_nrmse_mean"]),
        "trajectory_test_rollout_nrmse_mean": float(aggregate["trajectory_test_rollout_nrmse_mean"]),
        "trajectory_test_one_step_nrmse_mean": float(aggregate["trajectory_test_one_step_nrmse_mean"]),
    }


def _mean_nested(items, path):
    vals = []
    for item in items:
        curr = item
        for k in path:
            curr = curr[k]
        vals.append(float(curr))
    return float(statistics.mean(vals))


# --------------------------------------------------------------------------
# Dataset extension
# --------------------------------------------------------------------------

def _write_extended_dataset(
    *, previous_records, previous_manifest, new_records,
    dataset_root: Path, dataset_version: str,
):
    dataset_root.mkdir(parents=True, exist_ok=True)
    all_records = [*previous_records, *new_records]
    splits: dict[str, int] = {}
    tids = set()
    for r in all_records:
        splits[r.metadata.split] = splits.get(r.metadata.split, 0) + 1
        tids.add(r.metadata.trajectory_id)
    manifest = DatasetManifest(
        dataset_name=str(previous_manifest["dataset_name"]),
        dataset_version=dataset_version,
        generator_git_hash=get_git_commit_hash(),
        solver_version=str(previous_manifest["solver_version"]),
        parameter_space_signature=dict(previous_manifest["parameter_space_signature"]),
        seed_policy=dict(previous_manifest["seed_policy"]),
        samples=len(all_records), trajectories=len(tids),
        split_counts=splits,
        created_at=datetime.now(UTC).isoformat(),
        sample_shape=tuple(previous_manifest["sample_shape"]),
        extras=dict(previous_manifest.get("extras", {})),
    )
    OfflineDatasetWriter(dataset_root).write(all_records, manifest)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _plot_all_curves(results: dict[str, dict[str, Any]], out: Path):
    colors_heuristic = {
        "offline_only": "#999999",
        "random_states": "#FFA500",
        "uncertainty_only": "#4A90D9",
        "diversity_only": "#50C878",
        "uncertainty_diversity": "#9370DB",
        "ours": "#DC143C",
    }
    colors_generative = {
        "generative_loss_weighted": "#FF1493",
        "generative_uniform": "#FF69B4",
        "generative_combined": "#8B0000",
    }
    colors = {**colors_heuristic, **colors_generative}

    for split in ("val", "test"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for strategy, r in results.items():
            xs = [p["online_solver_transitions"] for p in r["curve"]]
            c = colors.get(strategy, "#333333")
            lw = 2.5 if strategy.startswith("generative_") else 1.5
            ls = "-" if not strategy.startswith("generative_") else "--"
            axes[0].plot(
                xs, [p[f"trajectory_{split}_one_step_nrmse_mean"] for p in r["curve"]],
                marker="o", color=c, linewidth=lw, linestyle=ls, label=strategy, markersize=4,
            )
            axes[1].plot(
                xs, [p[f"trajectory_{split}_rollout_nrmse_mean"] for p in r["curve"]],
                marker="o", color=c, linewidth=lw, linestyle=ls, label=strategy, markersize=4,
            )
        axes[0].set_title(f"{split.title()} — One-Step NRMSE vs Online Transitions")
        axes[1].set_title(f"{split.title()} — Rollout NRMSE vs Online Transitions")
        for ax in axes:
            ax.set_xlabel("Online Solver Transitions")
            ax.set_ylabel("NRMSE")
            ax.grid(True, alpha=0.3)
        axes[1].legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{split}_nrmse_curves.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------
# Markdown summary
# --------------------------------------------------------------------------

def _format_benchmark_markdown(s: dict[str, Any]) -> str:
    lines = [
        "# Generative Acquisition Benchmark", "",
        f"- selected regime: `{s['selected_regime']}`",
        f"- offline transitions: `{s['offline_transitions']}`", "",
        "## Strategy Comparison", "",
        "| Strategy | Type | Online Trans. | Val Rollout NRMSE | Test Rollout NRMSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for strat, r in s["strategy_results"].items():
        final = r["curve"][-1]
        stype = "generative" if strat.startswith("generative_") else "heuristic"
        lines.append(
            f"| {strat} | {stype} | {int(r['online_solver_transitions'])} | "
            f"{float(final['trajectory_val_rollout_nrmse_mean']):.6f} | "
            f"{float(final['trajectory_test_rollout_nrmse_mean']):.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _run(cmd):
    subprocess.run(cmd, check=True)


def _wandb_overrides(*, enabled, project, entity, mode, group, name, tags):
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


if __name__ == "__main__":
    main()
