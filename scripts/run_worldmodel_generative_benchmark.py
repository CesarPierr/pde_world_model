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
import concurrent.futures
import json
import math
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
import torch
from tqdm.auto import tqdm

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
from pdewm.utils.wandb import compose_wandb_group


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

STRATEGY_DISPLAY_NAMES = {
    "offline_only": "Offline Only",
    "random_states": "Random States",
    "uncertainty_only": "Uncertainty Only",
    "diversity_only": "Diversity Only",
    "uncertainty_diversity": "Uncertainty + Diversity",
    "ours": "Ours (Uncertainty + Diversity + Transition Gain)",
    "generative_loss_weighted": "Generative (Loss-Weighted CFM)",
    "generative_uniform": "Generative (Uniform CFM)",
    "generative_combined": "Generative + Heuristic Combined",
}


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
    parser.add_argument(
        "--benchmark-profile",
        choices=["custom", "realistic_1d", "realistic_1d_gpumax"],
        default="realistic_1d",
        help="Preset for realistic AL4PDE-like 1D runs. Use custom to keep manual values.",
    )
    parser.add_argument(
        "--burgers-al4pde-time",
        action="store_true",
        default=True,
        help="Use AL4PDE-inspired Burgers time setup (solver.dt=0.01, num_steps=200 => T=2.0).",
    )
    parser.add_argument(
        "--no-burgers-al4pde-time",
        action="store_false",
        dest="burgers_al4pde_time",
    )

    # Dataset
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--train-trajectories", type=int, default=32)
    parser.add_argument("--eval-trajectories", type=int, default=8)
    parser.add_argument("--ood-trajectories", type=int, default=8)

    # Burgers challenge (applied only when data/solver config is burgers_1d)
    parser.add_argument("--burgers-challenging", action="store_true", default=True)
    parser.add_argument("--no-burgers-challenging", action="store_false", dest="burgers_challenging")
    parser.add_argument("--burgers-viscosity-range", nargs=2, type=float, default=[0.001, 0.08])
    parser.add_argument("--burgers-ic-amplitude-range", nargs=2, type=float, default=[0.6, 1.8])
    parser.add_argument("--burgers-ic-bandwidth-range", nargs=2, type=int, default=[2, 6])
    parser.add_argument("--burgers-forcing-amplitude-range", nargs=2, type=float, default=[0.0, 0.12])
    parser.add_argument("--burgers-forcing-mode-choices", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--burgers-ood-viscosity-range", nargs=2, type=float, default=[0.0005, 0.02])
    parser.add_argument("--burgers-ood-ic-amplitude-range", nargs=2, type=float, default=[1.2, 2.6])
    parser.add_argument("--burgers-ood-ic-bandwidth-range", nargs=2, type=int, default=[4, 10])
    parser.add_argument("--burgers-ood-forcing-amplitude-range", nargs=2, type=float, default=[0.05, 0.2])
    parser.add_argument("--burgers-ood-forcing-mode-choices", nargs="+", type=int, default=[2, 3, 4, 5, 6])

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
    parser.add_argument("--train-num-workers", type=int, default=8)
    parser.add_argument("--train-prefetch-factor", type=int, default=4)

    # Ensemble
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument(
        "--ensemble-train-mode",
        choices=["sequential", "parallel"],
        default="sequential",
        help="How to train committee members: sequential (safer on single GPU) or parallel.",
    )
    parser.add_argument(
        "--ensemble-max-parallel",
        type=int,
        default=2,
        help="Max concurrent member trainings when --ensemble-train-mode=parallel.",
    )
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
    parser.add_argument(
        "--wandb-log-acquisition-training",
        action="store_true",
        default=True,
        help="Log committee fine-tuning during acquisition to W&B (enabled by default, resumes the same run per member across rounds).",
    )
    parser.add_argument(
        "--no-wandb-log-acquisition-training",
        action="store_false",
        dest="wandb_log_acquisition_training",
        help="Disable W&B logging for acquisition fine-tuning rounds.",
    )

    args = parser.parse_args()
    _apply_benchmark_profile(args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    benchmark_run_token = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    planned_rounds = max(1, math.ceil(args.online_solver_transitions / max(1, args.transitions_per_round)))
    strategy_steps = sum(1 if s == "offline_only" else (1 + planned_rounds) for s in args.strategies)
    total_steps = 1 + 1 + len(args.regimes) + strategy_steps + 1
    progress = tqdm(total=total_steps, desc="Generative benchmark", unit="step")

    try:
        # --- Data generation ---
        if args.prepare_data or not dataset_root.exists():
            generate_data_cmd = [
                sys.executable, "scripts/generate_offline_data.py",
                "--data-config", args.data_config,
                "--solver-config", args.solver_config,
                f"data.dataset_version={args.data_version}",
                f"data.output_dir={Path(args.dataset_root).parents[1]}",
                f"data.splits.train.num_trajectories={args.train_trajectories}",
                f"data.splits.test.num_trajectories={args.eval_trajectories}",
                f"data.splits.parameter_ood.num_trajectories={args.ood_trajectories}",
                f"data.num_steps={args.num_steps}",
                f"solver.grid_size={args.grid_size}",
                f"project.seed={args.seed}",
            ]
            generate_data_cmd.extend(_burgers_time_overrides(args))
            generate_data_cmd.extend(_burgers_challenging_overrides(args))
            _run(generate_data_cmd)
        progress.update(1)

        # --- AE warm-up ---
        ae_dir = output_root / "warmup_autoencoder"
        ae_checkpoint = ae_dir / "last.pt"
        if args.prepare_ae or not ae_checkpoint.exists():
            ae_cmd = [
                sys.executable, "scripts/train_autoencoder.py",
                f"train.dataset_root={dataset_root}",
                f"train.output_dir={ae_dir}",
                f"train.epochs={args.ae_epochs}",
                f"train.batch_size={args.batch_size}",
                f"train.num_workers={args.train_num_workers}",
                f"train.prefetch_factor={args.train_prefetch_factor}",
                f"project.seed={args.seed}",
                f"model.base_channels={args.ae_base_channels}",
                f"model.latent_channels={args.ae_latent_channels}",
                f"model.channel_multipliers=[{','.join(str(m) for m in args.ae_channel_mults)}]",
            ]
            ae_cmd.extend(_wandb_overrides(
                enabled=args.wandb, project=args.wandb_project,
                entity=args.wandb_entity, mode=args.wandb_mode,
                group=_resolve_wandb_group(
                    args.wandb_group,
                    "gb",
                    args.data_version,
                    "ae",
                ),
                name=f"gb/ae/{args.data_version}/s{int(args.seed)}",
                tags=["generative_benchmark", args.data_config, "warmup_ae"],
            ))
            _run(ae_cmd)
        progress.update(1)

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
            progress.update(1)
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
            strategy_label = _strategy_display_name(strategy)
            print(f"\n{'='*60}")
            print(f"Strategy: {strategy_label} ({strategy})")
            print(f"{'='*60}")
            strategy_results[strategy] = _run_strategy_benchmark(
                strategy=strategy,
                dataset_root=dataset_root,
                output_dir=acquisition_root / strategy,
                ae_checkpoint=ae_checkpoint,
                offline_transitions=offline_transitions,
                selected_regime=selected_regime,
                selected_regime_result=selected_regime_result,
                benchmark_run_token=benchmark_run_token,
                args=args,
                planned_rounds=planned_rounds,
                progress=progress,
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
        progress.update(1)
        print(f"\nResults: {output_root / 'benchmark_summary.md'}")
    finally:
        progress.close()


# --------------------------------------------------------------------------
# Strategy execution
# --------------------------------------------------------------------------

def _run_strategy_benchmark(
    *, strategy: str, dataset_root: Path, output_dir: Path,
    ae_checkpoint: Path, offline_transitions: int,
    selected_regime: str, selected_regime_result: dict[str, Any],
    benchmark_run_token: str,
    args,
    planned_rounds: int,
    progress,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        progress.update(1 if strategy == "offline_only" else (1 + planned_rounds))
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
        progress.update(1)
        return result

    is_generative = strategy.startswith("generative_")
    current_dataset_root = dataset_root
    current_committee_checkpoints = list(selected_regime_result["member_checkpoints"])
    acquisition_wandb_run_ids = _build_acquisition_wandb_run_ids(
        strategy=strategy,
        regime=selected_regime,
        ensemble_size=len(current_committee_checkpoints),
        seed=int(args.seed),
        run_token=benchmark_run_token,
    )
    cumulative_online = 0
    round_index = 0
    device = resolve_device("auto")
    device_str = str(device)

    executed_rounds = 0
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
            wandb_run_ids=acquisition_wandb_run_ids,
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
        executed_rounds += 1
        progress.update(1)
        print(f"  Round {round_index}: +{acquisition.transitions_acquired} transitions "
              f"(total online: {cumulative_online})")

    skipped_rounds = max(0, planned_rounds - executed_rounds)
    if skipped_rounds:
        progress.update(skipped_rounds)
    progress.update(1)

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

    pending_jobs: list[tuple[int, list[str]]] = []
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
                f"train.num_workers={wandb_args.train_num_workers}",
                f"train.prefetch_factor={wandb_args.train_prefetch_factor}",
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
                group=_resolve_wandb_group(
                    wandb_args.wandb_group,
                    "gb",
                    wandb_args.data_version,
                    "abl",
                    regime,
                ),
                name=f"gb/abl/{regime}/m{mi:02d}/s{seed + mi}",
                tags=["generative_benchmark", "regime_ablation", regime],
            ))

            pending_jobs.append((mi, cmd))

    _run_ensemble_jobs(
        jobs=pending_jobs,
        mode=str(getattr(wandb_args, "ensemble_train_mode", "sequential")),
        max_parallel=int(getattr(wandb_args, "ensemble_max_parallel", 1)),
        context=f"regime-ablation/{regime}",
    )

    member_summaries = []
    member_checkpoints = []
    for mi in range(ensemble_size):
        member_dir = output_dir / f"member_{mi:02d}"
        member_summary = member_dir / "summary.json"
        member_summaries.append(json.loads(member_summary.read_text(encoding="utf-8")))
        member_checkpoints.append(str(member_dir / "last.pt"))

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
    wandb_run_ids: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(wandb_run_ids) != ensemble_size:
        raise ValueError(
            f"Expected {ensemble_size} wandb run ids, got {len(wandb_run_ids)}."
        )
    pending_jobs: list[tuple[int, list[str]]] = []
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
                f"train.epoch_offset={(round_index - 1) * epochs}",
                f"train.batch_size={batch_size}",
                f"train.num_workers={wandb_args.train_num_workers}",
                f"train.prefetch_factor={wandb_args.train_prefetch_factor}",
                "train.ae_loss_scale=1.0", "train.ema.decay=0.995",
                f"model.hidden_channels={dyn_hidden}",
                f"model.context_hidden_dim={dyn_ctx_hidden}",
                f"model.context_output_dim={dyn_ctx_output}",
                f"model.num_blocks={dyn_num_blocks}",
            ]
            cmd.extend(_wandb_overrides(
                enabled=wandb_args.wandb and bool(
                    getattr(wandb_args, "wandb_log_acquisition_training", False)
                ),
                project=wandb_args.wandb_project,
                entity=wandb_args.wandb_entity,
                mode=wandb_args.wandb_mode,
                group=_resolve_wandb_group(
                    wandb_args.wandb_group,
                    "gb",
                    wandb_args.data_version,
                    "acq",
                    strategy,
                    regime,
                ),
                name=f"gb/acq/{strategy}/{regime}/m{mi:02d}",
                tags=["generative_benchmark", "acquisition", strategy, regime],
                run_id=wandb_run_ids[mi],
                resume="allow",
            ))

            pending_jobs.append((mi, cmd))

    _run_ensemble_jobs(
        jobs=pending_jobs,
        mode=str(getattr(wandb_args, "ensemble_train_mode", "sequential")),
        max_parallel=int(getattr(wandb_args, "ensemble_max_parallel", 1)),
        context=f"acquisition/{strategy}/round_{round_index:02d}",
    )

    member_summaries = []
    member_checkpoints = []
    for mi in range(ensemble_size):
        member_dir = output_dir / f"member_{mi:02d}"
        member_summary = member_dir / "summary.json"
        member_summaries.append(json.loads(member_summary.read_text(encoding="utf-8")))
        member_checkpoints.append(str(member_dir / "last.pt"))
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
        "final_eval_loss_mean": _mean_nested(summaries, ("final_eval_loss",)),
        "ae_eval_loss_mean": _mean_nested(summaries, ("ae_eval_metrics", "loss")),
        "trajectory_eval_one_step_rmse_mean": _mean_nested(
            summaries, ("trajectory_eval_metrics", "one_step_rmse", "mean")),
        "trajectory_eval_one_step_nrmse_mean": _mean_nested(
            summaries, ("trajectory_eval_metrics", "one_step_nrmse", "mean")),
        "trajectory_eval_one_step_nrmse_p25": _mean_nested(
            summaries, ("trajectory_eval_metrics", "one_step_nrmse", "p25")),
        "trajectory_eval_one_step_nrmse_p50": _mean_nested(
            summaries, ("trajectory_eval_metrics", "one_step_nrmse", "p50")),
        "trajectory_eval_one_step_nrmse_p75": _mean_nested(
            summaries, ("trajectory_eval_metrics", "one_step_nrmse", "p75")),
        "trajectory_eval_rollout_rmse_mean": _mean_nested(
            summaries, ("trajectory_eval_metrics", "rollout_rmse", "mean")),
        "trajectory_eval_rollout_nrmse_mean": _mean_nested(
            summaries, ("trajectory_eval_metrics", "rollout_nrmse", "mean")),
        "trajectory_eval_rollout_nrmse_p25": _mean_nested(
            summaries, ("trajectory_eval_metrics", "rollout_nrmse", "p25")),
        "trajectory_eval_rollout_nrmse_p50": _mean_nested(
            summaries, ("trajectory_eval_metrics", "rollout_nrmse", "p50")),
        "trajectory_eval_rollout_nrmse_p75": _mean_nested(
            summaries, ("trajectory_eval_metrics", "rollout_nrmse", "p75")),
    }


def _select_regime(results: dict[str, dict[str, Any]], *, preferred: str) -> str:
    if preferred != "auto":
        return preferred
    ae_floor = min(r["aggregate"]["ae_eval_loss_mean"] for r in results.values())
    rollout_floor = min(
        r["aggregate"]["trajectory_eval_rollout_nrmse_mean"] for r in results.values()
    )
    best, best_score = "", float("inf")
    for regime, result in results.items():
        a = result["aggregate"]
        score = (
            a["ae_eval_loss_mean"] / max(ae_floor, 1e-8)
            * a["trajectory_eval_rollout_nrmse_mean"] / max(rollout_floor, 1e-8)
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
        "trajectory_eval_rollout_nrmse_mean": float(aggregate["trajectory_eval_rollout_nrmse_mean"]),
        "trajectory_eval_rollout_nrmse_p25": float(aggregate["trajectory_eval_rollout_nrmse_p25"]),
        "trajectory_eval_rollout_nrmse_p50": float(aggregate["trajectory_eval_rollout_nrmse_p50"]),
        "trajectory_eval_rollout_nrmse_p75": float(aggregate["trajectory_eval_rollout_nrmse_p75"]),
        "trajectory_eval_one_step_nrmse_mean": float(aggregate["trajectory_eval_one_step_nrmse_mean"]),
        "trajectory_eval_one_step_nrmse_p25": float(aggregate["trajectory_eval_one_step_nrmse_p25"]),
        "trajectory_eval_one_step_nrmse_p50": float(aggregate["trajectory_eval_one_step_nrmse_p50"]),
        "trajectory_eval_one_step_nrmse_p75": float(aggregate["trajectory_eval_one_step_nrmse_p75"]),
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for strategy, r in results.items():
        xs = [p["online_solver_transitions"] for p in r["curve"]]
        c = colors.get(strategy, "#333333")
        lw = 2.5 if strategy.startswith("generative_") else 1.5
        ls = "-" if not strategy.startswith("generative_") else "--"
        label = _strategy_display_name(strategy)
        axes[0].plot(
            xs, [p["trajectory_eval_one_step_nrmse_mean"] for p in r["curve"]],
            marker="o", color=c, linewidth=lw, linestyle=ls, label=label, markersize=4,
        )
        axes[1].plot(
            xs, [p["trajectory_eval_rollout_nrmse_mean"] for p in r["curve"]],
            marker="o", color=c, linewidth=lw, linestyle=ls, label=label, markersize=4,
        )
    axes[0].set_title("Eval — One-Step NRMSE vs Online Transitions")
    axes[1].set_title("Eval — Rollout NRMSE vs Online Transitions")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Online Solver Transitions")
        ax.set_ylabel("NRMSE")
        ax.grid(True, alpha=0.3)
    axes[1].set_ylabel("NRMSE (log scale)")
    axes[1].legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(str(out / "eval_nrmse_curves.png"), dpi=150)
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
        "| Strategy | Type | Online Trans. | Eval Rollout NRMSE (mean) | Eval Rollout NRMSE (p50 [p25, p75]) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for strat, r in s["strategy_results"].items():
        final = r["curve"][-1]
        stype = "generative" if strat.startswith("generative_") else "heuristic"
        lines.append(
            f"| {_strategy_display_name(strat)} ({strat}) | {stype} | {int(r['online_solver_transitions'])} | "
            f"{float(final['trajectory_eval_rollout_nrmse_mean']):.6f} | "
            f"{float(final['trajectory_eval_rollout_nrmse_p50']):.6f} "
            f"[{float(final['trajectory_eval_rollout_nrmse_p25']):.6f}, {float(final['trajectory_eval_rollout_nrmse_p75']):.6f}] |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _run(cmd):
    subprocess.run(cmd, check=True)


def _run_ensemble_jobs(
    *,
    jobs: list[tuple[int, list[str]]],
    mode: str,
    max_parallel: int,
    context: str,
) -> None:
    if not jobs:
        return

    if mode != "parallel":
        for _, cmd in jobs:
            _run(cmd)
        return

    workers = max(1, min(max_parallel, len(jobs)))
    if workers == 1:
        for _, cmd in jobs:
            _run(cmd)
        return

    print(
        f"  Launching {len(jobs)} ensemble jobs in parallel "
        f"(workers={workers}, context={context})."
    )
    errors: list[tuple[int, Exception]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_member = {
            executor.submit(_run, cmd): member_idx
            for member_idx, cmd in jobs
        }
        for future in concurrent.futures.as_completed(future_to_member):
            member_idx = future_to_member[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - surfaced to caller
                errors.append((member_idx, exc))

    if errors:
        first_member, first_exc = errors[0]
        raise RuntimeError(
            f"Ensemble parallel training failed for member {first_member:02d} in {context}."
        ) from first_exc


def _wandb_overrides(
    *,
    enabled,
    project,
    entity,
    mode,
    group,
    name,
    tags,
    run_id: str | None = None,
    resume: str | None = None,
):
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
    if run_id:
        overrides.append(f"logging.wandb.id={run_id}")
    if resume:
        overrides.append(f"logging.wandb.resume={resume}")
    if entity:
        overrides.append(f"logging.wandb.entity={entity}")
    return overrides


def _resolve_wandb_group(group_override: str | None, *group_parts: Any) -> str:
    """Build a hierarchical W&B group.

    If ``group_override`` is provided, treat it as a campaign prefix instead of
    replacing the full group, so experiment types remain separable in the W&B UI.
    """
    prefix = str(group_override).strip() if group_override is not None else ""
    if prefix:
        return compose_wandb_group(prefix, *group_parts)
    return compose_wandb_group(*group_parts)


def _build_acquisition_wandb_run_ids(
    *,
    strategy: str,
    regime: str,
    ensemble_size: int,
    seed: int,
    run_token: str,
) -> list[str]:
    prefix = _normalise_wandb_id(
        f"gb_{run_token}_{strategy}_{regime}_s{seed}"
    )
    return [f"{prefix}_m{mi:02d}" for mi in range(ensemble_size)]


def _normalise_wandb_id(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    cleaned = cleaned.strip("_-")
    return cleaned[:96] or "gb_run"


def _strategy_display_name(strategy: str) -> str:
    return STRATEGY_DISPLAY_NAMES.get(strategy, strategy)


def _burgers_challenging_overrides(args) -> list[str]:
    if not args.burgers_challenging:
        return []
    if args.data_config != "burgers_1d" or args.solver_config != "burgers_1d":
        return []
    mode_choices = ",".join(str(m) for m in args.burgers_forcing_mode_choices)
    ood_mode_choices = ",".join(str(m) for m in args.burgers_ood_forcing_mode_choices)
    return [
        f"solver.parameter_space.viscosity_range=[{args.burgers_viscosity_range[0]},{args.burgers_viscosity_range[1]}]",
        f"solver.parameter_space.ic_amplitude_range=[{args.burgers_ic_amplitude_range[0]},{args.burgers_ic_amplitude_range[1]}]",
        f"solver.parameter_space.ic_bandwidth_range=[{args.burgers_ic_bandwidth_range[0]},{args.burgers_ic_bandwidth_range[1]}]",
        f"solver.parameter_space.forcing_amplitude_range=[{args.burgers_forcing_amplitude_range[0]},{args.burgers_forcing_amplitude_range[1]}]",
        f"solver.parameter_space.forcing_mode_choices=[{mode_choices}]",
        f"data.splits.parameter_ood.param_overrides.viscosity_range=[{args.burgers_ood_viscosity_range[0]},{args.burgers_ood_viscosity_range[1]}]",
        f"data.splits.parameter_ood.param_overrides.ic_amplitude_range=[{args.burgers_ood_ic_amplitude_range[0]},{args.burgers_ood_ic_amplitude_range[1]}]",
        f"data.splits.parameter_ood.param_overrides.ic_bandwidth_range=[{args.burgers_ood_ic_bandwidth_range[0]},{args.burgers_ood_ic_bandwidth_range[1]}]",
        f"data.splits.parameter_ood.param_overrides.forcing_amplitude_range=[{args.burgers_ood_forcing_amplitude_range[0]},{args.burgers_ood_forcing_amplitude_range[1]}]",
        f"data.splits.parameter_ood.param_overrides.forcing_mode_choices=[{ood_mode_choices}]",
    ]


def _burgers_time_overrides(args) -> list[str]:
    if not args.burgers_al4pde_time:
        return []
    if args.data_config != "burgers_1d" or args.solver_config != "burgers_1d":
        return []
    return [
        "solver.dt=0.01",
        "data.num_steps=200",
    ]


def _apply_benchmark_profile(args) -> None:
    if args.benchmark_profile == "custom":
        return

    ensemble_size_overridden = "--ensemble-size" in sys.argv

    # AL4PDE-like 1D budget: realistic signal for method comparisons.
    args.grid_size = 256
    args.num_steps = 200
    args.train_trajectories = 48
    args.eval_trajectories = 12
    args.ood_trajectories = 12
    args.ae_epochs = 120
    args.dynamics_epochs = 120
    args.fine_tune_epochs = 60
    if not ensemble_size_overridden:
        args.ensemble_size = 5
    args.online_solver_transitions = 384
    args.transitions_per_round = 96
    args.rollout_horizon = 16
    args.pool_size = 256
    args.top_m = 64
    args.flow_epochs = 120
    args.flow_candidates = 256
    args.batch_size = 32
    args.train_num_workers = max(8, int(args.train_num_workers))
    args.train_prefetch_factor = max(2, int(args.train_prefetch_factor))
    args.burgers_al4pde_time = True

    if args.benchmark_profile == "realistic_1d_gpumax":
        # Aggressive but generally safe on modern 24GB+ GPUs.
        args.batch_size = 64
        args.pool_size = 384
        args.top_m = 96
        args.flow_candidates = 384
        args.train_num_workers = max(12, int(args.train_num_workers))
        args.train_prefetch_factor = max(4, int(args.train_prefetch_factor))


if __name__ == "__main__":
    main()
