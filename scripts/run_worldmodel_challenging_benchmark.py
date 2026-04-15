from __future__ import annotations

import argparse
import json
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
from pdewm.utils.device import resolve_device
from pdewm.utils.git import get_git_commit_hash

# ---------------------------------------------------------------------------
# Challenging benchmark — designed for RTX 2070 Super (8 GB VRAM),
# 16 CPU cores and generous host RAM.
#
# Key differences vs the basic benchmark:
#   • Resolution  : 256 grid points (not 64)
#   • Data        : 64 train / 12 val / 12 test / 12 OOD trajectories, 48 time steps
#   • Models      : base_channels=32, latent_channels=64, channel_mults=[1,2,4,8]
#                   dynamics hidden=64, context=64, 4 blocks
#   • Ensemble    : 5 members (not 3)
#   • Training    : 300 AE epochs, 300 dynamics epochs, 200 fine-tune epochs
#   • Acquisition : 512 online transitions budget, 128 per round, rollout horizon 16
#   • Candidate pool : 256 proposals, top-64 screened
#   • Multi-seed  : supports --seeds flag for full statistical significance
#   • Both PDEs   : supports --pdes flag to run on Burgers and/or KS
#   • Batch size  : 32 (fits RTX 2070 Super well at 256 grid)
# ---------------------------------------------------------------------------

DEFAULT_REGIMES = ("frozen", "joint_no_ema", "joint_ema")
DEFAULT_STRATEGIES = (
    "offline_only",
    "random_states",
    "uncertainty_only",
    "diversity_only",
    "uncertainty_diversity",
    "ours",
)
DEFAULT_PDES = ("burgers_1d",)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Challenging world-model acquisition benchmark (GPU-targeted)."
    )

    # --- PDE & data ----------------------------------------------------------
    parser.add_argument("--pdes", nargs="+", default=list(DEFAULT_PDES),
                        help="PDE identifiers to benchmark (burgers_1d, ks_1d)")
    parser.add_argument("--data-root", default="data/generated_challenging")
    parser.add_argument("--output-root", default="artifacts/runs/challenging_benchmark")
    parser.add_argument("--data-version", default="challenging_v1")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--prepare-ae", action="store_true")

    # --- Resolution & dataset size -------------------------------------------
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=48)
    parser.add_argument("--train-trajectories", type=int, default=64)
    parser.add_argument("--val-trajectories", type=int, default=12)
    parser.add_argument("--test-trajectories", type=int, default=12)
    parser.add_argument("--ood-trajectories", type=int, default=12)

    # --- Model capacity (AE) -------------------------------------------------
    parser.add_argument("--ae-base-channels", type=int, default=32)
    parser.add_argument("--ae-latent-channels", type=int, default=64)
    parser.add_argument("--ae-channel-mults", nargs="+", type=int, default=[1, 2, 4, 8])

    # --- Model capacity (dynamics) --------------------------------------------
    parser.add_argument("--dyn-hidden-channels", type=int, default=64)
    parser.add_argument("--dyn-context-hidden", type=int, default=64)
    parser.add_argument("--dyn-context-output", type=int, default=64)
    parser.add_argument("--dyn-num-blocks", type=int, default=4)

    # --- Training budget ------------------------------------------------------
    parser.add_argument("--ae-epochs", type=int, default=300)
    parser.add_argument("--dynamics-epochs", type=int, default=300)
    parser.add_argument("--fine-tune-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)

    # --- Ensemble & seeds -----------------------------------------------------
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7],
                        help="Seeds for full reproducibility study (e.g. 7 42 123)")

    # --- Acquisition budget ---------------------------------------------------
    parser.add_argument("--online-solver-transitions", type=int, default=512)
    parser.add_argument("--transitions-per-round", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--top-m", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--diversity-lambda", type=float, default=0.2)

    # --- Strategy & regime selection ------------------------------------------
    parser.add_argument("--selected-regime", default="auto")
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))

    # --- W&B -----------------------------------------------------------------
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="pde-world-model")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-group", default="")

    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_pde_results: dict[str, dict[str, Any]] = {}

    for pde_id in args.pdes:
        for seed in args.seeds:
            pde_seed_tag = f"{pde_id}_seed{seed}"
            pde_output = output_root / pde_seed_tag
            pde_output.mkdir(parents=True, exist_ok=True)

            # --- Data generation ---
            dataset_root = (
                Path(args.data_root) / f"{pde_id}_offline" / args.data_version
            )
            if args.prepare_data or not dataset_root.exists():
                _run([
                    sys.executable, "scripts/generate_offline_data.py",
                    "--data-config", pde_id,
                    "--solver-config", pde_id,
                    f"data.dataset_version={args.data_version}",
                    f"data.output_dir={args.data_root}",
                    f"data.splits.train.num_trajectories={args.train_trajectories}",
                    f"data.splits.val.num_trajectories={args.val_trajectories}",
                    f"data.splits.test.num_trajectories={args.test_trajectories}",
                    f"data.splits.parameter_ood.num_trajectories={args.ood_trajectories}",
                    f"data.num_steps={args.num_steps}",
                    f"solver.grid_size={args.grid_size}",
                    f"project.seed={seed}",
                ])

            # --- Autoencoder warm-up ---
            ae_dir = pde_output / "warmup_autoencoder"
            ae_checkpoint = ae_dir / "best.pt"
            if args.prepare_ae or not ae_checkpoint.exists():
                ae_cmd = [
                    sys.executable, "scripts/train_autoencoder.py",
                    f"train.dataset_root={dataset_root}",
                    f"train.output_dir={ae_dir}",
                    f"train.epochs={args.ae_epochs}",
                    f"train.batch_size={args.batch_size}",
                    f"project.seed={seed}",
                    f"model.base_channels={args.ae_base_channels}",
                    f"model.latent_channels={args.ae_latent_channels}",
                    f"model.channel_multipliers=[{','.join(str(m) for m in args.ae_channel_mults)}]",
                ]
                ae_cmd.extend(_wandb_overrides(
                    enabled=args.wandb, project=args.wandb_project, entity=args.wandb_entity,
                    mode=args.wandb_mode,
                    group=args.wandb_group or f"challenging_{args.data_version}",
                    name=f"{pde_seed_tag}_warmup_ae",
                    tags=["challenging_benchmark", pde_id, "warmup_ae", f"seed_{seed}"],
                ))
                _run(ae_cmd)

            _, manifest = load_transition_records(dataset_root)
            offline_transitions = int(manifest["samples"])

            # --- Regime ablation ---
            regime_root = pde_output / "regime_ablation"
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
                    seed=seed,
                    batch_size=args.batch_size,
                    dyn_hidden=args.dyn_hidden_channels,
                    dyn_ctx_hidden=args.dyn_context_hidden,
                    dyn_ctx_output=args.dyn_context_output,
                    dyn_num_blocks=args.dyn_num_blocks,
                    wandb_args=args,
                    pde_id=pde_id,
                )
            regime_summary = {
                "regimes": regime_results,
                "selected_regime": _select_regime(regime_results, preferred=args.selected_regime),
            }
            (regime_root / "summary.json").write_text(
                json.dumps(regime_summary, indent=2), encoding="utf-8"
            )
            (regime_root / "summary.md").write_text(
                _format_regime_summary(regime_summary), encoding="utf-8"
            )

            selected_regime = str(regime_summary["selected_regime"])
            selected_regime_result = regime_results[selected_regime]

            # --- Acquisition benchmark ---
            acquisition_root = pde_output / "acquisition_benchmark"
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
                    batch_size=args.batch_size,
                    dyn_hidden=args.dyn_hidden_channels,
                    dyn_ctx_hidden=args.dyn_context_hidden,
                    dyn_ctx_output=args.dyn_context_output,
                    dyn_num_blocks=args.dyn_num_blocks,
                    seed=seed,
                    wandb_args=args,
                    pde_id=pde_id,
                )

            pde_benchmark = {
                "pde_id": pde_id,
                "seed": seed,
                "dataset_root": str(dataset_root),
                "offline_transitions": offline_transitions,
                "online_solver_transition_budget": int(args.online_solver_transitions),
                "selected_regime": selected_regime,
                "regime_summary": regime_summary,
                "strategy_results": strategy_results,
            }
            (pde_output / "benchmark_summary.json").write_text(
                json.dumps(pde_benchmark, indent=2), encoding="utf-8"
            )
            (pde_output / "benchmark_summary.md").write_text(
                _format_benchmark_summary(pde_benchmark), encoding="utf-8"
            )
            _plot_strategy_curves(strategy_results, pde_output)
            all_pde_results[pde_seed_tag] = pde_benchmark

    # --- Global summary ---
    (output_root / "full_results.json").write_text(
        json.dumps(all_pde_results, indent=2), encoding="utf-8"
    )
    _write_global_summary(all_pde_results, output_root)


# ---------------------------------------------------------------------------
# Regime ablation
# ---------------------------------------------------------------------------

def _run_regime_ablation(
    *,
    dataset_root: Path,
    output_dir: Path,
    ae_checkpoint: Path,
    regime: str,
    epochs: int,
    ensemble_size: int,
    seed: int,
    batch_size: int,
    dyn_hidden: int,
    dyn_ctx_hidden: int,
    dyn_ctx_output: int,
    dyn_num_blocks: int,
    wandb_args,
    pde_id: str,
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
                sys.executable, "scripts/train_dynamics.py",
                f"train.dataset_root={dataset_root}",
                f"train.ae_checkpoint={ae_checkpoint}",
                f"train.output_dir={member_dir}",
                f"train.epochs={epochs}",
                f"project.seed={seed + member_index}",
                f"train.regime={regime}",
                f"train.batch_size={batch_size}",
                "train.ae_loss_scale=1.0",
                "train.ema.decay=0.995",
                f"model.hidden_channels={dyn_hidden}",
                f"model.context_hidden_dim={dyn_ctx_hidden}",
                f"model.context_output_dim={dyn_ctx_output}",
                f"model.num_blocks={dyn_num_blocks}",
            ]
            command.extend(_wandb_overrides(
                enabled=wandb_args.wandb,
                project=wandb_args.wandb_project,
                entity=wandb_args.wandb_entity,
                mode=wandb_args.wandb_mode,
                group=wandb_args.wandb_group or f"challenging_{pde_id}_regime",
                name=f"{pde_id}_s{seed}_{regime}_m{member_index:02d}",
                tags=["challenging_benchmark", pde_id, "regime_ablation", regime, f"seed_{seed}"],
            ))
            _run(command)
        member_summaries.append(json.loads(member_summary_path.read_text(encoding="utf-8")))
        member_checkpoints.append(str(member_dir / "best.pt"))

    result = {
        "regime": regime,
        "ensemble_size": ensemble_size,
        "aggregate": _aggregate_ensemble_summaries(member_summaries),
        "members": member_summaries,
        "member_checkpoints": member_checkpoints,
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Acquisition benchmark
# ---------------------------------------------------------------------------

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
    batch_size: int,
    dyn_hidden: int,
    dyn_ctx_hidden: int,
    dyn_ctx_output: int,
    dyn_num_blocks: int,
    seed: int,
    wandb_args,
    pde_id: str,
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
    device_str = str(resolve_device("auto"))

    while cumulative_online_transitions < online_solver_transitions:
        remaining_budget = online_solver_transitions - cumulative_online_transitions
        round_budget = min(transitions_per_round, remaining_budget)
        round_dir = output_dir / f"round_{round_index + 1:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        members = load_world_model_committee(
            current_committee_checkpoints,
            str(ae_checkpoint),
            device=device_str,
        )
        records, manifest = load_transition_records(current_dataset_root)
        train_records = [r for r in records if r.metadata.split == "train"]
        memory_records, memory_latents = build_memory_bank(train_records, members)
        candidates = propose_candidates(
            memory_records, memory_latents, members,
            pool_size=pool_size, noise_std=noise_std,
            alpha=1.0, beta=0.2, gamma=0.5,
            max_abs_factor=2.0, seed=seed + round_index,
        )
        ordered_candidates = rank_candidates(
            candidates, strategy=strategy,
            top_m=top_m, diversity_lambda=diversity_lambda,
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
            batch_size=batch_size,
            dyn_hidden=dyn_hidden,
            dyn_ctx_hidden=dyn_ctx_hidden,
            dyn_ctx_output=dyn_ctx_output,
            dyn_num_blocks=dyn_num_blocks,
            wandb_args=wandb_args,
            strategy=strategy,
            round_index=round_index + 1,
            pde_id=pde_id,
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
    batch_size: int,
    dyn_hidden: int,
    dyn_ctx_hidden: int,
    dyn_ctx_output: int,
    dyn_num_blocks: int,
    wandb_args,
    strategy: str,
    round_index: int,
    pde_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    member_summaries = []
    member_checkpoints = []

    for member_index in range(ensemble_size):
        member_dir = output_dir / f"member_{member_index:02d}"
        member_summary_path = member_dir / "summary.json"
        if not member_summary_path.exists():
            command = [
                sys.executable, "scripts/train_dynamics.py",
                f"train.dataset_root={dataset_root}",
                f"train.ae_checkpoint={ae_checkpoint}",
                f"train.output_dir={member_dir}",
                f"train.epochs={epochs}",
                f"project.seed={seed + member_index}",
                f"train.regime={regime}",
                f"train.resume_checkpoint={resume_checkpoints[member_index]}",
                f"train.batch_size={batch_size}",
                "train.ae_loss_scale=1.0",
                "train.ema.decay=0.995",
                f"model.hidden_channels={dyn_hidden}",
                f"model.context_hidden_dim={dyn_ctx_hidden}",
                f"model.context_output_dim={dyn_ctx_output}",
                f"model.num_blocks={dyn_num_blocks}",
            ]
            command.extend(_wandb_overrides(
                enabled=wandb_args.wandb,
                project=wandb_args.wandb_project,
                entity=wandb_args.wandb_entity,
                mode=wandb_args.wandb_mode,
                group=wandb_args.wandb_group or f"challenging_{pde_id}_{strategy}",
                name=f"{pde_id}_{strategy}_r{round_index:02d}_m{member_index:02d}",
                tags=["challenging_benchmark", pde_id, "acquisition_benchmark", strategy, regime],
            ))
            _run(command)
        member_summaries.append(json.loads(member_summary_path.read_text(encoding="utf-8")))
        member_checkpoints.append(str(member_dir / "best.pt"))

    return {
        "aggregate": _aggregate_ensemble_summaries(member_summaries),
        "members": member_summaries,
        "member_checkpoints": member_checkpoints,
    }


# ---------------------------------------------------------------------------
# Helpers shared with the basic benchmark
# ---------------------------------------------------------------------------

def _aggregate_ensemble_summaries(member_summaries: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "best_val_loss_mean": _mean_nested(member_summaries, ("best_val_loss",)),
        "ae_val_loss_mean": _mean_nested(member_summaries, ("ae_val_metrics", "loss")),
        "ae_test_loss_mean": _mean_nested(member_summaries, ("ae_test_metrics", "loss")),
        "trajectory_val_one_step_rmse_mean": _mean_nested(
            member_summaries, ("trajectory_val_metrics", "one_step_rmse", "mean")),
        "trajectory_val_one_step_nrmse_mean": _mean_nested(
            member_summaries, ("trajectory_val_metrics", "one_step_nrmse", "mean")),
        "trajectory_val_rollout_rmse_mean": _mean_nested(
            member_summaries, ("trajectory_val_metrics", "rollout_rmse", "mean")),
        "trajectory_val_rollout_nrmse_mean": _mean_nested(
            member_summaries, ("trajectory_val_metrics", "rollout_nrmse", "mean")),
        "trajectory_test_one_step_rmse_mean": _mean_nested(
            member_summaries, ("trajectory_test_metrics", "one_step_rmse", "mean")),
        "trajectory_test_one_step_nrmse_mean": _mean_nested(
            member_summaries, ("trajectory_test_metrics", "one_step_nrmse", "mean")),
        "trajectory_test_rollout_rmse_mean": _mean_nested(
            member_summaries, ("trajectory_test_metrics", "rollout_rmse", "mean")),
        "trajectory_test_rollout_nrmse_mean": _mean_nested(
            member_summaries, ("trajectory_test_metrics", "rollout_nrmse", "mean")),
    }


def _select_regime(
    regime_results: dict[str, dict[str, Any]], *, preferred: str,
) -> str:
    if preferred != "auto":
        if preferred not in regime_results:
            raise ValueError(f"Requested regime {preferred!r} is not available.")
        return preferred
    ae_floor = min(r["aggregate"]["ae_val_loss_mean"] for r in regime_results.values())
    rollout_floor = min(
        r["aggregate"]["trajectory_val_rollout_nrmse_mean"] for r in regime_results.values()
    )
    best_regime, best_score = "", float("inf")
    for regime, result in regime_results.items():
        ae_ratio = result["aggregate"]["ae_val_loss_mean"] / max(ae_floor, 1e-8)
        rollout_ratio = result["aggregate"]["trajectory_val_rollout_nrmse_mean"] / max(rollout_floor, 1e-8)
        score = ae_ratio * rollout_ratio
        result["aggregate"]["selection_score"] = score
        if score < best_score:
            best_score, best_regime = score, regime
    if not best_regime:
        raise ValueError("Failed to select a training regime.")
    return best_regime


def _curve_point(
    *, aggregate: dict[str, Any], online_solver_transitions: int,
    offline_transitions: int, round_index: int,
) -> dict[str, Any]:
    return {
        "round_index": round_index,
        "online_solver_transitions": int(online_solver_transitions),
        "total_transitions": int(offline_transitions + online_solver_transitions),
        "trajectory_val_one_step_rmse_mean": float(aggregate["trajectory_val_one_step_rmse_mean"]),
        "trajectory_val_one_step_nrmse_mean": float(aggregate["trajectory_val_one_step_nrmse_mean"]),
        "trajectory_val_rollout_rmse_mean": float(aggregate["trajectory_val_rollout_rmse_mean"]),
        "trajectory_val_rollout_nrmse_mean": float(aggregate["trajectory_val_rollout_nrmse_mean"]),
        "trajectory_test_one_step_rmse_mean": float(aggregate["trajectory_test_one_step_rmse_mean"]),
        "trajectory_test_one_step_nrmse_mean": float(aggregate["trajectory_test_one_step_nrmse_mean"]),
        "trajectory_test_rollout_rmse_mean": float(aggregate["trajectory_test_rollout_rmse_mean"]),
        "trajectory_test_rollout_nrmse_mean": float(aggregate["trajectory_test_rollout_nrmse_mean"]),
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
    *, previous_records: list[TransitionRecord],
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_strategy_curves(strategy_results: dict[str, dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for x_key, title_suffix in [("online_solver_transitions", "Online"), ("total_transitions", "Total")]:
        for split, split_label in [("val", "Validation"), ("test", "Test")]:
            _plot_metric_group(
                strategy_results,
                output_root / f"{split}_nrmse_vs_{x_key}.png",
                x_key=x_key,
                one_step_key=f"trajectory_{split}_one_step_nrmse_mean",
                rollout_key=f"trajectory_{split}_rollout_nrmse_mean",
                title_prefix=f"{split_label} NRMSE vs {title_suffix} Transitions",
            )


def _plot_metric_group(
    strategy_results: dict[str, dict[str, Any]],
    figure_path: Path, *, x_key: str, one_step_key: str,
    rollout_key: str, title_prefix: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for strategy, result in strategy_results.items():
        xs = [p[x_key] for p in result["curve"]]
        axes[0].plot(xs, [p[one_step_key] for p in result["curve"]], marker="o", label=strategy)
        axes[1].plot(xs, [p[rollout_key] for p in result["curve"]], marker="o", label=strategy)
    axes[0].set_title(f"{title_prefix} — One-Step")
    axes[1].set_title(f"{title_prefix} — Rollout")
    for ax in axes:
        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel("NRMSE")
        ax.grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _format_regime_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Regime Ablation", "",
        "| Regime | Selection Score | AE Val Loss | Val Rollout NRMSE | Test Rollout NRMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    selected = str(summary["selected_regime"])
    for regime, result in summary["regimes"].items():
        a = result["aggregate"]
        marker = " <- selected" if regime == selected else ""
        lines.append(
            f"| {regime}{marker} | {float(a.get('selection_score', 0)):.6f} | "
            f"{float(a['ae_val_loss_mean']):.6f} | "
            f"{float(a['trajectory_val_rollout_nrmse_mean']):.6f} | "
            f"{float(a['trajectory_test_rollout_nrmse_mean']):.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_benchmark_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"# Challenging Benchmark — {summary['pde_id']} (seed {summary['seed']})", "",
        f"- selected regime: `{summary['selected_regime']}`",
        f"- offline transitions: `{summary['offline_transitions']}`",
        f"- online transition budget: `{summary['online_solver_transition_budget']}`", "",
        "| Strategy | Online Transitions | Val Rollout NRMSE | Test Rollout NRMSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for strategy, result in summary["strategy_results"].items():
        final = result["curve"][-1]
        lines.append(
            f"| {strategy} | {int(result['online_solver_transitions'])} | "
            f"{float(final['trajectory_val_rollout_nrmse_mean']):.6f} | "
            f"{float(final['trajectory_test_rollout_nrmse_mean']):.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_global_summary(all_results: dict[str, dict[str, Any]], output_root: Path) -> None:
    lines = ["# Challenging Benchmark — Global Summary", ""]
    for tag, result in all_results.items():
        lines.append(f"## {tag}")
        lines.append(f"- regime: `{result['selected_regime']}`")
        lines.append(f"- offline transitions: `{result['offline_transitions']}`")
        for strategy, sresult in result["strategy_results"].items():
            final = sresult["curve"][-1]
            lines.append(
                f"  - **{strategy}**: val rollout NRMSE "
                f"{float(final['trajectory_val_rollout_nrmse_mean']):.6f}, "
                f"test rollout NRMSE "
                f"{float(final['trajectory_test_rollout_nrmse_mean']):.6f}"
            )
        lines.append("")
    (output_root / "global_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _wandb_overrides(
    *, enabled: bool, project: str, entity: str,
    mode: str, group: str, name: str, tags: list[str],
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


if __name__ == "__main__":
    main()
