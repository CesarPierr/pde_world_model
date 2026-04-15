from __future__ import annotations

import argparse

from pdewm.training.baselines import train_baseline
from pdewm.utils.config import load_named_config
from pdewm.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="cnn_ar_1d")
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides in key=value form.")
    args = parser.parse_args()

    cfg = load_named_config(
        "train_baseline",
        overrides=list(args.overrides),
        defaults_overrides={"model": args.model_config},
    )
    seed_everything(int(cfg.project.seed))
    summary = train_baseline(cfg)
    print(f"model={summary['model_name']}")
    print(f"best_val_loss={summary['best_val_loss']:.6f}")
    print(f"test_one_step={summary['test_metrics']['phys_1step']:.6f}")
    print(f"test_rollout={summary['test_metrics']['rollout']:.6f}")
    print(
        "trajectory_val_rollout_nrmse="
        f"{summary['trajectory_val_metrics']['rollout_nrmse']['mean']:.6f}"
    )
    print(
        "trajectory_test_rollout_nrmse="
        f"{summary['trajectory_test_metrics']['rollout_nrmse']['mean']:.6f}"
    )


if __name__ == "__main__":
    main()
