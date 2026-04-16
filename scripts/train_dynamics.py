from __future__ import annotations

import argparse

from pdewm.training.dynamics import train_latent_dynamics
from pdewm.utils.config import load_named_config
from pdewm.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides in key=value form.")
    args = parser.parse_args()

    cfg = load_named_config("train_dynamics", overrides=list(args.overrides))
    seed_everything(int(cfg.project.seed))
    result = train_latent_dynamics(cfg)
    print(f"regime={result['regime']}")
    print(f"final_epoch={result['final_epoch']}")
    print(f"final_eval_loss={result['final_eval_loss']:.6f}")
    print(f"eval_one_step={result['eval_metrics']['phys_1step']:.6f}")
    print(f"eval_rollout={result['eval_metrics']['rollout']:.6f}")
    print(
        "trajectory_eval_rollout_nrmse="
        f"{result['trajectory_eval_metrics']['rollout_nrmse']['mean']:.6f}"
    )


if __name__ == "__main__":
    main()
