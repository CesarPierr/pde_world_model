from __future__ import annotations

import argparse

from pdewm.training.autoencoder import train_autoencoder
from pdewm.utils.config import load_named_config
from pdewm.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides in key=value form.")
    args = parser.parse_args()

    cfg = load_named_config("train_autoencoder", overrides=list(args.overrides))
    seed_everything(int(cfg.project.seed))
    result = train_autoencoder(cfg)
    print(f"final_epoch={result['final_epoch']}")
    print(f"final_eval_loss={result['final_eval_loss']:.6f}")


if __name__ == "__main__":
    main()
