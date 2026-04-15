from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from pdewm.utils.git import repo_root


def load_named_config(
    config_name: str,
    overrides: list[str] | None = None,
    defaults_overrides: dict[str, str] | None = None,
) -> DictConfig:
    config_root = repo_root() / "configs"
    root_cfg = OmegaConf.load(config_root / f"{config_name}.yaml")

    fragments: list[DictConfig] = []
    defaults = root_cfg.pop("defaults", [])
    for entry in defaults:
        if entry == "_self_":
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(f"Unsupported defaults entry: {entry!r}")
        key, value = next(iter(entry.items()))
        if defaults_overrides and key in defaults_overrides:
            value = defaults_overrides[key]
        fragments.append(OmegaConf.create({key: OmegaConf.load(config_root / key / f"{value}.yaml")}))

    fragments.append(root_cfg)
    merged = OmegaConf.merge(*fragments)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(merged)
    return merged


def config_to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)
