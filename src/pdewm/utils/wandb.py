from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _load_wandb_module() -> Any | None:
    try:
        import wandb
    except ImportError:
        return None
    return wandb


@dataclass(slots=True)
class WandbRunHandle:
    run: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if self.run is None:
            return
        payload = {key: value for key, value in metrics.items() if value is not None}
        if not payload:
            return
        self.run.log(payload, step=step)

    def update_summary(self, metrics: dict[str, Any]) -> None:
        if self.run is None:
            return
        for key, value in metrics.items():
            self.run.summary[key] = value

    def save_file(self, path: str | Path) -> None:
        if self.run is None:
            return
        path = Path(path)
        if not path.exists():
            return
        self.run.save(str(path), policy="now")

    def finish(self) -> None:
        if self.run is None:
            return
        self.run.finish()


def init_wandb_run(
    cfg: DictConfig,
    *,
    default_name: str,
    default_group: str | None,
    default_job_type: str,
    extra_tags: list[str] | None = None,
) -> WandbRunHandle:
    wandb_cfg = OmegaConf.select(cfg, "logging.wandb")
    if not wandb_cfg or not bool(wandb_cfg.enabled):
        return WandbRunHandle()

    wandb = _load_wandb_module()
    if wandb is None:
        print("wandb logging requested but the package is not installed in the active environment.")
        return WandbRunHandle()

    run_name = _maybe_string(wandb_cfg.name) or default_name
    group = _maybe_string(wandb_cfg.group) or default_group
    job_type = _maybe_string(wandb_cfg.job_type) or default_job_type
    tags = list(wandb_cfg.tags or [])
    if extra_tags:
        tags.extend(extra_tags)
    run_dir = Path(_maybe_string(wandb_cfg.dir) or "artifacts/wandb")
    run_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=_maybe_string(wandb_cfg.project) or _maybe_string(OmegaConf.select(cfg, "project.name")) or "pde-world-model",
        entity=_maybe_string(wandb_cfg.entity),
        mode=_maybe_string(wandb_cfg.mode) or "online",
        group=group,
        name=run_name,
        job_type=job_type,
        notes=_maybe_string(wandb_cfg.notes),
        tags=tags or None,
        dir=str(run_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return WandbRunHandle(run=run)


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flattened.update(flatten_metrics(f"{prefix}/{key}", value))
        else:
            flattened[f"{prefix}/{key}"] = value
    return flattened


def _maybe_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
