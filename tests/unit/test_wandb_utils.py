from __future__ import annotations

from omegaconf import OmegaConf

from pdewm.utils.wandb import WandbRunHandle, flatten_metrics, init_wandb_run


def test_flatten_metrics_nests_with_prefix() -> None:
    flattened = flatten_metrics("val", {"loss": 1.0, "sub": {"metric": 2.0}})
    assert flattened == {"val/loss": 1.0, "val/sub/metric": 2.0}


def test_init_wandb_run_returns_noop_when_disabled() -> None:
    cfg = OmegaConf.create(
        {
            "project": {"name": "pde-world-model"},
            "logging": {"wandb": {"enabled": False}},
        }
    )
    handle = init_wandb_run(
        cfg,
        default_name="unit-test",
        default_group="tests",
        default_job_type="pytest",
    )
    assert isinstance(handle, WandbRunHandle)
    assert not handle.enabled
