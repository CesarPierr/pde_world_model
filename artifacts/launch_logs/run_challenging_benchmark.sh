#!/usr/bin/env bash
set -euo pipefail
cd /home/cesarpi-ext/pde_world_model
export PYTHONUNBUFFERED=1
set +e
uv run python scripts/run_worldmodel_challenging_benchmark.py \
  --pdes burgers_1d \
  --seeds 7 42 123 \
  --wandb --prepare-data 2>&1 | tee artifacts/launch_logs/challenging_benchmark.log
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > artifacts/launch_logs/challenging_benchmark.exit
touch artifacts/launch_logs/challenging_benchmark.done
exit "$status"
