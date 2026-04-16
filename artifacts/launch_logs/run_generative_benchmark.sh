#!/usr/bin/env bash
set -euo pipefail
cd /home/cesarpi-ext/pde_world_model
export PYTHONUNBUFFERED=1
while tmux has-session -t challenging_benchmark 2>/dev/null; do
  sleep 60
done
set +e
uv run python scripts/run_worldmodel_generative_benchmark.py \
  --output-root artifacts/runs/generative_benchmark \
  --wandb --prepare-data --prepare-ae 2>&1 | tee artifacts/launch_logs/generative_benchmark.log
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > artifacts/launch_logs/generative_benchmark.exit
touch artifacts/launch_logs/generative_benchmark.done
exit "$status"
