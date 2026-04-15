#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/launches
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="artifacts/launches/long_campaign_${STAMP}.log"
PID_PATH="artifacts/launches/long_campaign_${STAMP}.pid"

nohup uv run python scripts/run_long_campaign.py "$@" >"${LOG_PATH}" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${PID_PATH}"
echo "pid=${PID}"
echo "log=${LOG_PATH}"
echo "pid_file=${PID_PATH}"
