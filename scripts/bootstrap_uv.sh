#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

uv python install "$PYTHON_VERSION"
uv sync \
  --managed-python \
  --python "$PYTHON_VERSION" \
  --group dev \
  "$@"

if [[ "$TORCH_BACKEND" != "auto" ]]; then
  uv pip install \
    --python .venv/bin/python \
    --editable . \
    --group dev \
    --exact \
    --reinstall \
    --torch-backend "$TORCH_BACKEND"
fi
