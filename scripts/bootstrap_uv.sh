#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${TORCH_BACKEND:-cpu}"

case "$TORCH_BACKEND" in
  cpu|auto|cu124|cu126|cu128)
    ;;
  *)
    echo "Unsupported TORCH_BACKEND=$TORCH_BACKEND" >&2
    echo "Expected one of: cpu, auto, cu124, cu126, cu128" >&2
    exit 1
    ;;
esac

uv python install "$PYTHON_VERSION"
uv sync --managed-python --python "$PYTHON_VERSION" --group dev "$@"

if [[ "$TORCH_BACKEND" != "cpu" ]]; then
  uv pip install \
    --python .venv/bin/python \
    --reinstall \
    --torch-backend "$TORCH_BACKEND" \
    "torch>=2.5"
fi
