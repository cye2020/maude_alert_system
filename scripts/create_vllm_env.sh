#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_DIR="$ROOT_DIR/lib"

# Default path differs by environment:
# - Docker (Airflow image): /opt/vllm-env
# - Local workspace:        <repo>/.venvs/vllm-env
if [[ -f "/.dockerenv" ]]; then
  DEFAULT_VENV_PATH="/opt/vllm-env"
else
  DEFAULT_VENV_PATH="$ROOT_DIR/.venvs/vllm-env"
fi

VENV_PATH="${1:-$DEFAULT_VENV_PATH}"
# Normalize to absolute path so later subshell `cd` does not break resolution.
VENV_PATH="$(mkdir -p "$VENV_PATH" && cd "$VENV_PATH" && pwd)"
PYTHON_BIN="$VENV_PATH/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv command not found. Install uv first." >&2
  exit 1
fi

if [[ ! -f "$LIB_DIR/pyproject.toml" ]]; then
  echo "[ERROR] lib/pyproject.toml not found: $LIB_DIR/pyproject.toml" >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
mkdir -p "$UV_CACHE_DIR"

echo "[1/4] Creating venv: $VENV_PATH"
uv venv "$VENV_PATH" --python 3.11

echo "[2/4] Installing worker stack from lib/pyproject.toml"
(
  cd "$LIB_DIR"
  uv pip install --no-cache --python "$PYTHON_BIN" --index-strategy unsafe-best-match -e ".[worker]"
)

echo "[3/4] Installing extra package: polars"
uv pip install --no-cache --python "$PYTHON_BIN" polars

echo "[4/4] Verifying key packages"
uv pip show --python "$PYTHON_BIN" numpy numba cupy-cuda12x cudf-cu12 cuml-cu12 polars || true

echo "Done. Python: $PYTHON_BIN"
