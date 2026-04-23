#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_PATH="/opt/conda/envs/recon-serve-py312"
export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

PORT="${RECON_SERVE_PORT:-${VGGT_SERVE_PORT:-8000}}"
HOST="${RECON_SERVE_HOST:-${VGGT_SERVE_HOST:-0.0.0.0}}"

exec python -m uvicorn vggt_serve.app:app --host "${HOST}" --port "${PORT}"
