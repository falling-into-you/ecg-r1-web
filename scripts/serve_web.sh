#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR"

CONDA_ACTIVATE_SCRIPT="$(python -c 'import config; print(config.CONDA_ACTIVATE_SCRIPT)')"
WEB_CONDA_ENV="$(python -c 'import config; print(config.WEB_CONDA_ENV)')"
if [[ -n "$WEB_CONDA_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_ACTIVATE_SCRIPT" "$WEB_CONDA_ENV"
fi

WEB_HOST="$(python -c 'import config; print(config.WEB_HOST)')"
WEB_PORT="$(python -c 'import config; print(config.WEB_PORT)')"
WEB_LOG_LEVEL="$(python -c 'import config; print(config.WEB_LOG_LEVEL)')"

exec uvicorn main:app \
  --host "$WEB_HOST" \
  --port "$WEB_PORT" \
  --log-level "$WEB_LOG_LEVEL"
