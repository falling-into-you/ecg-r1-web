#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export INFERENCE_BACKEND="${INFERENCE_BACKEND:-swift_rollout}"
export SWIFT_ROLLOUT_URL="${SWIFT_ROLLOUT_URL:-http://127.0.0.1:8023/infer/}"
export SWIFT_ROLLOUT_HEALTH_URL="${SWIFT_ROLLOUT_HEALTH_URL:-http://127.0.0.1:8023/health/}"

exec uvicorn main:app \
  --host "${WEB_HOST:-0.0.0.0}" \
  --port "${WEB_PORT:-8000}" \
  --log-level "${WEB_LOG_LEVEL:-info}"
