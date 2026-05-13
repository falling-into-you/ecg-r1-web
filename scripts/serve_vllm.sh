#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR"

CONDA_ACTIVATE_SCRIPT="$(python -c 'import config; print(config.CONDA_ACTIVATE_SCRIPT)')"
VLLM_CONDA_ENV="$(python -c 'import config; print(config.VLLM_CONDA_ENV)')"
if [[ -n "$VLLM_CONDA_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_ACTIVATE_SCRIPT" "$VLLM_CONDA_ENV"
fi

eval "$(
python - <<'PY'
import shlex

import config

for key, value in config.RUNTIME_ENV_VARS.items():
    print(f"export {key}={shlex.quote(str(value))}")

print(f"MODEL_PATH={shlex.quote(config.MODEL_PATH)}")
print(f"export VLLM_LOAD_FORMAT={shlex.quote(str(config.VLLM_LOAD_FORMAT))}")
PY
)"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is empty. Set MODEL_PATH in config.py before starting vLLM." >&2
  exit 1
fi

exec python -m ecg_r1_runtime.serve_vllm
