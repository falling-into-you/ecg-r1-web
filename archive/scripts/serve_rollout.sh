#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "archive/scripts/serve_rollout.sh is kept for compatibility. Starting direct vLLM inference instead." >&2
exec bash "$ROOT_DIR/scripts/serve_vllm.sh"
