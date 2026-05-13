#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${MODEL_PATH:?Set MODEL_PATH to the ECG-R1 checkpoint directory before starting rollout.}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"
export ECG_SEQ_LENGTH="${ECG_SEQ_LENGTH:-5000}"
export ECG_PATCH_SIZE="${ECG_PATCH_SIZE:-50}"
export ROOT_ECG_DIR="${ROOT_ECG_DIR:-/}"
export ROOT_IMAGE_DIR="${ROOT_IMAGE_DIR:-/}"
export ECG_TOWER_PATH="${ECG_TOWER_PATH:-$ROOT_DIR/checkpoints/cpt_wfep_epoch_20.pt}"
export ECG_PROJECTOR_TYPE="${ECG_PROJECTOR_TYPE:-mlp2x_gelu}"
export ECG_MODEL_CONFIG="${ECG_MODEL_CONFIG:-coca_ViT-B-32}"
export FREEZE_ECG_TOWER="${FREEZE_ECG_TOWER:-True}"
export FREEZE_ECG_PROJECTOR="${FREEZE_ECG_PROJECTOR:-True}"

exec swift rollout \
  --model "$MODEL_PATH" \
  --custom_register "$ROOT_DIR/ecg_r1_runtime/swift_register.py" \
  --vllm_data_parallel_size "${VLLM_DATA_PARALLEL_SIZE:-1}" \
  --port "${SWIFT_ROLLOUT_PORT:-8023}"
