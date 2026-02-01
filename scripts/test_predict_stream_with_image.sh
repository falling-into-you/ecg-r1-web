#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:44000}"
IMAGE_PATH="${2:-/data/jinjiarui/run/ecg-r1-web/scripts/47099212.png}"
TIMEOUT_STREAM="${TIMEOUT_STREAM:-180}"
MAX_STREAM_LINES="${MAX_STREAM_LINES:-240}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

if [[ ! -f "$IMAGE_PATH" ]]; then
  IMAGE_PATH="$tmp_dir/dummy.png"
  python - <<'PY'
import base64
from pathlib import Path
p = Path("/tmp/dummy.png")
p.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="))
print("wrote", p)
PY
  IMAGE_PATH="/tmp/dummy.png"
fi

echo "BASE_URL=$BASE_URL"
echo "IMAGE_PATH=$IMAGE_PATH"
echo "TIMEOUT_STREAM=${TIMEOUT_STREAM}s MAX_STREAM_LINES=$MAX_STREAM_LINES"

start_ts="$(date +%s)"
set +e
timeout "${TIMEOUT_STREAM}s" env IMAGE_PATH="$IMAGE_PATH" curl -sS -i -N \
  -F "image=@$IMAGE_PATH;type=image/png" \
  "$BASE_URL/predict_stream" | awk -v start="$start_ts" '{ printf("[+%ds] %s\n", systime()-start, $0); fflush(); }' | head -n "$MAX_STREAM_LINES"
curl_rc="${PIPESTATUS[0]}"
set -e
if [[ "$curl_rc" == "124" || "$curl_rc" == "143" ]]; then
  echo "stream request timed out after ${TIMEOUT_STREAM}s (rc=$curl_rc). Increase TIMEOUT_STREAM if needed."
elif [[ "$curl_rc" != "0" && "$curl_rc" != "23" ]]; then
  exit "$curl_rc"
fi
