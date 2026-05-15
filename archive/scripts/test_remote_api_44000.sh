#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:44000}"
TIMEOUT_STATUS="${TIMEOUT_STATUS:-5}"
TIMEOUT_PREDICT="${TIMEOUT_PREDICT:-180}"
TIMEOUT_STREAM="${TIMEOUT_STREAM:-180}"
MAX_STREAM_LINES="${MAX_STREAM_LINES:-160}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

printf 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==' | base64 -d > "$tmp_dir/dummy.png"

echo "BASE_URL=$BASE_URL"
echo "TIMEOUT_STATUS=${TIMEOUT_STATUS}s TIMEOUT_PREDICT=${TIMEOUT_PREDICT}s TIMEOUT_STREAM=${TIMEOUT_STREAM}s MAX_STREAM_LINES=$MAX_STREAM_LINES"

echo
echo "== GET /status =="
timeout "${TIMEOUT_STATUS}s" curl -sS -i "$BASE_URL/status" | head -n 30

echo
echo "== GET / (should be 404 for API-only) =="
timeout "${TIMEOUT_STATUS}s" curl -sS -i "$BASE_URL/" | head -n 30

echo
echo "== POST /predict (JSON) =="
set +e
timeout "${TIMEOUT_PREDICT}s" curl -sS -i \
  -F "image=@$tmp_dir/dummy.png;type=image/png" \
  "$BASE_URL/predict" | head -n 80
predict_rc="${PIPESTATUS[0]}"
set -e
if [[ "$predict_rc" == "124" || "$predict_rc" == "143" ]]; then
  echo "predict request timed out after ${TIMEOUT_PREDICT}s (rc=$predict_rc). Increase TIMEOUT_PREDICT if model is slow to start."
elif [[ "$predict_rc" != "0" ]]; then
  exit "$predict_rc"
fi

echo
echo "== POST /predict_stream (SSE) =="
start_ts="$(date +%s)"
set +e
timeout "${TIMEOUT_STREAM}s" curl -sS -i -N \
  -F "image=@$tmp_dir/dummy.png;type=image/png" \
  "$BASE_URL/predict_stream" | awk -v start="$start_ts" '{ printf("[+%ds] %s\n", systime()-start, $0); fflush(); }' | head -n "$MAX_STREAM_LINES"
curl_rc="${PIPESTATUS[0]}"
set -e
if [[ "$curl_rc" == "124" || "$curl_rc" == "143" ]]; then
  echo "stream request timed out after ${TIMEOUT_STREAM}s (rc=$curl_rc). Increase TIMEOUT_STREAM if needed."
elif [[ "$curl_rc" != "0" && "$curl_rc" != "23" ]]; then
  exit "$curl_rc"
fi
