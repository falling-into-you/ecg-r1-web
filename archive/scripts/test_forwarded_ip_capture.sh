#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:44000}"
FAKE_CLIENT_IP="${FAKE_CLIENT_IP:-8.8.8.8}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

printf 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==' | base64 -d > "$tmp_dir/dummy.png"

echo "BASE_URL=$BASE_URL"
echo "FAKE_CLIENT_IP=$FAKE_CLIENT_IP"

resp="$(curl -sS -X POST "$BASE_URL/predict" \
  -H "X-Forwarded-For: ${FAKE_CLIENT_IP}, 127.0.0.1" \
  -H "X-Real-IP: ${FAKE_CLIENT_IP}" \
  -F "image=@$tmp_dir/dummy.png;type=image/png")"

req_id="$(python -c 'import json,sys; print(json.loads(sys.stdin.read()).get("request_id",""))' <<<"$resp" 2>/dev/null || true)"

if [[ -z "$req_id" ]]; then
  echo "predict did not return request_id; falling back to /admin/whoami (model may be loading)"
  WHOAMI_URL="${WHOAMI_URL:-http://127.0.0.1:8000/admin/whoami}"
  whoami="$(curl -sS -i "$WHOAMI_URL" \
    -H "X-Forwarded-For: ${FAKE_CLIENT_IP}, 127.0.0.1" \
    -H "X-Real-IP: ${FAKE_CLIENT_IP}")"
  echo "$whoami" | sed -n '1,25p'
  exit 0
fi

date="${req_id:0:8}"
date="${date:0:4}-${date:4:2}-${date:6:2}"
path="data_collection/${date}/${req_id}/data.json"

echo "request_id=$req_id"
echo "record_path=$path"

python - <<PY
import json
from pathlib import Path
p = Path("$path")
obj = json.loads(p.read_text(encoding="utf-8"))
client = obj.get("client") or {}
print("stored client.ip =", client.get("ip"))
print("stored client.geo =", client.get("geo"))
PY
