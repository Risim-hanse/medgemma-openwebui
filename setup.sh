#!/usr/bin/env bash
# One-shot setup: verify access, download weights, smoke test.
# Safe to re-run -- the HF cache is incremental.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_ID="${MEDGEMMA_MODEL_ID:-google/medgemma-4b-it}"
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}"

echo "==> Checking access to ${MODEL_ID}"
code=$(curl -s -o /dev/null -w "%{http_code}" -L \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/${MODEL_ID}/resolve/main/config.json")

if [[ "${code}" != "200" ]]; then
  cat <<EOF

BLOCKED (HTTP ${code})

Accept the license for this account at:
  https://huggingface.co/google/medgemma-4b-it

Click "Agree"/"Accept", then re-run this script.
Gated "auto" repos are usually approved within seconds; allow up to 1-2 days.

EOF
  exit 1
fi

echo "==> Access granted. Downloading weights (~8 GB)..."
./venv/bin/python - "$MODEL_ID" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], max_workers=4)
print("downloaded to:", path)
PY

echo "==> Running smoke test"
./venv/bin/python smoke_test.py

echo
echo "==> All good. Start the server with:"
echo "    ./venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000"