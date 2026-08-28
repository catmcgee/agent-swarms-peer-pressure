#!/usr/bin/env bash
set -euo pipefail

cd /workspace/swarmstop
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run a dirty worktree" >&2
  exit 2
fi
runner_revision=$(git rev-parse --verify HEAD)
behavior_config=${BEHAVIOR_CONFIG:?missing BEHAVIOR_CONFIG}
run_label=${BEHAVIOR_RUN_LABEL:?missing BEHAVIOR_RUN_LABEL}
max_wall_seconds=${BEHAVIOR_MAX_WALL_SECONDS:-42000}
if [[ ! "$run_label" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "BEHAVIOR_RUN_LABEL must contain lowercase letters, digits, or hyphens" >&2
  exit 2
fi
if [[ ! -f "$behavior_config" ]]; then
  echo "behavior config does not exist: $behavior_config" >&2
  exit 2
fi
runtime_venv=/root/swarmstop-behavior-venv
python -m venv --system-site-packages "$runtime_venv"
. "$runtime_venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  'accelerate==1.10.1' \
  'fastmcp==2.14.7' \
  'huggingface-hub==1.5.0' \
  'transformers==5.5.0' \
  'PyYAML==6.0.3' \
  'Pillow==12.3.0'
python -m pip install --no-deps \
  'torchvision==0.24.1' \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e . --no-deps
python scripts/fetch_upstreams.py --source
if [[ ! -s data/upstreams/agentabstain-data/tasks.jsonl ]]; then
  python scripts/fetch_upstreams.py --dataset
fi

export HF_HOME=/workspace/hf
output_root="/workspace/swarmstop-results/${run_label}-${runner_revision:0:12}"
mkdir -p "$HF_HOME" "$output_root"
python - "$behavior_config" "$run_label" "$runner_revision" \
  "${RUNPOD_HOURLY_PRICE_USD:?missing RUNPOD_HOURLY_PRICE_USD}" \
  "${RUNPOD_POD_ID:-unknown}" <<'PY' > "$output_root/launch_metadata.json"
import json
import sys
from datetime import UTC, datetime

config, run_label, revision, price, pod_id = sys.argv[1:]
print(json.dumps({
    "config": config,
    "created_at": datetime.now(UTC).isoformat(),
    "hourly_price_usd": float(price),
    "pod_id": pod_id,
    "run_label": run_label,
    "runner_revision": revision,
}, indent=2, sort_keys=True))
PY
nvidia-smi \
  --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader \
  > "$output_root/gpu.txt"
python -m pip freeze > "$output_root/environment.txt"
python -m swarmstop validate --config "$behavior_config" \
  > "$output_root/validation.json"
python scripts/audit_behavior_screen.py --config "$behavior_config" \
  > "$output_root/behavior_audit.json"

set +e
timeout --signal=TERM --kill-after=120s 43200s \
  python scripts/run_behavior_screen.py \
    --config "$behavior_config" \
    --output-root "$output_root" \
    --hourly-price "${RUNPOD_HOURLY_PRICE_USD:?missing RUNPOD_HOURLY_PRICE_USD}" \
    --max-wall-seconds "$max_wall_seconds" \
  2>&1 | tee "$output_root/run.log"
run_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_status" > "$output_root/exit_code.txt"
exit "$run_status"
