#!/usr/bin/env bash
set -euo pipefail

billing_start_unix=${DIAGNOSTIC_BILLING_START_UNIX:-$(date +%s)}
if [[ ! "$billing_start_unix" =~ ^[0-9]+$ ]] || [[ "$billing_start_unix" -le 0 ]]; then
  echo "DIAGNOSTIC_BILLING_START_UNIX must be a positive Unix timestamp" >&2
  exit 2
fi
cd /workspace/swarmstop
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run a dirty worktree" >&2
  exit 2
fi
runner_revision=$(git rev-parse --verify HEAD)
diagnostic_config=${DIAGNOSTIC_CONFIG:?missing DIAGNOSTIC_CONFIG}
run_label=${DIAGNOSTIC_RUN_LABEL:?missing DIAGNOSTIC_RUN_LABEL}
max_wall_seconds=${DIAGNOSTIC_MAX_WALL_SECONDS:-108000}
if [[ ! "$run_label" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "DIAGNOSTIC_RUN_LABEL must contain lowercase letters, digits, or hyphens" >&2
  exit 2
fi
if [[ ! -f "$diagnostic_config" ]]; then
  echo "diagnostic config does not exist: $diagnostic_config" >&2
  exit 2
fi

runtime_venv=/root/swarmstop-diagnostic-venv
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
if grep -q 'Qwen3.5-122B-A10B-GPTQ-Int4' "$diagnostic_config"; then
  python -m pip install --no-build-isolation 'gptqmodel==7.3.5'
fi
python -m pip install --no-deps \
  'torchvision==0.23.0' \
  --index-url https://download.pytorch.org/whl/cu128
python - <<'PY'
import torch
import torchvision
from transformers import AutoProcessor

if not torch.__version__.startswith("2.8."):
    raise RuntimeError(f"expected PyTorch 2.8.x, found {torch.__version__}")
if not torchvision.__version__.startswith("0.23."):
    raise RuntimeError(f"expected torchvision 0.23.x, found {torchvision.__version__}")
print(f"validated torch={torch.__version__} torchvision={torchvision.__version__}")
PY
python -m pip install -e . --no-deps
python scripts/fetch_upstreams.py --source
if [[ ! -s data/upstreams/agentabstain-data/tasks.jsonl ]]; then
  python scripts/fetch_upstreams.py --dataset
fi

export HF_HOME=/workspace/hf
output_root="/workspace/swarmstop-results/${run_label}-${runner_revision:0:12}"
session_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
session_dir="$output_root/sessions/$session_id"
mkdir -p "$HF_HOME" "$session_dir"
python - "$diagnostic_config" "$run_label" "$runner_revision" \
  "${RUNPOD_HOURLY_PRICE_USD:?missing RUNPOD_HOURLY_PRICE_USD}" \
  "${RUNPOD_POD_ID:-unknown}" "$billing_start_unix" <<'PY' > "$session_dir/launch_metadata.json"
import json
import sys
from datetime import UTC, datetime

config, run_label, revision, price, pod_id, billing_start = sys.argv[1:]
print(json.dumps({
    "billing_start_unix": int(billing_start),
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
  > "$session_dir/gpu.txt"
python -m pip freeze > "$session_dir/environment.txt"
python -m swarmstop validate --config "$diagnostic_config" \
  > "$session_dir/validation.json"
python scripts/validate_delegation_contact_tasks.py \
  > "$session_dir/task_audit.yaml"

set +e
timeout --signal=TERM --kill-after=120s 110000s \
  python scripts/run_delegation_contact.py \
    --config "$diagnostic_config" \
    --output-root "$output_root" \
    --hourly-price "${RUNPOD_HOURLY_PRICE_USD:?missing RUNPOD_HOURLY_PRICE_USD}" \
    --max-wall-seconds "$max_wall_seconds" \
    --billing-start-unix "$billing_start_unix" \
  2>&1 | tee "$session_dir/run.log"
run_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_status" > "$session_dir/exit_code.txt"
exit "$run_status"
