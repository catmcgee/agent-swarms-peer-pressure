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
max_cost_usd=${DIAGNOSTIC_MAX_COST_USD:-45}
hourly_price=${RUNPOD_HOURLY_PRICE_USD:?missing RUNPOD_HOURLY_PRICE_USD}
one_shot=${DIAGNOSTIC_ONE_SHOT:-0}
if [[ ! "$run_label" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "DIAGNOSTIC_RUN_LABEL must contain lowercase letters, digits, or hyphens" >&2
  exit 2
fi
if [[ ! -f "$diagnostic_config" ]]; then
  echo "diagnostic config does not exist: $diagnostic_config" >&2
  exit 2
fi
if [[ "$one_shot" != "0" && "$one_shot" != "1" ]]; then
  echo "DIAGNOSTIC_ONE_SHOT must be 0 or 1" >&2
  exit 2
fi
large_runtime=0
if grep -q 'Qwen3.5-122B-A10B-GPTQ-Int4' "$diagnostic_config"; then
  large_runtime=1
  if [[ "$run_label" != "delegation-contact-v1-122b-gptq-rerun-v2" ]]; then
    echo "the pinned 122B config requires the exact rerun-v2 label" >&2
    exit 2
  fi
  if [[ "$one_shot" != "1" ]]; then
    echo "the pinned 122B rerun requires DIAGNOSTIC_ONE_SHOT=1" >&2
    exit 2
  fi
fi
read -r remaining_seconds absolute_deadline_unix < <(
  python scripts/compute_run_deadline.py \
    "$billing_start_unix" "$max_wall_seconds" "$hourly_price" "$max_cost_usd"
)
if [[ "${DIAGNOSTIC_DEADLINE_GUARD:-0}" != "1" ]]; then
  export DIAGNOSTIC_BILLING_START_UNIX="$billing_start_unix"
  export DIAGNOSTIC_MAX_WALL_SECONDS="$max_wall_seconds"
  export DIAGNOSTIC_MAX_COST_USD="$max_cost_usd"
  export DIAGNOSTIC_DEADLINE_GUARD=1
  exec timeout --signal=KILL "${remaining_seconds}s" bash "$0" "$@"
fi
output_root="/workspace/swarmstop-results/${run_label}-${runner_revision:0:12}"
provider_pod_id=${RUNPOD_POD_ID:-}
if [[ "$one_shot" == "1" && -z "$provider_pod_id" ]]; then
  echo "RUNPOD_POD_ID is required for a one-shot run" >&2
  exit 2
fi
results_root=/workspace/swarmstop-results
mkdir -p "$results_root"
if [[ "$one_shot" == "1" ]]; then
  if ! mkdir "$output_root"; then
    echo "refusing to resume a one-shot result root: $output_root" >&2
    exit 2
  fi
else
  mkdir -p "$output_root"
fi
session_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
session_dir="$output_root/sessions/$session_id"
mkdir -p /workspace/hf "$output_root/sessions"
mkdir "$session_dir"
export HF_HOME=/workspace/hf
python - "$diagnostic_config" "$run_label" "$runner_revision" \
  "$hourly_price" "${provider_pod_id:-unknown}" "$billing_start_unix" \
  "$absolute_deadline_unix" "$max_cost_usd" "$max_wall_seconds" "$one_shot" \
  <<'PY' > "$session_dir/launch_metadata.json"
import json
import sys
from datetime import UTC, datetime

(
    config,
    run_label,
    revision,
    price,
    pod_id,
    billing_start,
    absolute_deadline,
    max_cost,
    max_wall,
    one_shot,
) = sys.argv[1:]
print(json.dumps({
    "absolute_deadline_unix": int(absolute_deadline),
    "billing_start_unix": int(billing_start),
    "config": config,
    "created_at": datetime.now(UTC).isoformat(),
    "hourly_price_usd": float(price),
    "max_cost_usd": float(max_cost),
    "max_wall_seconds": int(max_wall),
    "one_shot": one_shot == "1",
    "pod_id": pod_id,
    "run_label": run_label,
    "runner_revision": revision,
}, indent=2, sort_keys=True))
PY
nvidia-smi \
  --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader \
  > "$session_dir/gpu.txt"
trap 'status=$?; printf "%s\n" "$status" > "$session_dir/wrapper_exit_code.txt"' EXIT

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
if [[ "$large_runtime" == "1" ]]; then
  python -m pip install --no-build-isolation \
    'gptqmodel==7.3.5' \
    'optimum==2.3.0' \
    'torchao==0.16.0' \
    'transformers==5.16.1' \
    'accelerate==1.14.0' \
    'tokenizers==0.23.1'
fi
python -m pip install --no-deps \
  'torchvision==0.23.0' \
  --index-url https://download.pytorch.org/whl/cu128
python - "$large_runtime" <<'PY'
import sys

import accelerate
import torch
import tokenizers
import torchvision
import transformers
from transformers import AutoProcessor

if not torch.__version__.startswith("2.8."):
    raise RuntimeError(f"expected PyTorch 2.8.x, found {torch.__version__}")
if not torchvision.__version__.startswith("0.23."):
    raise RuntimeError(f"expected torchvision 0.23.x, found {torchvision.__version__}")
if sys.argv[1] == "1":
    import gptqmodel
    import torchao
    from gptqmodel import BACKEND, GPTQModel
    from importlib.metadata import version
    from optimum.gptq import GPTQQuantizer

    if BACKEND.GPTQ_MARLIN.value != "gptq_marlin" or not callable(GPTQModel.load):
        raise RuntimeError("native GPTQModel Marlin loader is unavailable")

    expected = {
        "accelerate": (accelerate.__version__, "1.14.0"),
        "gptqmodel": (gptqmodel.__version__, "7.3.5"),
        "optimum": (version("optimum"), "2.3.0"),
        "tokenizers": (tokenizers.__version__, "0.23.1"),
        "torchao": (torchao.__version__, "0.16.0"),
        "transformers": (transformers.__version__, "5.16.1"),
    }
else:
    expected = {
        "accelerate": (accelerate.__version__, "1.10.1"),
        "transformers": (transformers.__version__, "5.5.0"),
    }
for package, (actual, required) in expected.items():
    if actual != required:
        raise RuntimeError(f"expected {package} {required}, found {actual}")
print(f"validated torch={torch.__version__} torchvision={torchvision.__version__}")
PY
python -m pip install -e . --no-deps
python scripts/fetch_upstreams.py --source
if [[ ! -s data/upstreams/agentabstain-data/tasks.jsonl ]]; then
  python scripts/fetch_upstreams.py --dataset
fi

python -m pip freeze > "$session_dir/environment.txt"
python -m swarmstop validate --config "$diagnostic_config" \
  > "$session_dir/validation.json"
python scripts/validate_delegation_contact_tasks.py \
  > "$session_dir/task_audit.yaml"

set +e
one_shot_args=()
if [[ "$one_shot" == "1" ]]; then
  one_shot_args+=(--one-shot)
fi
python scripts/run_delegation_contact.py \
    --config "$diagnostic_config" \
    --run-label "$run_label" \
    --output-root "$output_root" \
    --hourly-price "$hourly_price" \
    --max-cost-usd "$max_cost_usd" \
    --max-wall-seconds "$max_wall_seconds" \
    --billing-start-unix "$billing_start_unix" \
    --provider-pod-id "${provider_pod_id:-unknown}" \
    "${one_shot_args[@]}" \
  2>&1 | tee "$session_dir/run.log"
run_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_status" > "$session_dir/exit_code.txt"
exit "$run_status"
