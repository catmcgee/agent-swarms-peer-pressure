#!/usr/bin/env bash
set -euo pipefail

cd /workspace/swarmstop
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "refusing to run a dirty tracked worktree" >&2
  exit 2
fi
runner_revision=$(git rev-parse --verify HEAD)
runtime_venv=/root/swarmstop-behavior-venv
python -m venv --system-site-packages "$runtime_venv"
. "$runtime_venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  'accelerate==1.10.1' \
  'fastmcp==2.14.7' \
  'huggingface-hub==1.5.0' \
  'transformers==5.5.0' \
  'PyYAML==6.0.3'
python -m pip install -e . --no-deps
python scripts/fetch_upstreams.py --source --dataset

export HF_HOME=/workspace/hf
output_root="/workspace/swarmstop-results/behavior-screen-qwen35-9b-v12-${runner_revision:0:12}"
mkdir -p "$HF_HOME" "$output_root"
python -m pip freeze > "$output_root/environment.txt"
python -m swarmstop validate --config configs/behavior_screen.yaml \
  > "$output_root/validation.json"

set +e
timeout --signal=TERM --kill-after=120s 43200s \
  python scripts/run_behavior_screen.py \
    --config configs/behavior_screen.yaml \
    --output-root "$output_root" \
    --hourly-price "${RUNPOD_HOURLY_PRICE_USD:?missing RUNPOD_HOURLY_PRICE_USD}" \
    --max-wall-seconds 42000 \
    --max-cost-usd 8 \
  2>&1 | tee "$output_root/run.log"
run_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_status" > "$output_root/exit_code.txt"
exit "$run_status"
