#!/usr/bin/env bash
set -euo pipefail

cd /workspace/swarmstop
runtime_venv=/root/swarmstop-venv
python -m venv --system-site-packages "$runtime_venv"
. "$runtime_venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  'accelerate>=1.10.0' \
  'huggingface-hub>=0.30.0' \
  'transformers>=5.5.0' \
  'PyYAML>=6.0.2'
python -m pip install -e . --no-deps

export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME" results/jr-lens-runpod-smoke

python -m swarmstop lens-provenance --config configs/jr_lens.yaml
python scripts/run_jr_smoke.py 2>&1 | tee results/jr-lens-runpod-smoke/run.log
