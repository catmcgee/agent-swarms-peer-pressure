#!/usr/bin/env bash
set -euo pipefail

cd /workspace/swarmstop
python -m venv --system-site-packages .venv-gpu
. .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gpu]'

export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME" results/jr-lens-runpod-smoke

python -m swarmstop lens-provenance --config configs/jr_lens.yaml
python scripts/run_jr_smoke.py 2>&1 | tee results/jr-lens-runpod-smoke/run.log
