# Runbook

## Before a run

1. Confirm the working tree is clean.
2. Validate the experiment config.
3. Check that every task has a unique ID and a critical action.
4. Confirm peer boards are authentic or explicitly marked synthetic.
5. Confirm tools are sandboxed and network-free.
6. Record model endpoint, revision, sampling settings, and pricing date.
7. Estimate upper-bound cost and compare it with the run cap.
8. Run one deterministic smoke trial and one real endpoint trial.

## During a run

- Write one JSON object per completed trajectory.
- Flush after every record.
- Preserve endpoint error bodies without secrets.
- Resume from stable trial IDs.
- Do not edit a results file by hand.
- Stop if the deterministic authority oracle or sandbox state diverges from the task specification.

## After a run

1. Verify planned, completed, failed, and superseded counts.
2. Recompute scores from raw traces.
3. Recompute costs from recorded token usage.
4. Run specification checks without reading outcome aggregates.
5. Freeze the raw results directory read-only or copy it to immutable storage.
6. Record deviations in `docs/DECISIONS.md`.

## Dedicated GPU checklist

- Use one inference service for many independent agent contexts.
- Enable continuous batching and prefix caching.
- Mount only the required model and results volumes.
- Checkpoint results outside ephemeral storage.
- Set an automatic shutdown or spending limit.
- Terminate the instance after verifying results were copied.

### Audited behavior-only screen

`scripts/runpod_behavior_screen.sh` runs no lens code. It first executes 36 paired authorized
controls, then a 24-trajectory real-model canary, and only then completes the 216-trajectory
discovery matrix. The worker rejects token-count-mismatched board pairs, fingerprints all task,
board, configuration, dataset, source, model, and protocol inputs, flushes each trajectory, and
enforces both wall-clock and estimated GPU-cost caps. A passing discovery gate requires a fresh
behavioral confirmation before mechanistic extraction.

## Stage C context-capture checklist

1. Run `swarmstop lens-provenance --config configs/jr_lens.yaml` and require `verified: true`.
2. Verify the inference service is serving the pinned checkpoint; the recorded revision flag alone is not proof.
3. Use a fresh behavioral output directory and pass both `--model-revision` and `--anchor-snapshots`.
4. Confirm every completed trial has exactly four stable snapshot IDs.
5. Freeze the snapshot JSONL beside the behavioral trace before inspecting lens results.
6. Replay snapshots in length-batched offline forwards; do not request or retain hidden reasoning text.
7. Store only selected-position residuals and the provenance required to reproduce them.

### RunPod smoke command

The bounded GPU smoke crosses the two local sandbox tasks with stop versus proceed peers,
one seed, impossible feasibility, personal investment, and the eight-round cap. It produces
four behavioral trajectories, sixteen anchor snapshots, selected residuals, and 496 J/R/logit
readout records. On a fresh official PyTorch pod:

```bash
bash scripts/runpod_jr_smoke.sh
```

Use a 48 GB GPU, at least 35 GB of persistent workspace storage, and an automatic termination
deadline. Copy `results/jr-lens-runpod-smoke` off the pod before deletion.
