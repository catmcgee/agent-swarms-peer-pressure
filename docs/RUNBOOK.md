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

`scripts/runpod_behavior_screen.sh` runs no lens code. It executes a 12-response recognition-only
preflight, 36 paired authorized controls, and a separate 24-trajectory action canary before it can
start the 216-trajectory discovery matrix. Each stage has a distinct fingerprint and output
directory. The worker rejects token-count-mismatched board pairs, fingerprints all task, board,
configuration, dataset, source, model, and protocol inputs, flushes each trajectory, and enforces
both wall-clock and estimated GPU-cost caps. The RunPod instance must additionally be created with
provider-side automatic termination no later than 13 hours after creation, so a detached or failed
worker cannot continue billing. A passing discovery gate requires a fresh behavioral confirmation
before mechanistic extraction.

The worker refuses tracked changes, keys its output directory by the committed runner revision,
and fetches the pinned AgentAbstain source and dataset before validation. Launch 9B on an explicit
volume of at least 40 GB. Set provider auto-stop before the calculated experiment cost cap and a
later auto-termination fallback, monitor the control-plane state, copy the commit-keyed result tree
off the pod with hashes, and only then delete that exact experiment pod. Do not rely on automatic
termination to preserve `/workspace`; pod-volume data is deleted with the pod.

For protocol-v1.3 runs, set `BEHAVIOR_CONFIG` and a filesystem-safe `BEHAVIOR_RUN_LABEL`
explicitly. The 9B run uses `configs/behavior_screen_v13_9b.yaml` on a 48 GB GPU. The unquantized
27B run uses `configs/behavior_screen_v13_27b.yaml` on one GPU with at least 80 GB and a workspace
of at least 100 GB for its roughly 56 GB checkpoint plus results and installation overhead. Transfer the
pinned `data/upstreams/agentabstain-data` tree before launch; the shell verifies it and downloads it
only when absent. The shell derives the scientific cost cap from the selected config, while
`RUNPOD_HOURLY_PRICE_USD` records the actual provisioned rate.

For each model in the ladder:

1. Save the provider's pod JSON immediately after creation, including the exact ID, name, GPU,
   image, hourly price, storage, and automatic stop/termination deadlines. Keep an allow-list
   containing only this newly returned pod ID.
2. Transfer a real Git checkout at the committed full SHA and the pinned AgentAbstain dataset.
   Require a completely clean worktree before launch.
3. Record the full SHA, selected config, run label, actual hourly price, `nvidia-smi` identity and
   memory, installed packages, validation output, and dedicated behavior-audit output in the
   commit-keyed result root before model loading.
4. Monitor phase counts and terminal state without inspecting condition outcomes mid-run. Treat
   every model-behavior gate as fail-closed; never selectively rerun a completed trial.
5. At a terminal state, generate a relative SHA-256 manifest on the pod, copy the entire result
   root and provider metadata off-pod, and verify every hash locally. Independently recompute phase
   counts, recognition fidelity, condition rates, matched contrasts, and cost from raw records.
6. Confirm the exact allow-listed pod ID and expected SwarmStop name, then delete only that pod.
   Confirm a subsequent lookup returns not found. Never select a pod by list position or touch an
   unrelated running pod.
7. Preserve completed and clean model-gate-abort bundles unchanged. Apply the preregistered ladder
   transition rule before provisioning the next model.

### Delegation/contact diagnostic

`scripts/runpod_delegation_contact.sh` runs the held-out diagnostic specified in
`docs/DELEGATION_CONTACT_PROTOCOL.md`. It revalidates the deterministic 24-task
selection on the worker, checks all 864 generated board records against the
exact tokenizer, runs the 24-response recognition gate, shuffles the 1,008
initial trajectories in a frozen order, and applies the treatment-blind event
yield rule before any optional extension. Use the dedicated runner; the generic
CLI deliberately refuses this source-crossed config.

For 27B, use one secure-cloud GPU with at least 80 GB, at least 120 GB of
workspace storage, a hard provider termination deadline beyond the 30-hour
worker limit, and the actual hourly price in `RUNPOD_HOURLY_PRICE_USD`. Preserve
and verify the complete commit-keyed result tree off-pod before deleting the
exact allow-listed pod.

The separate larger-model config is
`configs/delegation_contact_v1_122b_gptq.yaml`. Use a single GPU with enough
memory for the pinned 78.9 GB checkpoint plus runtime overhead; do not assume an
80 GB card is sufficient. Its first seed is a treatment-blind cost calibration
and continuation gate implemented by the dedicated runner. The worker installs
the pinned GPT-QModel backend only for this config. Launch it only after the 27B
run is terminal.

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
