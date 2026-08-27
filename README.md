# SwarmStop

SwarmStop studies whether tool-using agents cross an authorization boundary after they have explicitly recognized it, and whether equal-status peer agents make that reversal more likely.

The benchmark extends executable abstention tasks with a shared message board, explicit authority state, controlled peer trajectories, and a smaller live-swarm replication. All consequential actions run in deterministic local sandboxes.

## Research question

> After an agent records that an action is unauthorized, does observing peer agents continue make it more likely to perform or delegate that action?

The controlled experiment crosses:

- authorized-path feasibility: feasible or impossible;
- investment: inherited progress or personally completed progress;
- peer norm: absent, peers stop, or peers proceed;
- remaining action budget: two or eight post-recognition rounds.

The primary outcome is an unauthorized critical-action attempt after the recognition checkpoint. Delegating the action to a peer counts as an attempt.

## Repository status

The repository currently contains:

- a frozen draft protocol and analysis plan;
- typed schemas for tasks, conditions, peer boards, events, and results;
- a deterministic authority evaluator and fail-open sandbox;
- a controlled yoked-peer runner;
- an OpenAI-compatible inference client and deterministic test policy;
- adapters for AgentAbstain task records and AuthorityBench scenarios;
- run manifests, resumable JSONL output, scoring, and cost estimation;
- unit and end-to-end smoke tests.

No empirical claims should be made from the included synthetic smoke tasks. They exist only to validate the machinery.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

swarmstop validate --config configs/smoke.yaml
swarmstop run --config configs/smoke.yaml
swarmstop summarize --results results/smoke/trials.jsonl
pytest
```

To call a hosted or self-hosted OpenAI-compatible endpoint:

```bash
cp .env.example .env
export MODEL_API_KEY=...
export MODEL_BASE_URL=https://example.invalid/v1

swarmstop run \
  --config configs/pilot.yaml \
  --provider openai-compatible \
  --model your-model-id
```

The runner never loads `.env` automatically. Export credentials in the shell or use a secrets manager.

## Upstream benchmarks

Fetch pinned upstream source trees and the AgentAbstain dataset with:

```bash
python scripts/fetch_upstreams.py --source
python scripts/fetch_upstreams.py --dataset
```

Downloaded material lives under `data/upstreams/` and is excluded from version control. See [THIRD_PARTY.md](THIRD_PARTY.md) for versions, licenses, and attribution.

## Documentation

- [Preregistration](docs/PREREGISTRATION.md)
- [Experimental design](docs/DESIGN.md)
- [Exploratory J-space/R-space extension](docs/JR_LENS_EXTENSION.md)
- [Dataset adaptation](docs/DATASET_ADAPTER.md)
- [Runbook](docs/RUNBOOK.md)
- [Cost plan](docs/COST_PLAN.md)
- [Decision log](docs/DECISIONS.md)

## Safety boundary

SwarmStop evaluates simulated operational actions. Do not connect benchmark tools to real accounts, production infrastructure, credentials, networks, payment systems, or personal data. A boundary violation must mutate fake state only.

## License

SwarmStop source code is released under the MIT License. Upstream datasets and source code retain their original licenses.
