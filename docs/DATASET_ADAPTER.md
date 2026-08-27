# Dataset adaptation and task admission

## Pinned inputs

The fetch script obtains the AgentAbstain source at the commit recorded in `THIRD_PARTY.md` and downloads its separate dataset snapshot. AuthorityBench source is pinned independently.

The initial AgentAbstain download contains 263 should-abstain task artifacts. SwarmStop's behavioral study uses only operational tasks with explicit critical actions. Informational tasks remain useful background cases but cannot measure executable boundary crossing.

## Provisional pilot sample

`configs/task_selection.yaml` selects twelve operational tasks across six abstention categories and twelve primary domains. The selection intentionally mixes pre-execution and runtime triggers and avoids concentrating the pilot in email or filesystem scenarios.

Run:

```bash
python scripts/validate_task_selection.py
```

The validator checks that every selected task exists, is operational, has an explicit critical action, and contains the declared domain.

## Required adaptation steps

Each admitted task needs a SwarmStop wrapper with:

1. the original instruction, system prompt, initial state, tools, and critical-action labels;
2. an explicit authority policy naming the allowed principal;
3. a deterministic boundary notice presented at the recognition checkpoint;
4. a feasible-condition alternative that does not accomplish the forbidden effect;
5. matched personal and inherited progress histories ending in identical state;
6. a paired should-act capability control;
7. a state-diff assertion showing the critical tool mutates only the intended fake state;
8. a test showing peer messages cannot alter the authority oracle.

The schema-only loader in `swarmstop.adapters.agentabstain` is for inventory and conversion checks. It marks converted tasks as requiring the pinned upstream runtime, and the generic sandbox refuses to execute them. This prevents placeholder tool results from entering empirical runs.

## Peer-board admission

Synthetic boards are allowed for machinery tests only. An authentic board must retain:

- the generating model endpoint and revision;
- task and seed;
- raw agent trace;
- authority assessment;
- tool audit log;
- unedited board messages;
- a deterministic label confirming stop or proceed behavior.

If a board is edited beyond formatting or identifier redaction, mark it synthetic.

## Attribution

Do not commit upstream task artifacts into this repository without preserving CC BY 4.0 attribution. Prefer stable task references and the fetch script.
