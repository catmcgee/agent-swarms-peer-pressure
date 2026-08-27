# Compute and cost plan

Pricing is volatile. The checked-in table records the observation date and source; refresh it before approving a large run.

## Default execution strategy

1. Validate locally with deterministic policies.
2. Run the pilot on token-priced open-model endpoints using batch inference where available.
3. Measure billed input tokens, output tokens, latency, and cost per valid trajectory.
4. Decide endpoint versus dedicated GPU from observed cost rather than advertised throughput.
5. Self-host one exact open-weight checkpoint for reproducibility.
6. Run costly frontier endpoints only on prespecified high-information cells unless the first result justifies the full matrix.

## Cost equations

For token-priced inference:

```text
cost = input_tokens × input_price_per_million / 1,000,000
     + output_tokens × output_price_per_million / 1,000,000
```

For a dedicated GPU:

```text
cost = wall_clock_hours × hourly_rate + storage + data transfer
```

Use wall-clock measurements from a saturated batch. Output-tokens-per-second alone omits prompt prefill, tool latency, failed requests, and idle gaps.

## Cost controls

- Generate each authentic peer board once and yoke it to many focal trials.
- Branch from the recognition checkpoint instead of recomputing task prefixes.
- Use provider prompt caching or inference-server prefix caching where available.
- Batch independent focal agents.
- Stop after the first critical attempt while retaining the trace.
- Resume by stable trial ID; never rerun completed trials silently.
- Use a targeted live-swarm subset.
- Set provider spending limits and terminate dedicated instances after jobs.

## Provisional caps

- Machinery tests: $10.
- Pilot: $100.
- Controlled open-model study: $500.
- Self-hosted reproducibility run: $700.
- Frontier replication: separate approval after pilot accounting.

These are operational caps, not sample-size criteria. A run stops before exceeding its cap and is resumed only after a logged decision.
