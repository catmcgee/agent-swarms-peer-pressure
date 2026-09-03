#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time


def compute_deadline(
    *,
    billing_start_unix: float,
    max_wall_seconds: float,
    hourly_price_usd: float,
    max_cost_usd: float,
    now_unix: float | None = None,
) -> tuple[int, int]:
    values = (billing_start_unix, max_wall_seconds, hourly_price_usd, max_cost_usd)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("deadline inputs must be finite")
    if billing_start_unix <= 0 or min(values[1:]) <= 0:
        raise ValueError("deadline inputs must be positive")
    if max_wall_seconds != 108_000 or max_cost_usd != 45:
        raise ValueError("deadline caps differ from the registered protocol")
    current = time.time() if now_unix is None else now_unix
    if billing_start_unix > current:
        raise ValueError("billing epoch cannot begin in the future")
    hard_seconds = min(
        math.floor(max_wall_seconds),
        math.floor(max_cost_usd / hourly_price_usd * 3600),
    )
    absolute_deadline = math.floor(billing_start_unix + hard_seconds)
    remaining = math.floor(absolute_deadline - current)
    if remaining <= 0:
        raise ValueError("no time remains under the absolute worker/cost deadline")
    return remaining, absolute_deadline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("billing_start_unix", type=float)
    parser.add_argument("max_wall_seconds", type=float)
    parser.add_argument("hourly_price_usd", type=float)
    parser.add_argument("max_cost_usd", type=float)
    args = parser.parse_args()
    remaining, deadline = compute_deadline(
        billing_start_unix=args.billing_start_unix,
        max_wall_seconds=args.max_wall_seconds,
        hourly_price_usd=args.hourly_price_usd,
        max_cost_usd=args.max_cost_usd,
    )
    print(remaining, deadline)


if __name__ == "__main__":
    main()
