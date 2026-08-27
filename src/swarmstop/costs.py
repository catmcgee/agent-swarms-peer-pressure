from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import iter_results


@dataclass(frozen=True)
class TokenPrice:
    provider: str
    model: str
    input_per_million: float
    cached_input_per_million: float | None
    output_per_million: float
    observed_at: str
    source: str

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        cached = min(cached_input_tokens, input_tokens)
        uncached = input_tokens - cached
        cached_rate = (
            self.cached_input_per_million
            if self.cached_input_per_million is not None
            else self.input_per_million
        )
        return (
            uncached * self.input_per_million
            + cached * cached_rate
            + output_tokens * self.output_per_million
        ) / 1_000_000


def load_prices(path: str | Path) -> list[TokenPrice]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    prices: list[TokenPrice] = []
    for item in raw.get("token_prices", []):
        prices.append(
            TokenPrice(
                provider=str(item["provider"]),
                model=str(item["model"]),
                input_per_million=float(item["input_per_million"]),
                cached_input_per_million=(
                    float(item["cached_input_per_million"])
                    if item.get("cached_input_per_million") is not None
                    else None
                ),
                output_per_million=float(item["output_per_million"]),
                observed_at=str(item["observed_at"]),
                source=str(item["source"]),
            )
        )
    return prices


def find_price(prices: list[TokenPrice], provider: str, model: str) -> TokenPrice:
    matches = [price for price in prices if price.provider == provider and price.model == model]
    if len(matches) != 1:
        raise ValueError(f"expected one price for {provider}/{model}, found {len(matches)}")
    return matches[0]


def usage_totals(results_path: str | Path) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "trajectories": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
    }
    for result in iter_results(results_path):
        if result.get("status") != "completed":
            continue
        usage = result.get("usage") or {}
        totals["trajectories"] += 1
        totals["input_tokens"] += int(usage.get("input_tokens", 0))
        totals["cached_input_tokens"] += int(usage.get("cached_input_tokens", 0))
        totals["output_tokens"] += int(usage.get("output_tokens", 0))
    return totals
