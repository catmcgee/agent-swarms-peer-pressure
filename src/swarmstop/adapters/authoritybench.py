from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AuthorityBenchPolicy:
    authorized_actions: tuple[str, ...]
    require_approval_actions: tuple[str, ...]
    authority_constraints: dict[str, dict[str, Any]]
    chain_blocked_after: dict[str, tuple[str, ...]]


def load_authoritybench_scenarios(root: str | Path) -> list[dict[str, Any]]:
    scenarios_root = Path(root) / "authoritybench" / "scenarios"
    scenarios: list[dict[str, Any]] = []
    for path in sorted(scenarios_root.glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        for scenario in raw.get("scenarios", []):
            item = dict(scenario)
            item["category"] = raw.get("category", path.stem)
            item["available_tools"] = list(raw.get("available_tools", []))
            item["source_path"] = str(path)
            scenarios.append(item)
    return scenarios


def policy_from_scenario(scenario: dict[str, Any]) -> AuthorityBenchPolicy:
    return AuthorityBenchPolicy(
        authorized_actions=tuple(str(item) for item in scenario.get("authorized_tools", [])),
        require_approval_actions=tuple(
            str(item) for item in scenario.get("require_approval_tools", [])
        ),
        authority_constraints=dict(scenario.get("authority_constraints") or {}),
        chain_blocked_after={
            str(prior): tuple(str(item) for item in blocked)
            for prior, blocked in (scenario.get("chain_blocked_after") or {}).items()
        },
    )
