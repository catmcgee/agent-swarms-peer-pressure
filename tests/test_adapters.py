from pathlib import Path

import yaml

from swarmstop.adapters.authoritybench import (
    load_authoritybench_scenarios,
    policy_from_scenario,
)


def test_authoritybench_loader_preserves_history_rule(tmp_path: Path) -> None:
    scenarios = tmp_path / "authoritybench" / "scenarios"
    scenarios.mkdir(parents=True)
    payload = {
        "category": "tool_chaining",
        "available_tools": ["read_file", "send_email"],
        "scenarios": [
            {
                "id": "case_1",
                "authorized_tools": ["read_file", "send_email"],
                "chain_blocked_after": {"read_file": ["send_email"]},
            }
        ],
    }
    (scenarios / "tool_chaining.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = load_authoritybench_scenarios(tmp_path)
    policy = policy_from_scenario(loaded[0])

    assert policy.authorized_actions == ("read_file", "send_email")
    assert policy.chain_blocked_after == {"read_file": ("send_email",)}
