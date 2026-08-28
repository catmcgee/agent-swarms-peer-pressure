#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from swarmstop.adapters.agentabstain_runtime import make_authorized_control
from swarmstop.boards import render_board
from swarmstop.config import (
    load_experiment_config,
    load_peer_boards,
    load_tasks,
    validate_experiment,
)
from swarmstop.environment import SandboxEnvironment
from swarmstop.schema import AuthorityVerdict, PeerNorm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavior_screen.yaml")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    tasks = load_tasks(config.tasks_path)
    boards = load_peer_boards(config.peer_boards_path)
    validate_experiment(config, tasks, boards)
    assert len(tasks) == 12
    assert len(boards) == 72
    assert len(tasks) * len(config.conditions) * len(config.seeds) == 216

    pre_boundary_calls = 0
    broken_tool_observations = 0
    for task in tasks:
        for source_path, expected in task.metadata["source_hashes"].items():
            actual = hashlib.sha256(
                (Path(task.metadata["dataset_root"]) / source_path).read_bytes()
            ).hexdigest()
            assert actual == expected
        tool_map = {tool.name: tool for tool in task.tools}
        assert all(action in tool_map for action in task.critical_effect_actions)
        assert all(
            tool_map[action].kind != "safe_alternative"
            for action in task.critical_effect_actions
        )
        assert set(task.primary_critical_effect_actions).isdisjoint(
            task.authorized_actions
        )
        assert set(task.delegation_action_aliases.values()).issubset(
            task.critical_effect_actions
        )

        available = task.metadata.get("available_tools")
        if available is not None:
            upstream_names = {
                str(item["name"] if isinstance(item, dict) else item)
                for item in available
            }
            exposed_names = {
                tool.name for tool in task.tools if tool.name != task.feasible_alternative
            }
            assert exposed_names == upstream_names

        environment = SandboxEnvironment(task)
        for rule in task.equivalent_critical_rules:
            action = str(rule["tool"])
            arguments = _matching_arguments(rule)
            assert task.is_critical_action(action, arguments)
            decision = environment.authority.evaluate(action, arguments, [])
            assert decision.verdict is AuthorityVerdict.DENY
        for action in task.critical_effect_actions:
            delegated = environment.execute(
                "request_peer_action", {"peer": "peer-a", "action": action}
            )
            assert delegated.critical_attempt is True
            assert delegated.delegated_attempt is True
        for item in task.metadata["pre_boundary_calls"]:
            assert not task.is_critical_action(item["tool"], item["arguments"])
            record = environment.execute(item["tool"], item["arguments"])
            assert record.critical_attempt is False
            pre_boundary_calls += 1
            if isinstance(record.result, dict) and record.result.get("success") is False:
                broken_tool_observations += 1
        safe = environment.execute(task.feasible_alternative or "", {})
        assert safe.critical_attempt is False
        assert safe.result["critical_effect_applied"] is False

        control = make_authorized_control(task)
        assert control.task_type == "act"
        assert set(control.critical_actions).issubset(control.authorized_actions)
        assert control.equivalent_critical_rules == ()
        assert control.metadata["pre_boundary_calls"] == []

    by_id = {task.id: task for task in tasks}
    assert [
        item["tool"]
        for item in by_id["policy_briefing"].metadata["pre_boundary_calls"]
    ][-1] == "documents_and_analysis.read_repository_record"
    assert [
        item["tool"]
        for item in by_id["grid_deployment"].metadata["pre_boundary_calls"]
    ] == ["industrial_and_infrastructure_control.grid_config_reader"]
    assert by_id["bank_transfer"].metadata["pre_boundary_calls"] == []
    root = by_id["root_deletion"]
    assert not root.is_critical_action("filesystem.transfer_file", {})
    assert root.authority_constraints.get("filesystem.transfer_file", {}) == {}

    pairs: dict[str, dict[PeerNorm, object]] = defaultdict(dict)
    for board in boards:
        assert board.pair_id is not None
        pairs[board.pair_id][board.norm] = board
    assert len(pairs) == 36
    for pair in pairs.values():
        stop = render_board(pair[PeerNorm.STOP])
        proceed = render_board(pair[PeerNorm.PROCEED])
        assert stop.replace('"outcome_code":0', '"outcome_code":1') == proceed
        assert "independent copy" in stop
        assert "focal sandbox and task state are unchanged" in stop
        assert "PROCEEDED_AFTER_BOUNDARY" not in proceed

    print(
        json.dumps(
            {
                "status": "passed",
                "tasks": len(tasks),
                "categories": len({task.family for task in tasks}),
                "domains": len(
                    {
                        domain
                        for task in tasks
                        for domain in task.metadata["source_domain"]
                    }
                ),
                "boards": len(boards),
                "board_pairs": len(pairs),
                "authorized_controls": len(tasks) * len(config.seeds),
                "discovery_trajectories": len(tasks)
                * len(config.conditions)
                * len(config.seeds),
                "pre_boundary_runtime_calls": pre_boundary_calls,
                "expected_broken_tool_observations": broken_tool_observations,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _matching_arguments(rule: dict[str, object]) -> dict[str, object]:
    if "all" in rule:
        merged: dict[str, object] = {}
        for item in rule.get("all") or []:
            merged.update(_matching_arguments(dict(item)))
        return merged
    if "any" in rule:
        options = list(rule.get("any") or [])
        return _matching_arguments(dict(options[0])) if options else {}
    field = str(rule.get("field", ""))
    if not field:
        return {}
    if "equals" in rule:
        return {field: rule["equals"]}
    if "contains" in rule:
        return {field: f"prefix {rule['contains']} suffix"}
    choices = list(rule.get("one_of") or [])
    return {field: choices[0]} if choices else {}


if __name__ == "__main__":
    main()
