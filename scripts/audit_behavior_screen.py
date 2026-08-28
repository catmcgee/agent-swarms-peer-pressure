#!/usr/bin/env python3
from __future__ import annotations

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
from swarmstop.schema import PeerNorm


def main() -> None:
    config = load_experiment_config("configs/behavior_screen.yaml")
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
        assert all(tool_map[action].kind == "commit" for action in task.critical_actions)
        assert set(task.critical_actions).isdisjoint(task.authorized_actions)

        environment = SandboxEnvironment(task)
        for item in task.metadata["pre_boundary_calls"]:
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
        assert control.metadata["pre_boundary_calls"] == []

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


if __name__ == "__main__":
    main()
