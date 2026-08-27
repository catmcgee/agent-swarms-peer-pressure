#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", default=ROOT / "configs" / "task_selection.yaml", type=Path
    )
    parser.add_argument(
        "--dataset", default=ROOT / "data" / "upstreams" / "agentabstain-data", type=Path
    )
    args = parser.parse_args()

    selection = yaml.safe_load(args.selection.read_text(encoding="utf-8"))
    selected = selection["agentabstain"]["tasks"]
    rows = []
    for item in selected:
        task_root = args.dataset / "tasks" / item["category"] / item["task_id"]
        task_path = task_root / item["task_type"] / "task.yaml"
        metadata_path = task_root / "metadata.yaml"
        if not task_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"missing selected task artifact: {task_root}")
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("action_type") != "operational":
            raise ValueError(f"selected task is not operational: {task_root}")
        critical = task.get("critical_actions") or []
        if not critical:
            raise ValueError(f"selected task has no critical action: {task_root}")
        if item["domain"] not in metadata.get("environments", []):
            raise ValueError(f"declared domain is absent from task environments: {task_root}")
        rows.append(
            {
                **item,
                "phase": metadata.get("phase"),
                "critical_actions": critical,
                "environments": metadata.get("environments", []),
            }
        )

    ids = [(row["category"], row["task_id"], row["task_type"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("selected task IDs are not unique")
    summary = {
        "tasks": len(rows),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "phases": dict(sorted(Counter(row["phase"] for row in rows).items())),
        "domains": len({row["domain"] for row in rows}),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
