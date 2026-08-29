#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from swarmstop.adapters.agentabstain_runtime import (
    DATASET_REVISION,
    SOURCE_REVISION,
    _build_task,
    _conflicting_evidence_stop_index,
)
from swarmstop.config import load_tasks
from swarmstop.environment import SandboxEnvironment

FAMILIES = (
    "ambiguous_action_specification",
    "conflicting_evidence",
    "critical_tool_failure",
    "emergent_risk_discovery",
    "high_stakes_action",
    "insufficient_tool_capability",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default="configs/tasks/delegation_contact_v1.yaml",
    )
    parser.add_argument(
        "--selection",
        default="configs/task_selection_delegation_contact_v1.yaml",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    tasks_path = (root / args.tasks).resolve()
    selection_path = (root / args.selection).resolve()
    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8"))["agentabstain"]
    semantic_exclusions = {
        str(key): str(reason).strip()
        for key, reason in (selection.get("semantic_equivalence_exclusions") or {}).items()
    }
    operational_exclusions = {
        str(key): str(reason).strip()
        for key, reason in (selection.get("operational_exclusions") or {}).items()
    }
    if not semantic_exclusions or any(not reason for reason in semantic_exclusions.values()):
        raise ValueError("semantic-equivalence exclusion ledger is absent or incomplete")
    if any(not reason for reason in operational_exclusions.values()):
        raise ValueError("operational exclusion ledger contains an empty reason")
    salt = str(selection["selection_salt"])
    if hashlib.sha256(salt.encode()).hexdigest() != selection["selection_salt_sha256"]:
        raise ValueError("selection salt hash mismatch")

    tasks = load_tasks(tasks_path)
    if len(tasks) != 24:
        raise ValueError("diagnostic requires exactly 24 held-out tasks")
    counts = Counter(task.family for task in tasks)
    if counts != Counter({family: 4 for family in FAMILIES}):
        raise ValueError(f"task-family balance mismatch: {dict(counts)}")

    old_selection = yaml.safe_load(
        (root / "configs/task_selection.yaml").read_text(encoding="utf-8")
    )["agentabstain"]["tasks"]
    old_keys = {(str(item["category"]), str(item["task_id"])) for item in old_selection}
    selected_keys = {(str(item["category"]), str(item["task_id"])) for item in selection["tasks"]}
    overlap = selected_keys & old_keys
    if overlap:
        raise ValueError(f"held-out selection reuses prior tasks: {sorted(overlap)}")
    selected_labels = {f"{family}/{pair_id}" for family, pair_id in selected_keys}
    if selected_labels & set(semantic_exclusions):
        raise ValueError("selected task appears in semantic-equivalence exclusion ledger")
    if selected_labels & set(operational_exclusions):
        raise ValueError("selected task appears in operational exclusion ledger")

    dataset_root = root / "data/upstreams/agentabstain-data"
    source_root = root / "data/upstreams/agentabstain"
    eligible: dict[str, list[tuple[str, str]]] = defaultdict(list)
    rejected: dict[str, list[str]] = defaultdict(list)
    for family in FAMILIES:
        for pair_root in sorted((dataset_root / "tasks" / family).glob("preview_*")):
            pair_id = pair_root.name
            if (family, pair_id) in old_keys:
                continue
            candidate_key = f"{family}/{pair_id}"
            if candidate_key in semantic_exclusions:
                rejected[candidate_key].append(
                    "semantic same-effect substitute: " + semantic_exclusions[candidate_key]
                )
                continue
            if candidate_key in operational_exclusions:
                rejected[candidate_key].append(
                    "outcome-blind operational audit: " + operational_exclusions[candidate_key]
                )
                continue
            try:
                task = _build_task(
                    dataset_root=dataset_root,
                    source_root=source_root,
                    category=family,
                    pair_id=pair_id,
                    task_id=f"{family}__{pair_id}",
                    task_type="abstain",
                    alternative=None,
                    dataset_revision=DATASET_REVISION,
                    source_revision=SOURCE_REVISION,
                )
                issues = operational_issues(task)
            except Exception as exc:
                issues = [f"build failed: {type(exc).__name__}: {exc}"]
            if issues:
                rejected[f"{family}/{pair_id}"].extend(issues)
                continue
            digest = hashlib.sha256(f"{salt}|{family}/{pair_id}".encode()).hexdigest()
            eligible[family].append((digest, pair_id))

    expected: set[tuple[str, str]] = set()
    for family in FAMILIES:
        ranked = sorted(eligible[family])
        if len(ranked) < 4:
            raise ValueError(f"fewer than four eligible tasks in {family}")
        expected.update((family, pair_id) for _, pair_id in ranked[:4])
    if selected_keys != expected:
        missing = sorted(expected - selected_keys)
        extra = sorted(selected_keys - expected)
        raise ValueError(f"selection differs from hash rule; missing={missing}, extra={extra}")

    loaded = {(task.family, task.metadata["source_pair_id"]): task for task in tasks}
    selected_specs = {
        (str(item["category"]), str(item["task_id"])): item for item in selection["tasks"]
    }
    for key in sorted(selected_keys):
        task = loaded[key]
        spec = selected_specs[key]
        issues = operational_issues(task)
        if str(spec.get("task_type")) != "abstain":
            issues.append("selection task_type is not abstain")
        if str(spec.get("domain")) not in set(task.metadata.get("source_domain") or []):
            issues.append("selection domain does not match source metadata")
        if issues:
            raise ValueError(f"selected task failed operational audit {key}: {issues}")
    if "emergent_risk_discovery/preview_024" not in rejected:
        raise ValueError("known unresolved emergent-risk trigger was not rejected")

    print(
        yaml.safe_dump(
            {
                "status": "valid",
                "selected_tasks": len(tasks),
                "families": dict(sorted(counts.items())),
                "eligible_counts": {family: len(eligible[family]) for family in FAMILIES},
                "known_rejection": rejected["emergent_risk_discovery/preview_024"],
                "semantic_exclusions": len(semantic_exclusions),
                "operational_exclusions": len(operational_exclusions),
            },
            sort_keys=False,
        ).strip()
    )


def operational_issues(task: Any) -> list[str]:
    issues: list[str] = []
    visible = {tool.name for tool in task.tools}
    if not task.critical_effect_actions:
        issues.append("empty critical-action set")
    if not set(task.critical_effect_actions).issubset(visible):
        issues.append("critical action is not visible")
    if task.feasible_alternative is not None:
        issues.append("safe alternative is model-visible")
    if task.metadata.get("experimental_augmentation"):
        issues.append("task uses an experimental augmentation")
    if task.equivalent_critical_rules or task.delegation_action_aliases:
        issues.append("task needs a task-specific critical-action patch")

    environment = SandboxEnvironment(task)
    records = []
    for item in task.metadata.get("pre_boundary_calls") or []:
        record = environment.execute(str(item["tool"]), dict(item.get("arguments") or {}))
        records.append(record)
        if record.critical_attempt:
            issues.append(f"pre-boundary call is critical: {record.tool}")

    trigger = task.metadata.get("abstention_trigger") or {}
    trigger_type = str(trigger.get("trigger_type", ""))
    if trigger_type == "critical_tool_failure":
        broken = str((trigger.get("broken_tool") or {}).get("name", ""))
        if not records or records[-1].tool != broken:
            issues.append("setup does not end on the declared broken tool")
        elif not _contains_error(records[-1].result):
            issues.append("declared broken tool did not expose an error")
    if trigger_type == "emergent_risk_discovery":
        where = str((trigger.get("risk_signal") or {}).get("where", ""))
        resolved = [name for name in visible if name in where]
        if not resolved:
            issues.append("risk-signal location does not resolve to a visible tool")
        elif not records or records[-1].tool not in resolved:
            issues.append("setup does not end on a declared risk-signal tool")
    if trigger_type == "conflicting_evidence":
        preparatory = list(task.metadata.get("pre_boundary_calls") or [])
        stop_index = _conflicting_evidence_stop_index(preparatory, trigger)
        if stop_index is None:
            issues.append("contradiction sources do not resolve to setup calls")
        elif not records or records[-1].tool != preparatory[stop_index]["tool"]:
            issues.append("setup does not end when both contradiction sources are visible")
    return issues


def _contains_error(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("error")) or value.get("success") is False
    return "error" in str(value).lower() or "unavailable" in str(value).lower()


if __name__ == "__main__":
    main()
