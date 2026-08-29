from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from ..schema import TaskSpec, ToolSpec, critical_rule_matches

DATASET_REVISION = "842228426c2a703347396501af61c7890972c7ee"
SOURCE_REVISION = "f581249704b26804e28a39e37396f1be00b71a4d"


def load_agentabstain_screen(path: str | Path) -> list[TaskSpec]:
    """Load a pinned task selection with AgentAbstain's in-memory runtimes."""
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    spec = raw.get("agentabstain_screen") or {}
    base = config_path.parent
    dataset_root = _resolve(base, spec["dataset_root"])
    source_root = _resolve(base, spec["source_root"])
    selection_path = _resolve(base, spec["selection_path"])
    expected_dataset = str(spec.get("dataset_revision", DATASET_REVISION))
    expected_source = str(spec.get("source_revision", SOURCE_REVISION))
    _validate_revisions(dataset_root, source_root, expected_dataset, expected_source)

    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8")) or {}
    selected = (selection.get("agentabstain") or {}).get("tasks") or []
    alternatives = spec.get("safe_alternatives") or {}
    names = spec.get("task_names") or {}
    tasks: list[TaskSpec] = []
    for item in selected:
        category = str(item["category"])
        pair_id = str(item["task_id"])
        key = f"{category}/{pair_id}"
        if names and key not in names:
            raise ValueError(f"screen task lacks a declared name: {key}")
        if alternatives and key not in alternatives:
            raise ValueError(f"screen task lacks a declared safe alternative: {key}")
        tasks.append(
            _build_task(
                dataset_root=dataset_root,
                source_root=source_root,
                category=category,
                pair_id=pair_id,
                task_id=str(names.get(key, f"{category}__{pair_id}")),
                task_type="abstain",
                alternative=(dict(alternatives[key]) if key in alternatives else None),
                dataset_revision=expected_dataset,
                source_revision=expected_source,
            )
        )
    return tasks


def make_authorized_control(task: TaskSpec) -> TaskSpec:
    """Return the paired AgentAbstain act task for a screen task."""
    metadata = task.metadata
    if metadata.get("runtime_adapter") != "agentabstain_in_memory":
        raise ValueError("authorized controls require an AgentAbstain runtime task")
    return _build_task(
        dataset_root=Path(str(metadata["dataset_root"])),
        source_root=Path(str(metadata["source_root"])),
        category=str(metadata["source_category"]),
        pair_id=str(metadata["source_pair_id"]),
        task_id=f"{task.id}__authorized_control",
        task_type="act",
        alternative=None,
        dataset_revision=str(metadata["dataset_revision"]),
        source_revision=str(metadata["source_revision"]),
    )


def build_runtime_environment(task: TaskSpec) -> Any:
    metadata = task.metadata
    source_root = Path(str(metadata["source_root"]))
    dataset_root = Path(str(metadata["dataset_root"]))
    build_multi_environment = _bootstrap_runtime(source_root, dataset_root)
    runtime = build_multi_environment(
        list(metadata["environment_types"]),
        copy.deepcopy(metadata["runtime_initial_states"]),
    )
    _apply_runtime_overrides(runtime, metadata)
    return runtime


def _build_task(
    *,
    dataset_root: Path,
    source_root: Path,
    category: str,
    pair_id: str,
    task_id: str,
    task_type: str,
    alternative: dict[str, Any] | None,
    dataset_revision: str,
    source_revision: str,
) -> TaskSpec:
    pair_root = dataset_root / "tasks" / category / pair_id
    variant_root = pair_root / task_type
    act_raw = _read_yaml(pair_root / "act" / "task.yaml")
    raw = _read_yaml(variant_root / "task.yaml")
    pair_metadata = _read_yaml(pair_root / "metadata.yaml")
    env_types = tuple(str(item) for item in pair_metadata.get("environments", []))
    initial_states = {
        env_name: json.loads(
            (variant_root / "initial_states" / f"{env_name}.json").read_text(
                encoding="utf-8"
            )
        )
        for env_name in env_types
    }
    runtime = _bootstrap_runtime(source_root, dataset_root)(list(env_types), initial_states)
    runtime_metadata = {
        "tool_overrides": raw.get("tool_overrides") or {},
        "available_tools": raw.get("available_tools"),
    }
    _apply_runtime_overrides(runtime, runtime_metadata)
    allowed_tools = raw.get("available_tools")
    allowed_names = (
        {
            str(item["name"] if isinstance(item, dict) else item)
            for item in allowed_tools
        }
        if allowed_tools is not None
        else None
    )
    tool_specs = tuple(
        _tool_spec(item)
        for item in runtime.get_tool_schemas()
        if allowed_names is None or str(item["name"]) in allowed_names
    )

    critical = tuple(str(item) for item in raw.get("critical_actions", []))
    if not critical:
        critical = tuple(str(item) for item in act_raw.get("critical_actions", []))
    excluded_primary = tuple(
        str(item)
        for item in (alternative or {}).get("excluded_primary_critical_actions", [])
    )
    primary_effect = tuple(action for action in critical if action not in set(excluded_primary))
    equivalent_rules = tuple(
        dict(item) for item in (alternative or {}).get("equivalent_critical_rules", [])
    )
    equivalent_names = tuple(str(item["tool"]) for item in equivalent_rules)
    effective_critical = tuple(dict.fromkeys((*primary_effect, *equivalent_names)))
    tool_names = {tool.name for tool in tool_specs}
    missing = set(effective_critical) - tool_names
    if missing:
        raise ValueError(f"critical actions hidden or absent for {category}/{pair_id}: {missing}")

    tools = list(tool_specs)
    feasible_alternative = None
    if alternative is not None:
        feasible_alternative = str(alternative["name"])
        tools.append(
            ToolSpec(
                name=feasible_alternative,
                description=str(alternative["description"]),
                input_schema={
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "work_summary": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                kind="safe_alternative",
                result={
                    "status": "saved_without_boundary_crossing",
                    "authorization_granted": False,
                    "critical_effect_applied": False,
                },
                effect={f"safe_record__{feasible_alternative}": True},
            )
        )

    dag_nodes = list((act_raw.get("execution_dag") or {}).get("nodes", []))
    visible_names = {tool.name for tool in tools}
    preparatory = [
        {
            "tool": str(node["tool"]),
            "arguments": dict(node.get("params") or {}),
            "description": str(node.get("description", node["tool"])),
        }
        for node in dag_nodes
        if not _node_has_critical_effect(
            str(node.get("tool")),
            dict(node.get("params") or {}),
            primary_effect,
            equivalent_rules,
        )
        and str(node.get("tool")) in visible_names
    ]
    preparatory = _truncate_at_runtime_trigger(
        preparatory,
        raw.get("abstention_trigger") if task_type == "abstain" else None,
        explicit_stop_after=(alternative or {}).get("pre_boundary_stop_after"),
    )
    progress = tuple(item["description"] for item in preparatory)
    source_files = [
        pair_root / "act" / "task.yaml",
        pair_root / "abstain" / "task.yaml",
        pair_root / "metadata.yaml",
        *(variant_root / "initial_states").glob("*.json"),
    ]
    source_hashes = {
        str(item.relative_to(dataset_root)): _sha256(item) for item in sorted(source_files)
    }
    authorized = tuple(tool.name for tool in tools)
    if task_type == "abstain":
        authorized = tuple(
            name for name in authorized if name not in set(primary_effect)
        )
    aliases = (
        {
            str(alias): primary_effect[0]
            for alias in (alternative or {}).get("delegation_aliases", [])
        }
        if primary_effect
        else {}
    )

    return TaskSpec(
        id=task_id,
        family=category,
        system_prompt=str(raw.get("system_prompt", act_raw.get("system_prompt", ""))),
        instruction=str(raw.get("instruction", act_raw.get("instruction", ""))),
        task_type=task_type,
        tools=tuple(tools),
        critical_actions=critical,
        authorized_actions=authorized,
        feasible_alternative=feasible_alternative,
        personal_progress=progress if task_type == "abstain" else (),
        inherited_progress=progress if task_type == "abstain" else (),
        metadata={
            "origin": "agentabstain",
            "runtime_adapter": "agentabstain_in_memory",
            "experimental_augmentation": bool(alternative),
            "source_category": category,
            "source_pair_id": pair_id,
            "source_phase": pair_metadata.get("phase"),
            "source_domain": list(env_types),
            "dataset_revision": dataset_revision,
            "source_revision": source_revision,
            "dataset_root": str(dataset_root),
            "source_root": str(source_root),
            "source_hashes": source_hashes,
            "environment_types": list(env_types),
            "runtime_initial_states": initial_states,
            "tool_overrides": raw.get("tool_overrides") or {},
            "available_tools": raw.get("available_tools"),
            "pre_boundary_calls": preparatory if task_type == "abstain" else [],
            "excluded_primary_critical_actions": list(excluded_primary),
            "equivalent_critical_rules": list(equivalent_rules),
            "delegation_action_aliases": aliases,
            "abstention_trigger": (
                raw.get("abstention_trigger") if task_type == "abstain" else None
            ),
        },
    )


def _truncate_at_runtime_trigger(
    preparatory: list[dict[str, Any]],
    trigger: Any,
    *,
    explicit_stop_after: Any = None,
) -> list[dict[str, Any]]:
    """Stop deterministic setup once a runtime failure or risk signal is observed."""
    if explicit_stop_after:
        trigger_tool = str(explicit_stop_after)
    elif not isinstance(trigger, dict):
        return preparatory
    else:
        trigger_tool = None
    trigger_type = str(trigger.get("trigger_type", "")) if isinstance(trigger, dict) else ""
    if trigger_tool is None and trigger_type == "critical_tool_failure":
        broken = trigger.get("broken_tool") or {}
        if isinstance(broken, dict):
            trigger_tool = str(broken.get("name") or "") or None
    elif trigger_tool is None and trigger_type == "emergent_risk_discovery":
        risk = trigger.get("risk_signal") or {}
        if isinstance(risk, dict):
            trigger_tool = str(risk.get("where") or "").split(" ", 1)[0] or None
    elif trigger_tool is None and trigger_type == "conflicting_evidence":
        stop_index = _conflicting_evidence_stop_index(preparatory, trigger)
        if stop_index is None:
            raise ValueError("conflicting-evidence sources do not resolve to setup calls")
        return preparatory[: stop_index + 1]
    if trigger_tool is None:
        return preparatory
    for index, item in enumerate(preparatory):
        if item["tool"] == trigger_tool:
            return preparatory[: index + 1]
    if explicit_stop_after:
        raise ValueError(
            f"declared pre_boundary_stop_after tool was not reachable: {trigger_tool}"
        )
    return preparatory


def _conflicting_evidence_stop_index(
    preparatory: list[dict[str, Any]],
    trigger: dict[str, Any],
) -> int | None:
    contradiction = trigger.get("contradiction") or {}
    evidence = [str(contradiction.get(name, "")) for name in ("evidence_a", "evidence_b")]
    if any(not item for item in evidence):
        return None
    ranked = [
        [(1000, -1)]
        if "user instruction" in item.lower()
        else _rank_evidence_sources(preparatory, item)
        for item in evidence
    ]
    candidates = [
        (score_a + score_b, index_a, index_b)
        for score_a, index_a in ranked[0]
        for score_b, index_b in ranked[1]
        if index_a != index_b
    ]
    if not candidates:
        return None
    _, index_a, index_b = max(candidates)
    return max(index_a, index_b)


def _best_evidence_source_index(
    preparatory: list[dict[str, Any]],
    evidence: str,
) -> int | None:
    ranked = _rank_evidence_sources(preparatory, evidence)
    return ranked[0][1] if ranked else None


def _rank_evidence_sources(
    preparatory: list[dict[str, Any]],
    evidence: str,
) -> list[tuple[int, int]]:
    evidence_lower = evidence.lower()
    evidence_tokens = _meaningful_tokens(evidence)
    ranked: list[tuple[int, int]] = []
    for index, item in enumerate(preparatory):
        tool = str(item["tool"])
        suffix = tool.rsplit(".", 1)[-1]
        corpus = " ".join(
            (
                tool,
                str(item.get("description", "")),
            )
        )
        overlap = len(evidence_tokens & _meaningful_tokens(corpus))
        explicit = (
            100
            if tool.lower() in evidence_lower
            else 80
            if suffix.lower() in evidence_lower
            else 0
        )
        read_source = 20 if suffix.startswith(("read_", "get_", "lookup_")) else 0
        if explicit or overlap >= 2:
            ranked.append((explicit + read_source + overlap, index))
    return sorted(ranked, reverse=True)


def _meaningful_tokens(value: str) -> set[str]:
    stopwords = {
        "about",
        "after",
        "against",
        "authoritative",
        "before",
        "cannot",
        "could",
        "entry",
        "shows",
        "source",
        "status",
        "their",
        "there",
        "these",
        "which",
        "would",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", value.lower())
        if len(token) >= 4 and token not in stopwords
    }


def _node_has_critical_effect(
    tool: str,
    arguments: dict[str, Any],
    primary_effect: tuple[str, ...],
    equivalent_rules: tuple[dict[str, Any], ...],
) -> bool:
    if tool in primary_effect:
        return True
    return any(
        rule.get("tool") == tool and critical_rule_matches(rule, arguments)
        for rule in equivalent_rules
    )


def _bootstrap_runtime(source_root: Path, dataset_root: Path):
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    from fastmcp import FastMCP

    if not hasattr(FastMCP, "list_tools"):
        async def list_tools(self):  # type: ignore[no-untyped-def]
            return list((await self.get_tools()).values())

        FastMCP.list_tools = list_tools  # type: ignore[attr-defined]
    if not hasattr(FastMCP, "call_tool"):
        async def call_tool(self, name, arguments):  # type: ignore[no-untyped-def]
            return await (await self.get_tool(name)).run(arguments)

        FastMCP.call_tool = call_tool  # type: ignore[attr-defined]

    env_package = importlib.import_module("abstention_factory.environments")
    generated_path = str((dataset_root / "environments").resolve())
    if generated_path not in env_package.__path__:
        env_package.__path__.append(generated_path)
    module = importlib.import_module("abstention_factory.runtime.multi")
    return module.build_multi_environment


def _apply_runtime_overrides(runtime: Any, metadata: dict[str, Any]) -> None:
    for broken in (metadata.get("tool_overrides") or {}).get("broken_tools", []):
        runtime.break_tool(str(broken["name"]), str(broken.get("error", "Service unavailable")))
    available = metadata.get("available_tools")
    if available is None:
        return
    allowed = {str(item["name"] if isinstance(item, dict) else item) for item in available}
    for schema in runtime.get_tool_schemas():
        if schema["name"] not in allowed:
            runtime.hide_tool(schema["name"])


def _tool_spec(raw: dict[str, Any]) -> ToolSpec:
    kind = str(raw.get("kind") or "lookup")
    if kind not in {"lookup", "verify", "commit"}:
        kind = "lookup"
    return ToolSpec(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        input_schema=dict(
            raw.get("input_schema")
            or {"type": "object", "properties": {}, "additionalProperties": True}
        ),
        kind=kind,
        result=None,
    )


def _validate_revisions(
    dataset_root: Path,
    source_root: Path,
    dataset_revision: str,
    source_revision: str,
) -> None:
    tree = dataset_root / ".cache" / "huggingface" / "trees" / f"{dataset_revision}.json"
    if not tree.exists():
        raise ValueError(f"AgentAbstain dataset revision is absent: {dataset_revision}")
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != source_revision:
        raise ValueError(
            f"AgentAbstain source mismatch: expected {source_revision}, found {actual}"
        )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
