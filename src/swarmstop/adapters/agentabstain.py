from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from ..schema import TaskSpec, ToolSpec


def scan_agentabstain_tasks(
    dataset_root: str | Path,
    *,
    task_types: Iterable[str] = ("abstain",),
    categories: Iterable[str] | None = None,
    operational_only: bool = True,
) -> list[Path]:
    root = Path(dataset_root) / "tasks"
    allowed_types = set(task_types)
    allowed_categories = set(categories) if categories is not None else None
    paths: list[Path] = []
    for path in root.glob("*/*/*/task.yaml"):
        task_type = path.parent.name
        category = path.parents[2].name
        if task_type not in allowed_types:
            continue
        if allowed_categories is not None and category not in allowed_categories:
            continue
        if operational_only:
            metadata_path = path.parent.parent / "metadata.yaml"
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = yaml.safe_load(handle) or {}
            if metadata.get("action_type") != "operational":
                continue
        paths.append(path)
    return sorted(paths)


def load_agentabstain_task(path: str | Path) -> TaskSpec:
    """Convert one downloaded AgentAbstain task artifact to the common schema.

    This loader preserves task identity and action labels. Execution against the
    original environment remains an explicit runtime integration step; generic
    results below are placeholders and must not be used for empirical runs.
    """

    task_path = Path(path)
    with task_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    metadata_path = task_path.parent.parent / "metadata.yaml"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle) or {}

    category = task_path.parents[2].name
    task_id = task_path.parent.parent.name
    task_type = task_path.parent.name
    critical = tuple(str(item) for item in raw.get("critical_actions", []))
    tools = _tool_specs(raw, critical)
    tool_names = {tool.name for tool in tools}
    if not critical:
        raise ValueError(f"AgentAbstain task has no explicit critical_actions: {task_path}")
    if set(critical) - tool_names:
        raise ValueError(f"critical actions are absent from available tools: {task_path}")

    prep = tuple(
        str(node.get("description", node.get("tool", "preparatory step")))
        for node in (raw.get("execution_dag") or {}).get("nodes", [])
        if str(node.get("tool")) not in set(critical)
    )
    return TaskSpec(
        id=f"agentabstain__{category}__{task_id}__{task_type}",
        family=category,
        system_prompt=str(raw.get("system_prompt", "")),
        instruction=str(raw.get("instruction", "")),
        task_type=task_type,
        tools=tools,
        critical_actions=critical,
        authorized_actions=tuple(sorted(tool_names - set(critical))),
        personal_progress=prep,
        inherited_progress=prep,
        metadata={
            "origin": "agentabstain",
            "source_path": str(task_path),
            "upstream_metadata": metadata,
            "runtime_integration_required": True,
        },
    )


def _tool_specs(raw: dict[str, Any], critical: tuple[str, ...]) -> tuple[ToolSpec, ...]:
    declared = raw.get("available_tools") or []
    specs: dict[str, ToolSpec] = {}
    for item in declared:
        if isinstance(item, dict):
            name = str(item["name"])
            specs[name] = ToolSpec(
                name=name,
                description=str(item.get("description", "Upstream benchmark tool")),
                input_schema=dict(
                    item.get("input_schema")
                    or item.get("parameters")
                    or {"type": "object", "properties": {}, "additionalProperties": True}
                ),
                kind=str(item.get("kind", "commit" if name in critical else "lookup")),
                result={"upstream_runtime_required": True},
            )
        else:
            name = str(item)
            specs[name] = ToolSpec(
                name=name,
                description="Upstream benchmark tool",
                kind="commit" if name in critical else "lookup",
                result={"upstream_runtime_required": True},
            )

    for node in (raw.get("execution_dag") or {}).get("nodes", []):
        name = str(node.get("tool", ""))
        if not name:
            continue
        properties = {
            key: {"type": _json_type(value)} for key, value in (node.get("params") or {}).items()
        }
        specs.setdefault(
            name,
            ToolSpec(
                name=name,
                description=str(node.get("description", "Upstream benchmark tool")),
                input_schema={
                    "type": "object",
                    "properties": properties,
                    "additionalProperties": True,
                },
                kind="commit" if name in critical else str(node.get("kind", "lookup")),
                result={"upstream_runtime_required": True},
            ),
        )
    for name in critical:
        specs.setdefault(
            name,
            ToolSpec(
                name=name,
                description="Upstream critical action",
                kind="commit",
                result={"upstream_runtime_required": True},
            ),
        )
    return tuple(specs[name] for name in sorted(specs))


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"
