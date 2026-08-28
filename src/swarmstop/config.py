from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import (
    ExperimentConfig,
    Feasibility,
    Investment,
    PeerBoard,
    PeerNorm,
    TaskSpec,
)


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path)
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a mapping")

    base = config_path.parent

    def resolved(value: str | None) -> str | None:
        if value is None:
            return None
        candidate = Path(value)
        return str(candidate if candidate.is_absolute() else (base / candidate).resolve())

    factors = raw.get("factors") or {}
    inference = raw.get("inference") or {}
    return ExperimentConfig(
        name=str(raw.get("name", config_path.stem)),
        tasks_path=resolved(str(raw["tasks_path"])) or "",
        peer_boards_path=resolved(raw.get("peer_boards_path")),
        output_dir=resolved(str(raw.get("output_dir", "../results/run"))) or "",
        seeds=tuple(int(seed) for seed in raw.get("seeds", [0])),
        feasibility=tuple(
            Feasibility(str(item)) for item in factors.get("feasibility", ["feasible"])
        ),
        investment=tuple(
            Investment(str(item)) for item in factors.get("investment", ["inherited"])
        ),
        peer_norm=tuple(PeerNorm(str(item)) for item in factors.get("peer_norm", ["absent"])),
        budget_rounds=tuple(int(item) for item in factors.get("budget_rounds", [2])),
        model=str(inference.get("model", "scripted-social")),
        provider=str(inference.get("provider", "scripted")),
        temperature=float(inference.get("temperature", 0.7)),
        max_output_tokens=int(inference.get("max_output_tokens", 800)),
        max_cost_usd=(
            float(raw["max_cost_usd"]) if raw.get("max_cost_usd") is not None else None
        ),
        require_authentic_boards=bool(raw.get("require_authentic_boards", False)),
        primary_horizon=int(raw.get("primary_horizon", 2)),
        protocol_version=str(raw.get("protocol_version", "behavior-v1")),
    )


def load_tasks(path: str | Path) -> list[TaskSpec]:
    task_path = Path(path)
    raw = _load_yaml(task_path)
    if isinstance(raw, dict) and "agentabstain_screen" in raw:
        from .adapters.agentabstain_runtime import load_agentabstain_screen

        tasks = load_agentabstain_screen(task_path)
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task IDs must be unique")
        return tasks
    values = raw.get("tasks", []) if isinstance(raw, dict) else raw
    if not isinstance(values, list):
        raise ValueError("task file must be a list or contain a tasks list")
    tasks = [TaskSpec.from_dict(item) for item in values]
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task IDs must be unique")
    return tasks


def load_peer_boards(path: str | Path | None) -> list[PeerBoard]:
    if path is None:
        return []
    raw = _load_yaml(Path(path))
    if isinstance(raw, dict) and "paired_screen" in raw:
        from .boards import generate_paired_boards

        values = (raw.get("paired_screen") or {}).get("tasks") or []
        if not isinstance(values, list):
            raise ValueError("paired_screen.tasks must be a list")
        boards = generate_paired_boards(values)
        ids = [board.id for board in boards]
        if len(ids) != len(set(ids)):
            raise ValueError("peer-board IDs must be unique")
        return boards
    values = raw.get("boards", []) if isinstance(raw, dict) else raw
    if not isinstance(values, list):
        raise ValueError("peer-board file must be a list or contain a boards list")
    boards = [PeerBoard.from_dict(item) for item in values]
    ids = [board.id for board in boards]
    if len(ids) != len(set(ids)):
        raise ValueError("peer-board IDs must be unique")
    return boards


def validate_experiment(
    config: ExperimentConfig,
    tasks: list[TaskSpec],
    boards: list[PeerBoard],
) -> None:
    if not tasks:
        raise ValueError("experiment has no tasks")
    if not config.seeds:
        raise ValueError("experiment has no seeds")
    if not config.conditions:
        raise ValueError("experiment has no conditions")

    for norm in config.peer_norm:
        if norm is PeerNorm.ABSENT:
            continue
        matching = [board for board in boards if board.norm is norm]
        if not matching:
            raise ValueError(f"no peer boards for norm {norm.value}")
        if config.require_authentic_boards and any(not board.authentic for board in matching):
            raise ValueError(
                "synthetic board present while authentic boards are required: "
                f"{norm.value}"
            )

    task_ids = {task.id for task in tasks}
    unknown = {board.task_id for board in boards if board.task_id not in {"*", *task_ids}}
    if unknown:
        raise ValueError(f"peer boards reference unknown tasks: {sorted(unknown)}")
