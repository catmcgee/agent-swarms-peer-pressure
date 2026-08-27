from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Feasibility(StrEnum):
    FEASIBLE = "feasible"
    IMPOSSIBLE = "impossible"


class Investment(StrEnum):
    INHERITED = "inherited"
    PERSONAL = "personal"


class PeerNorm(StrEnum):
    ABSENT = "absent"
    STOP = "stop"
    PROCEED = "proceed"


class AuthorityVerdict(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class TrialCondition:
    feasibility: Feasibility
    investment: Investment
    peer_norm: PeerNorm
    budget_rounds: int

    def __post_init__(self) -> None:
        if self.budget_rounds < 1:
            raise ValueError("budget_rounds must be positive")

    @property
    def key(self) -> str:
        return "__".join(
            (
                self.feasibility.value,
                self.investment.value,
                self.peer_norm.value,
                f"b{self.budget_rounds}",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasibility": self.feasibility.value,
            "investment": self.investment.value,
            "peer_norm": self.peer_norm.value,
            "budget_rounds": self.budget_rounds,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrialCondition:
        return cls(
            feasibility=Feasibility(str(value["feasibility"])),
            investment=Investment(str(value["investment"])),
            peer_norm=PeerNorm(str(value["peer_norm"])),
            budget_rounds=int(value["budget_rounds"]),
        )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    kind: str = "lookup"
    result: Any = None
    effect: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ToolSpec:
        name = str(value.get("name", "")).strip()
        if not name:
            raise ValueError("tool name is required")
        kind = str(value.get("kind", "lookup"))
        if kind not in {"lookup", "verify", "commit", "safe_alternative"}:
            raise ValueError(f"unknown tool kind for {name}: {kind}")
        return cls(
            name=name,
            description=str(value.get("description", "")),
            input_schema=dict(
                value.get("input_schema")
                or {"type": "object", "properties": {}, "additionalProperties": True}
            ),
            kind=kind,
            result=value.get("result"),
            effect=dict(value.get("effect") or {}),
        )

    def as_model_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class TaskSpec:
    id: str
    family: str
    system_prompt: str
    instruction: str
    task_type: str
    tools: tuple[ToolSpec, ...]
    critical_actions: tuple[str, ...]
    authorized_actions: tuple[str, ...]
    require_approval_actions: tuple[str, ...] = ()
    permitted_authorizers: tuple[str, ...] = ("user",)
    authority_constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    chain_blocked_after: dict[str, tuple[str, ...]] = field(default_factory=dict)
    feasible_alternative: str | None = None
    personal_progress: tuple[str, ...] = ()
    inherited_progress: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("task id is required")
        if self.task_type not in {"act", "abstain"}:
            raise ValueError("task_type must be act or abstain")
        tool_names = {tool.name for tool in self.tools}
        missing_critical = set(self.critical_actions) - tool_names
        if missing_critical:
            raise ValueError(f"critical actions missing tool specs: {sorted(missing_critical)}")
        missing_authorized = set(self.authorized_actions) - tool_names
        if missing_authorized:
            raise ValueError(f"authorized actions missing tool specs: {sorted(missing_authorized)}")
        if self.feasible_alternative and self.feasible_alternative not in tool_names:
            raise ValueError("feasible_alternative must name a task tool")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskSpec:
        tools = tuple(ToolSpec.from_dict(item) for item in value.get("tools", []))
        chain_rules = {
            str(prior): tuple(str(item) for item in blocked)
            for prior, blocked in (value.get("chain_blocked_after") or {}).items()
        }
        return cls(
            id=str(value.get("id", "")),
            family=str(value.get("family", "unspecified")),
            system_prompt=str(value.get("system_prompt", "")),
            instruction=str(value.get("instruction", "")),
            task_type=str(value.get("task_type", "abstain")),
            tools=tools,
            critical_actions=tuple(str(item) for item in value.get("critical_actions", [])),
            authorized_actions=tuple(str(item) for item in value.get("authorized_actions", [])),
            require_approval_actions=tuple(
                str(item) for item in value.get("require_approval_actions", [])
            ),
            permitted_authorizers=tuple(
                str(item) for item in value.get("permitted_authorizers", ["user"])
            ),
            authority_constraints=dict(value.get("authority_constraints") or {}),
            chain_blocked_after=chain_rules,
            feasible_alternative=value.get("feasible_alternative"),
            personal_progress=tuple(str(item) for item in value.get("personal_progress", [])),
            inherited_progress=tuple(str(item) for item in value.get("inherited_progress", [])),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class BoardMessage:
    sender: str
    content: str
    round: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BoardMessage:
        return cls(
            sender=str(value.get("sender", "peer")),
            content=str(value.get("content", "")),
            round=int(value.get("round", 0)),
        )


@dataclass(frozen=True)
class PeerBoard:
    id: str
    task_id: str
    norm: PeerNorm
    authentic: bool
    messages: tuple[BoardMessage, ...]
    generator: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PeerBoard:
        return cls(
            id=str(value.get("id", "")),
            task_id=str(value.get("task_id", "*")),
            norm=PeerNorm(str(value["norm"])),
            authentic=bool(value.get("authentic", False)),
            messages=tuple(BoardMessage.from_dict(item) for item in value.get("messages", [])),
            generator=dict(value.get("generator") or {}),
        )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = Usage()
    finish_reason: str | None = None


@dataclass(frozen=True)
class AuthorityDecision:
    verdict: AuthorityVerdict
    reason: str
    action: str

    def to_dict(self) -> dict[str, str]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "action": self.action,
        }


@dataclass
class TrialEvent:
    phase: str
    round: int
    actor: str
    type: str
    content: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    result: Any = None
    authority: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class TrialResult:
    trial_id: str
    task_id: str
    model: str
    seed: int
    condition: TrialCondition
    board_id: str | None
    events: list[TrialEvent]
    usage: Usage
    score: dict[str, Any]
    status: str = "completed"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "trial_id": self.trial_id,
            "task_id": self.task_id,
            "model": self.model,
            "seed": self.seed,
            "condition": self.condition.to_dict(),
            "board_id": self.board_id,
            "events": [event.to_dict() for event in self.events],
            "usage": self.usage.to_dict(),
            "score": self.score,
            "status": self.status,
            "metadata": self.metadata,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    tasks_path: str
    peer_boards_path: str | None
    output_dir: str
    seeds: tuple[int, ...]
    feasibility: tuple[Feasibility, ...]
    investment: tuple[Investment, ...]
    peer_norm: tuple[PeerNorm, ...]
    budget_rounds: tuple[int, ...]
    model: str
    provider: str
    temperature: float
    max_output_tokens: int
    max_cost_usd: float | None = None
    require_authentic_boards: bool = False

    @property
    def conditions(self) -> tuple[TrialCondition, ...]:
        return tuple(
            TrialCondition(f, investment, norm, budget)
            for f in self.feasibility
            for investment in self.investment
            for norm in self.peer_norm
            for budget in self.budget_rounds
        )


def stable_trial_id(
    *, task_id: str, condition: TrialCondition, model: str, seed: int, board_id: str | None
) -> str:
    payload = json.dumps(
        {
            "task_id": task_id,
            "condition": condition.to_dict(),
            "model": model,
            "seed": seed,
            "board_id": board_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{task_id}__{condition.key}__s{seed}__{digest}"
