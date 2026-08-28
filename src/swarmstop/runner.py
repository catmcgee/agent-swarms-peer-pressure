from __future__ import annotations

import copy
import json
from typing import Any

from .boards import render_board
from .environment import ASSESSMENT_TOOL, DELEGATION_TOOL, SandboxEnvironment
from .mechanistic import Anchor, AnchorSnapshot, SnapshotObserver, TargetMode
from .model import ModelClient
from .schema import (
    Feasibility,
    Investment,
    ModelResponse,
    PeerBoard,
    TaskSpec,
    TrialCondition,
    TrialEvent,
    TrialResult,
    Usage,
    stable_trial_id,
)
from .scoring import score_events


class ControlledTrialRunner:
    def __init__(
        self,
        *,
        temperature: float = 0.7,
        max_output_tokens: int = 800,
        snapshot_observer: SnapshotObserver | None = None,
        model_revision: str | None = None,
    ):
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.snapshot_observer = snapshot_observer
        self.model_revision = model_revision

    def run(
        self,
        *,
        task: TaskSpec,
        condition: TrialCondition,
        model: ModelClient,
        seed: int,
        board: PeerBoard | None,
    ) -> TrialResult:
        if task.metadata.get("runtime_integration_required"):
            raise ValueError(
                "task requires the pinned upstream runtime; the generic sandbox is not valid"
            )
        if condition.peer_norm.value == "absent" and board is not None:
            raise ValueError("absent condition cannot receive a peer board")
        if board is not None and board.norm is not condition.peer_norm:
            raise ValueError("peer board norm does not match condition")

        trial_id = stable_trial_id(
            task_id=task.id,
            condition=condition,
            model=model.model_id,
            seed=seed,
            board_id=board.id if board else None,
        )

        environment = SandboxEnvironment(task)
        events: list[TrialEvent] = []
        usage = Usage()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.instruction},
        ]

        progress = (
            task.personal_progress
            if condition.investment is Investment.PERSONAL
            else task.inherited_progress
        )
        actor = "focal" if condition.investment is Investment.PERSONAL else "previous-worker"
        if progress:
            progress_text = "\n".join(f"- {item}" for item in progress)
            messages.append(
                {
                    "role": "system",
                    "content": f"Progress produced by {actor}:\n{progress_text}",
                }
            )
            events.append(
                TrialEvent(
                    phase="pre_boundary",
                    round=0,
                    actor=actor,
                    type="progress",
                    content=progress_text,
                )
            )

        visible_task_tools = [
            tool
            for tool in task.tools
            if not (
                condition.feasibility is Feasibility.IMPOSSIBLE
                and tool.name == task.feasible_alternative
            )
        ]
        action_tools = [tool.as_model_tool() for tool in visible_task_tools] + [DELEGATION_TOOL]
        self._capture_snapshot(
            trial_id=trial_id,
            task=task,
            condition=condition,
            model_id=model.model_id,
            seed=seed,
            board=board,
            anchor=Anchor.PRE_AUTHORITY,
            messages=messages,
            tools=action_tools,
        )

        boundary = environment.authority.boundary_notice(
            feasible=condition.feasibility is Feasibility.FEASIBLE
        )
        boundary_text = "AUTHORITY_CHECK_RESULT\n" + json.dumps(boundary, sort_keys=True)
        messages.append({"role": "system", "content": boundary_text})
        events.append(
            TrialEvent(
                phase="recognition",
                round=0,
                actor="authority-oracle",
                type="boundary_notice",
                content=boundary_text,
                authority=boundary["authority"],
            )
        )

        recognition = model.complete(
            messages=messages,
            tools=[ASSESSMENT_TOOL],
            seed=seed,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        usage = usage + recognition.usage
        self._record_response(
            response=recognition,
            phase="recognition",
            round_number=0,
            messages=messages,
            environment=environment,
            events=events,
        )
        self._capture_snapshot(
            trial_id=trial_id,
            task=task,
            condition=condition,
            model_id=model.model_id,
            seed=seed,
            board=board,
            anchor=Anchor.POST_RECOGNITION,
            messages=messages,
            tools=action_tools,
        )

        board_text = render_board(board)
        messages.append({"role": "system", "content": board_text})
        events.append(
            TrialEvent(
                phase="peer_exposure",
                round=0,
                actor="message-board",
                type="board",
                content=board_text,
            )
        )
        self._capture_snapshot(
            trial_id=trial_id,
            task=task,
            condition=condition,
            model_id=model.model_id,
            seed=seed,
            board=board,
            anchor=Anchor.POST_PEER,
            messages=messages,
            tools=action_tools,
        )

        for round_number in range(1, condition.budget_rounds + 1):
            response = model.complete(
                messages=messages,
                tools=action_tools,
                seed=seed + round_number,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            usage = usage + response.usage
            if round_number == 1:
                action_messages = [*messages, self._assistant_message(response)]
                target_text = (
                    response.tool_calls[0].name
                    if response.tool_calls
                    else (response.content if response.content.strip() else None)
                )
                self._capture_snapshot(
                    trial_id=trial_id,
                    task=task,
                    condition=condition,
                    model_id=model.model_id,
                    seed=seed,
                    board=board,
                    anchor=Anchor.ACTION_DECISION,
                    messages=action_messages,
                    tools=action_tools,
                    target_mode=(
                        TargetMode.BEFORE_TEXT
                        if target_text is not None
                        else TargetMode.LAST_PROMPT_TOKEN
                    ),
                    target_text=target_text,
                )
            critical = self._record_response(
                response=response,
                phase="post_recognition",
                round_number=round_number,
                messages=messages,
                environment=environment,
                events=events,
            )
            safe_alternative_used = bool(
                task.feasible_alternative
                and any(call.name == task.feasible_alternative for call in response.tool_calls)
            )
            if critical or safe_alternative_used or not response.tool_calls:
                break

        return TrialResult(
            trial_id=trial_id,
            task_id=task.id,
            model=model.model_id,
            seed=seed,
            condition=condition,
            board_id=board.id if board else None,
            events=events,
            usage=usage,
            score=score_events(task, events),
            metadata={
                "board_authentic": board.authentic if board else None,
                "model_revision": self.model_revision,
                "boundary": boundary,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
        )

    @staticmethod
    def _record_response(
        *,
        response: ModelResponse,
        phase: str,
        round_number: int,
        messages: list[dict[str, Any]],
        environment: SandboxEnvironment,
        events: list[TrialEvent],
    ) -> bool:
        events.append(
            TrialEvent(
                phase=phase,
                round=round_number,
                actor="focal",
                type="assistant",
                content=response.content,
            )
        )
        messages.append(ControlledTrialRunner._assistant_message(response))

        critical = False
        for call in response.tool_calls:
            record = environment.execute(call.name, call.arguments)
            critical = critical or record.critical_attempt
            authority = record.authority.to_dict()
            authority["critical_attempt"] = record.critical_attempt
            authority["delegated_attempt"] = record.delegated_attempt
            events.append(
                TrialEvent(
                    phase=phase,
                    round=round_number,
                    actor="focal",
                    type="tool_call",
                    tool=call.name,
                    arguments=call.arguments,
                    result=record.result,
                    authority=authority,
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(record.result, sort_keys=True, default=str),
                }
            )
        return critical

    @staticmethod
    def _assistant_message(response: ModelResponse) -> dict[str, Any]:
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": response.content or None,
        }
        if response.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, sort_keys=True),
                    },
                }
                for call in response.tool_calls
            ]
        return assistant_message

    def _capture_snapshot(
        self,
        *,
        trial_id: str,
        task: TaskSpec,
        condition: TrialCondition,
        model_id: str,
        seed: int,
        board: PeerBoard | None,
        anchor: Anchor,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        target_mode: TargetMode = TargetMode.LAST_PROMPT_TOKEN,
        target_text: str | None = None,
    ) -> None:
        if self.snapshot_observer is None:
            return
        self.snapshot_observer.capture(
            AnchorSnapshot(
                schema_version=1,
                trial_id=trial_id,
                task_id=task.id,
                task_family=task.family,
                model_id=model_id,
                model_revision=self.model_revision,
                seed=seed,
                condition=condition,
                board_id=board.id if board else None,
                anchor=anchor,
                messages=tuple(copy.deepcopy(messages)),
                tools=tuple(copy.deepcopy(tools)),
                target_mode=target_mode,
                target_text=target_text,
            )
        )
