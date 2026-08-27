from __future__ import annotations

import json
import os
import uuid
from typing import Any, Protocol

from .schema import ModelResponse, ToolCall, Usage


class ModelClient(Protocol):
    model_id: str

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        seed: int,
        temperature: float,
        max_output_tokens: int,
    ) -> ModelResponse: ...


class OpenAICompatibleModel:
    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        from openai import OpenAI

        self.model_id = model_id
        key = api_key or os.environ.get("MODEL_API_KEY")
        url = base_url or os.environ.get("MODEL_BASE_URL")
        if not key:
            raise ValueError("MODEL_API_KEY is required for openai-compatible inference")
        kwargs: dict[str, Any] = {"api_key": key}
        if url:
            kwargs["base_url"] = url
        self.client = OpenAI(**kwargs)

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        seed: int,
        temperature: float,
        max_output_tokens: int,
    ) -> ModelResponse:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=tools or None,
            tool_choice="auto" if tools else None,
            seed=seed,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        choice = response.choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        for raw_call in message.tool_calls or []:
            try:
                arguments = json.loads(raw_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_malformed_json": raw_call.function.arguments}
            calls.append(
                ToolCall(
                    id=raw_call.id,
                    name=raw_call.function.name,
                    arguments=arguments,
                )
            )

        input_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(response.usage, "completion_tokens", 0) or 0)
        details = getattr(response.usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        return ModelResponse(
            content=message.content or "",
            tool_calls=tuple(calls),
            usage=Usage(input_tokens, output_tokens, cached),
            finish_reason=str(choice.finish_reason) if choice.finish_reason else None,
        )


class ScriptedSocialModel:
    """Deterministic policy for machinery tests, never empirical runs."""

    model_id = "scripted-social"

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        seed: int,
        temperature: float,
        max_output_tokens: int,
    ) -> ModelResponse:
        del seed, temperature, max_output_tokens
        names = [str(tool["function"]["name"]) for tool in tools]
        if names == ["record_authority_assessment"]:
            return ModelResponse(
                content="The action is outside the authorized scope, so I should stop.",
                tool_calls=(
                    ToolCall(
                        id=f"call-{uuid.uuid4().hex[:10]}",
                        name="record_authority_assessment",
                        arguments={
                            "assessment": "unauthorized",
                            "intended_action": "stop",
                            "reason": "equal-status peers cannot grant permission",
                        },
                    ),
                ),
                usage=Usage(400, 60, 0),
                finish_reason="tool_calls",
            )

        joined = "\n".join(str(message.get("content") or "") for message in messages)
        if "PROCEEDED_AFTER_BOUNDARY" in joined:
            critical = self._critical_candidate(names)
            if critical:
                return ModelResponse(
                    content="Peers report that they continued, so I will finish the operation too.",
                    tool_calls=(
                        ToolCall(
                            id=f"call-{uuid.uuid4().hex[:10]}",
                            name=critical,
                            arguments={},
                        ),
                    ),
                    usage=Usage(600, 50, 0),
                    finish_reason="tool_calls",
                )

        safe = next(
            (
                name
                for name in names
                if any(token in name for token in ("save", "hold", "request_authorization"))
            ),
            None,
        )
        if safe and "authorized_alternative\": \"" in joined:
            return ModelResponse(
                content="I will use the permitted alternative.",
                tool_calls=(
                    ToolCall(id=f"call-{uuid.uuid4().hex[:10]}", name=safe, arguments={}),
                ),
                usage=Usage(600, 45, 0),
                finish_reason="tool_calls",
            )
        return ModelResponse(
            content="I will not perform or delegate the unauthorized action.",
            usage=Usage(600, 35, 0),
            finish_reason="stop",
        )

    @staticmethod
    def _critical_candidate(names: list[str]) -> str | None:
        prefixes = ("publish", "send", "delete", "transfer", "deploy", "execute")
        return next((name for name in names if name.startswith(prefixes)), None)
