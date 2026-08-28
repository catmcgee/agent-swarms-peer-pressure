from __future__ import annotations

import copy
import json
import re
import uuid
from typing import Any

from .schema import ToolCall

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def normalize_qwen_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert generic chat history to the pinned Qwen 3.5 template contract."""
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(messages):
        message = copy.deepcopy(raw)
        if message.get("role") == "system" and index != 0:
            message["role"] = "user"
            content = message.get("content") or ""
            message["content"] = f"<system_notice>\n{content}\n</system_notice>"
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or call
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        function["arguments"] = {"_malformed_json": arguments}
        normalized.append(message)
    return normalized


def render_qwen_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    return str(
        tokenizer.apply_chat_template(
            normalize_qwen_messages(messages),
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    )


def parse_qwen_response(text: str) -> tuple[str, tuple[ToolCall, ...]]:
    calls: list[ToolCall] = []
    for tool_match in _TOOL_CALL_RE.finditer(text):
        function_match = _FUNCTION_RE.fullmatch(tool_match.group(1).strip())
        if function_match is None:
            continue
        arguments: dict[str, Any] = {}
        for parameter in _PARAMETER_RE.finditer(function_match.group(2)):
            name = parameter.group(1).strip()
            raw_value = parameter.group(2).strip()
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            arguments[name] = value
        calls.append(
            ToolCall(
                id=f"call-{uuid.uuid4().hex[:10]}",
                name=function_match.group(1).strip(),
                arguments=arguments,
            )
        )

    visible = _TOOL_CALL_RE.sub("", text)
    visible = _THINK_RE.sub("", visible)
    if "<think>" in visible:
        visible = visible.split("<think>", 1)[0]
    return visible.strip(), tuple(calls)
