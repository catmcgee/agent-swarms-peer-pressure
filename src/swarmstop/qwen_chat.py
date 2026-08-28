from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from .schema import ToolCall

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_RAW_FUNCTION_NAME_RE = re.compile(r"<function=\s*([^>\n]+?)\s*>")
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


def parse_qwen_response(
    text: str, *, call_id_prefix: str | None = None
) -> tuple[str, tuple[ToolCall, ...]]:
    calls: list[ToolCall] = []
    for call_index, tool_match in enumerate(_TOOL_CALL_RE.finditer(text)):
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
                id=(
                    f"{call_id_prefix}-{call_index}"
                    if call_id_prefix
                    else "call-"
                    + hashlib.sha256(
                        f"{call_index}:{tool_match.group(0)}".encode()
                    ).hexdigest()[:12]
                ),
                name=function_match.group(1).strip(),
                arguments=arguments,
            )
        )

    visible = _TOOL_CALL_RE.sub("", text)
    visible = _THINK_RE.sub("", visible)
    if "<think>" in visible:
        visible = visible.split("<think>", 1)[0]
    return visible.strip(), tuple(calls)


def has_malformed_tool_markup(text: str, calls: tuple[ToolCall, ...]) -> bool:
    """Detect unmatched or unparsable XML tool fragments, including mixed responses."""
    complete_blocks = list(_TOOL_CALL_RE.finditer(text))
    if len(complete_blocks) != len(calls):
        return True
    for block in complete_blocks:
        function_match = _FUNCTION_RE.fullmatch(block.group(1).strip())
        if function_match is None:
            return True
        body = function_match.group(2)
        if any(marker in body for marker in ("<function=", "</function>")):
            return True
        parameters = list(_PARAMETER_RE.finditer(body))
        names = [match.group(1).strip() for match in parameters]
        if len(names) != len(set(names)):
            return True
        if _PARAMETER_RE.sub("", body).strip():
            return True
    residual = _TOOL_CALL_RE.sub("", text)
    markers = (
        "<tool_call",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    )
    return any(marker in residual for marker in markers)


def raw_function_names(text: str) -> tuple[str, ...]:
    """Extract normalized function names even from truncated tool markup."""
    return tuple(match.group(1).strip() for match in _RAW_FUNCTION_NAME_RE.finditer(text))


def coerce_tool_arguments(
    calls: tuple[ToolCall, ...], tools: list[dict[str, Any]]
) -> tuple[tuple[ToolCall, ...], bool]:
    """Coerce Qwen XML parameter values to the advertised JSON-schema types."""
    schemas = {
        str(item["function"]["name"]): dict(
            (item["function"].get("parameters") or {}).get("properties") or {}
        )
        for item in tools
    }
    changed = False
    normalized: list[ToolCall] = []
    for call in calls:
        properties = schemas.get(call.name, {})
        arguments: dict[str, Any] = {}
        for name, value in call.arguments.items():
            expected = str((properties.get(name) or {}).get("type", ""))
            coerced = _coerce_value(value, expected)
            changed = changed or coerced != value or type(coerced) is not type(value)
            arguments[name] = coerced
        normalized.append(ToolCall(call.id, call.name, arguments))
    return tuple(normalized), changed


def tool_calls_match_schemas(
    calls: tuple[ToolCall, ...], tools: list[dict[str, Any]]
) -> bool:
    """Validate structural fields/types/enums; environment policy handles string patterns."""
    schemas = {
        str(item["function"]["name"]): dict(
            item["function"].get("parameters") or {}
        )
        for item in tools
    }
    for call in calls:
        schema = schemas.get(call.name)
        if schema is None:
            return False
        required = {str(item) for item in schema.get("required") or []}
        if not required.issubset(call.arguments):
            return False
        properties = dict(schema.get("properties") or {})
        if schema.get("additionalProperties") is False and (
            set(call.arguments) - set(properties)
        ):
            return False
        for name, value in call.arguments.items():
            rule = dict(properties.get(name) or {})
            if not _value_matches_schema(value, rule):
                return False
    return True


def _coerce_value(value: Any, expected: str) -> Any:
    if expected == "string" and not isinstance(value, str):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return str(value)
    if expected == "integer" and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    if expected == "number" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if expected == "boolean" and isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    if expected in {"array", "object"} and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if expected == "array" and isinstance(parsed, list):
            return parsed
        if expected == "object" and isinstance(parsed, dict):
            return parsed
    return value


def _value_matches_schema(value: Any, rule: dict[str, Any]) -> bool:
    expected = str(rule.get("type", ""))
    type_matches = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }
    if expected and not type_matches.get(expected, True):
        return False
    if "enum" in rule and value not in (rule.get("enum") or []):
        return False
    return True
