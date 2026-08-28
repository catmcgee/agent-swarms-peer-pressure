from swarmstop.mechanistic import Anchor, AnchorSnapshot, TargetMode
from swarmstop.replay import render_snapshot
from swarmstop.schema import Feasibility, Investment, PeerNorm, TrialCondition


class CharacterTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        assert enable_thinking is False
        tool_text = "|".join(tool["function"]["name"] for tool in tools or [])
        message_text = "".join(
            f"<{message['role']}>{message.get('content') or ''}"
            + "".join(call["function"]["name"] for call in message.get("tool_calls", []))
            for message in messages
        )
        return f"TOOLS:{tool_text}\n{message_text}" + (
            "<assistant>" if add_generation_prompt else ""
        )

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _snapshot(anchor: Anchor, mode: TargetMode, target_text: str | None = None):
    messages = [{"role": "user", "content": "Proceed carefully."}]
    if mode is TargetMode.BEFORE_TEXT:
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": target_text, "arguments": "{}"},
                    }
                ],
            }
        )
    return AnchorSnapshot(
        schema_version=1,
        trial_id="trial-1",
        task_id="task-1",
        task_family="family-1",
        model_id="model-1",
        model_revision="revision-1",
        seed=1,
        condition=TrialCondition(
            feasibility=Feasibility.IMPOSSIBLE,
            investment=Investment.PERSONAL,
            peer_norm=PeerNorm.PROCEED,
            budget_rounds=8,
        ),
        board_id="board-1",
        anchor=anchor,
        messages=tuple(messages),
        tools=(
            {
                "type": "function",
                "function": {"name": "publish_report", "parameters": {}},
            },
        ),
        target_mode=mode,
        target_text=target_text,
    )


def test_last_prompt_target_is_final_generation_marker_token() -> None:
    rendered = render_snapshot(
        _snapshot(Anchor.POST_PEER, TargetMode.LAST_PROMPT_TOKEN), CharacterTokenizer()
    )

    assert rendered.target_index == rendered.token_count - 1
    assert rendered.text.endswith("<assistant>")


def test_action_target_is_token_immediately_before_last_tool_name() -> None:
    rendered = render_snapshot(
        _snapshot(Anchor.ACTION_DECISION, TargetMode.BEFORE_TEXT, "publish_report"),
        CharacterTokenizer(),
    )
    action_start = rendered.text.rfind("publish_report")

    assert rendered.target_char_start == action_start
    assert rendered.target_index == action_start - 1
