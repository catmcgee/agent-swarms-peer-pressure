import json

from swarmstop.qwen_chat import normalize_qwen_messages, parse_qwen_response


def test_normalize_qwen_messages_converts_late_system_and_arguments():
    messages = [
        {"role": "system", "content": "root"},
        {"role": "user", "content": "task"},
        {"role": "system", "content": "authority"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "stop",
                        "arguments": json.dumps({"reason": "denied"}),
                    }
                }
            ],
        },
    ]

    normalized = normalize_qwen_messages(messages)

    assert normalized[0]["role"] == "system"
    assert normalized[2] == {
        "role": "user",
        "content": "<system_notice>\nauthority\n</system_notice>",
    }
    assert normalized[3]["tool_calls"][0]["function"]["arguments"] == {
        "reason": "denied"
    }


def test_parse_qwen_response_removes_reasoning_and_reads_xml_call():
    text = """<think>private trace</think>
I will wait.
<tool_call>
<function=request_authorization>
<parameter=reason>
\"scope denied\"
</parameter>
<parameter=metadata>
{\"round\": 1}
</parameter>
</function>
</tool_call>"""

    content, calls = parse_qwen_response(text)

    assert content == "I will wait."
    assert len(calls) == 1
    assert calls[0].name == "request_authorization"
    assert calls[0].arguments == {"reason": "scope denied", "metadata": {"round": 1}}
