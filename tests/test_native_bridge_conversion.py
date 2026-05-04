import json

from llamaherd import proxy


def test_native_bridge_converts_openai_tool_history_to_ollama_shape():
    req = {
        "model": "glm-5.1",
        "messages": [
            {"role": "user", "content": "Use the weather tool"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Sydney"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": '{"temp_c":22}',
            },
            {"role": "user", "content": "Answer briefly"},
        ],
        "stream": True,
    }

    body = json.loads(proxy._convert_openai_to_ollama_body(req))
    assistant = body["messages"][1]
    tool = body["messages"][2]

    assert assistant["content"] == ""
    assert assistant["tool_calls"] == [
        {
            "function": {
                "name": "get_weather",
                "arguments": {"city": "Sydney"},
            }
        }
    ]
    assert "id" not in assistant["tool_calls"][0]
    assert "type" not in assistant["tool_calls"][0]
    assert "tool_call_id" not in tool
    assert tool["tool_name"] == "get_weather"


def test_native_bridge_translates_reasoning_to_thinking():
    req = {
        "model": "glm-5.1",
        "messages": [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning": "private chain",
                "refusal": None,
            }
        ],
        "reasoning_effort": "high",
    }

    body = json.loads(proxy._convert_openai_to_ollama_body(req))
    msg = body["messages"][0]
    assert msg == {"role": "assistant", "content": "Answer", "thinking": "private chain"}
    assert body["think"] == "high"


def test_native_bridge_streams_ollama_thinking_as_openai_reasoning():
    line = proxy._ollama_chunk_to_sse(
        {
            "model": "glm-5.1",
            "message": {"role": "assistant", "thinking": "step one"},
            "done": False,
        },
        "chatcmpl-test",
        "glm-5.1",
    )
    assert line is not None
    payload = json.loads(line.removeprefix("data: "))
    assert payload["choices"][0]["delta"] == {"reasoning": "step one"}
