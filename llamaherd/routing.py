import json
import time
from typing import Any


def _parse_openai_tool_args(args: Any) -> Any:
    """Convert OpenAI JSON-string tool arguments to Ollama's native object shape."""
    if isinstance(args, str):
        try:
            return json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return args
    return args


def _openai_tool_calls_to_ollama(tool_calls: Any) -> list[dict]:
    """Convert OpenAI assistant tool_calls to Ollama native tool_calls."""
    out: list[dict] = []
    if not isinstance(tool_calls, list):
        return out
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if not isinstance(fn, dict):
            continue
        native = {
            "function": {
                "name": fn.get("name", ""),
                "arguments": _parse_openai_tool_args(fn.get("arguments", {})),
            }
        }
        out.append(native)
    return out


def _convert_openai_messages_to_ollama(messages: Any) -> list[dict]:
    """Convert OpenAI chat messages to Ollama native /api/chat messages."""
    if not isinstance(messages, list):
        return []

    # Map OpenAI tool_call IDs to names so following role=tool messages can use
    # Ollama's native tool_name field instead of OpenAI's tool_call_id.
    tool_id_to_name: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            fn = call.get("function") or {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if call_id and name:
                tool_id_to_name[str(call_id)] = str(name)

    converted: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        out: dict = {}
        if role:
            out["role"] = role

        # Ollama expects content to be a string. OpenAI sometimes uses null for
        # assistant tool-call messages; normalize that to an empty string.
        if "content" in msg:
            out["content"] = "" if msg.get("content") is None else msg.get("content")

        if "images" in msg:
            out["images"] = msg["images"]
        if "name" in msg:
            out["name"] = msg["name"]

        if "tool_calls" in msg:
            native_calls = _openai_tool_calls_to_ollama(msg["tool_calls"])
            if native_calls:
                out["tool_calls"] = native_calls

        # OpenAI uses tool_call_id on tool-result messages. Ollama native uses
        # tool_name. Translate when possible, otherwise omit the unsupported ID.
        if "tool_name" in msg:
            out["tool_name"] = msg["tool_name"]
        elif "tool_call_id" in msg:
            name = tool_id_to_name.get(str(msg["tool_call_id"]))
            if name:
                out["tool_name"] = name

        # OpenAI/Ollama Cloud call this reasoning; Ollama native calls it thinking.
        if "thinking" in msg:
            out["thinking"] = msg["thinking"]
        elif "reasoning" in msg:
            out["thinking"] = msg["reasoning"]

        converted.append(out)
    return converted


def _convert_openai_to_ollama_body(req_json: dict) -> bytes:
    """Convert an OpenAI /v1/chat/completions request body to Ollama /api/chat format.

    Maps: messages, tools, max_tokens→options.num_predict, temperature→options.temperature,
    top_p→options.top_p, stream, model.  OpenAI-specific fields (stream_options, n, etc.)
    are dropped.
    """
    ollama: dict = {"model": req_json.get("model", "")}
    if "messages" in req_json:
        ollama["messages"] = _convert_openai_messages_to_ollama(req_json["messages"])
    if "stream" in req_json:
        ollama["stream"] = req_json["stream"]
    if "tools" in req_json:
        ollama["tools"] = req_json["tools"]

    # OpenAI-compatible reasoning controls → Ollama native thinking controls.
    # Ollama native accepts: think=true/false/"low"/"medium"/"high".
    if "think" in req_json:
        ollama["think"] = req_json["think"]
    elif "reasoning_effort" in req_json:
        effort = req_json.get("reasoning_effort")
        ollama["think"] = False if effort in (None, "none", "off", "false") else effort
    elif "reasoning" in req_json:
        reasoning = req_json.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort") or reasoning.get("level")
            if effort:
                ollama["think"] = False if effort in ("none", "off", "false") else effort
        elif isinstance(reasoning, (bool, str)):
            ollama["think"] = reasoning

    # Pack OpenAI kwargs into Ollama options dict
    options: dict = {}
    if "max_tokens" in req_json:
        options["num_predict"] = req_json["max_tokens"]
    if "temperature" in req_json:
        options["temperature"] = req_json["temperature"]
    if "top_p" in req_json:
        options["top_p"] = req_json["top_p"]
    if "frequency_penalty" in req_json:
        options["frequency_penalty"] = req_json["frequency_penalty"]
    if "presence_penalty" in req_json:
        options["presence_penalty"] = req_json["presence_penalty"]
    if "seed" in req_json:
        options["seed"] = req_json["seed"]
    if "stop" in req_json:
        # Ollama uses 'stop' directly at top level
        ollama["stop"] = req_json["stop"]
    if options:
        ollama["options"] = options

    return json.dumps(ollama).encode()


def _convert_ollama_tool_calls(ollama_tools: list[dict]) -> list[dict]:
    """Convert Ollama tool_calls format to OpenAI streaming format.

    Ollama: {"function": {"name": "x", "arguments": {dict}}}
    OpenAI: {"index": 0, "id": "call_xxx", "type": "function",
             "function": {"name": "x", "arguments": "{json_string}"}}
    """
    openai_tools = []
    for i, tc in enumerate(ollama_tools or []):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        # Ollama returns arguments as a dict; OpenAI wants a JSON string
        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
        openai_tools.append({
            "index": i,
            # Generate a deterministic-ish call ID from function name
            "id": f"call_{fn.get('name', 'unknown')}_{i}",
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "arguments": args_str,
            },
        })
    return openai_tools


def _ollama_chunk_to_sse(ollama_chunk: dict, chunk_id: str, model: str) -> str | None:
    """Convert a single Ollama /api/chat NDJSON chunk to an OpenAI SSE data line.

    Returns None for chunks that shouldn't be emitted (empty content, non-message chunks).
    Returns the SSE line WITHOUT the trailing \\n\\n (caller adds SSE framing).
    """
    # Only process chunks with a "message" field (content or tool_calls chunks)
    if "message" not in ollama_chunk and not ollama_chunk.get("done", False):
        return None

    done = ollama_chunk.get("done", False)
    message = ollama_chunk.get("message", {})

    # Build the OpenAI streaming chunk
    choices: list[dict] = []
    usage: dict | None = None

    if done:
        # Final chunk: emit finish_reason and usage
        done_reason = ollama_chunk.get("done_reason", "stop")
        # Map Ollama done_reason to OpenAI finish_reason
        finish_reason = "length" if done_reason == "length" else (
            "tool_calls" if message.get("tool_calls") else "stop"
        )
        delta: dict = {}
        # If the final chunk has tool_calls, include them
        if message.get("tool_calls"):
            delta["tool_calls"] = _convert_ollama_tool_calls(message["tool_calls"])
        choices.append({"index": 0, "delta": delta, "finish_reason": finish_reason})

        # Usage from final NDJSON chunk
        pev = ollama_chunk.get("prompt_eval_count")
        ev = ollama_chunk.get("eval_count")
        if pev is not None or ev is not None:
            usage = {
                "prompt_tokens": int(pev or 0),
                "completion_tokens": int(ev or 0),
                "total_tokens": int(pev or 0) + int(ev or 0),
            }
    else:
        # Content chunk
        content = message.get("content", "")
        delta: dict = {}
        # Reasoning/thinking chunk. Ollama native emits message.thinking;
        # OpenAI-compatible clients expect reasoning on the delta.
        thinking = message.get("thinking", "")
        if thinking:
            delta["reasoning"] = thinking
        if content:
            delta["content"] = content
        # Tool calls chunk
        if message.get("tool_calls"):
            delta["tool_calls"] = _convert_ollama_tool_calls(message["tool_calls"])
            delta["content"] = None  # OpenAI: content is null when tool_calls present
        if not delta:
            return None  # Empty delta, skip
        choices.append({"index": 0, "delta": delta, "finish_reason": None})

    chunk: dict = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage:
        chunk["usage"] = usage

    return f"data: {json.dumps(chunk)}"
