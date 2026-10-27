import json
from typing import Any
from uuid import uuid4

from harness.errors import ModelError
from harness.messages import Message, ModelResponse, ToolCall, ToolSpec


# --- harness -> provider ---------------------------------------------------------------


def _to_provider_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.args,  # the JSON Schema that @tool built
        },
    }


def _to_provider_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}

    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        out["tool_calls"] = [
            {
                "id": c.tool_id,  # replayed unchanged, so the tool message's echo matches
                "type": "function",
                "function": {
                    "name": c.tool_name,
                    # The provider wants a JSON *string*, not a dict.
                    "arguments": json.dumps(c.tool_args, ensure_ascii=False, default=str),
                },
            }
            for c in m.tool_calls
        ]
    return out


# --- provider -> harness ---------------------------------------------------------------


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """(args, error). A bad payload is reported to the model, not raised.

    Models sometimes send cut-off or non-object JSON. Raising ModelError would kill a
    long run over one slip; instead the error travels on ToolCall.arg_error and the
    model is told to resend, exactly like a typo in a tool name.
    """
    if isinstance(raw, dict):  # some providers already parse it
        return raw, None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None  # no-argument tools: some models send "" or nothing
    if not isinstance(raw, str):
        return {}, f"arguments must be a JSON object, got {type(raw).__name__}"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, f"arguments were not valid JSON ({e.msg} at position {e.pos}); resend a complete JSON object"
    if not isinstance(value, dict):
        return {}, "arguments must be a JSON object"
    return value, None


def _to_model_response(raw: Any) -> ModelResponse:
    try:
        message = raw.choices[0].message
        seen: set[str] = set()
        calls: list[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or ():
            args, arg_error = _parse_arguments(tc.function.arguments)
            # Keep the provider's id, but make one up if it is missing, or reused within
            # this turn (some local models send "call_0" for every call). Our id is the
            # one replayed later, so results always match their calls.
            call_id = getattr(tc, "id", None)
            if not call_id or call_id in seen:
                call_id = f"call_{uuid4().hex[:24]}"
            seen.add(call_id)
            calls.append(ToolCall(
                    tool_name=tc.function.name or "",
                    tool_id=call_id,
                    tool_args=args,
                    arg_error=arg_error
                )
            )
        content = getattr(message, "content", None)
    except (AttributeError, IndexError, TypeError) as e:
        raise ModelError("unexpected response shape from the provider") from e

    text = content if isinstance(content, str) and content.strip() else None
    if text is None and not calls:
        raise ModelError("the model returned an empty response")  # Message.assistant would reject it
    return ModelResponse(response=text, tool_calls=tuple(calls))


def _short(text: str, limit: int = 300) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first if len(first) <= limit else first[: limit - 3] + "..."


# --- the client -------------------------------------------------------------------------


class LiteLLMClient:
    """One model turn through LiteLLM: works for API models and local ones (ollama/...)."""

    def __init__(self, model: str, *, temperature: float | None = None, max_tokens: int | None = None, timeout: float | None = 60.0, num_retries: int = 2, **extra: Any) -> None:  # extra is for api_base, api_key, ... passed straight to litellm.completion)
        try:
            import litellm  # lazy: `import harness` stays fast, and tests need no litellm
        except ImportError as e:
            raise ImportError("LiteLLMClient needs: pip install litellm") from e
        self._litellm = litellm
        self.model = model
        base = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "num_retries": num_retries,
        }
        self._params: dict[str, Any] = {k: v for k, v in base.items() if v is not None}
        self._params.update(extra)

    def generate(self, messages: tuple[Message, ...], tools: tuple[ToolSpec, ...]) -> ModelResponse:
        # Translate first, outside the try: a bug here is ours, not a provider failure.
        payload = [_to_provider_message(m) for m in messages]
        kwargs = dict(self._params)
        if tools:  # some providers reject an empty tools list
            kwargs["tools"] = [_to_provider_tool(t) for t in tools]

        try:
            raw = self._litellm.completion(model=self.model, messages=payload, **kwargs)
        except Exception as e:  # rate limit, timeout, auth, bad request, connection...
            raise ModelError(f"{type(e).__name__}: {_short(str(e))}") from e
        return _to_model_response(raw)