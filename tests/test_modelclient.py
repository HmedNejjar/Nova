#type: ignore
import json
import sys
from types import SimpleNamespace as NS

import pytest

from NovaHarness.harness.errors import ModelError
from NovaHarness.harness.messages import Message, ModelResponse, ToolCall, ToolResult, ToolSpec
from NovaHarness.models.modelClient import LiteLLMClient, _parse_arguments, _to_model_response, _to_provider_message, _to_provider_tool


def reply(content=None, tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


def call(name="read_file", arguments='{"path": "a.txt"}', id="call_1"):
    return NS(id=id, function=NS(name=name, arguments=arguments))


# --- provider -> harness -----------------------------------------------------------


def test_text_only_reply():
    r = _to_model_response(reply(content="hi", tool_calls=None))
    assert r == ModelResponse(response="hi", tool_calls=())


def test_tool_call_is_parsed():
    r = _to_model_response(reply(tool_calls=[call()]))
    (c,) = r.tool_calls
    assert (c.tool_name, c.tool_id, c.tool_args, c.arg_error) == (
        "read_file", "call_1", {"path": "a.txt"}, None,
    )
    assert r.response is None


def test_empty_string_content_with_calls_becomes_none():
    r = _to_model_response(reply(content="", tool_calls=[call()]))
    assert r.response is None


def test_missing_id_is_generated():
    (c,) = _to_model_response(reply(tool_calls=[call(id=None)])).tool_calls
    assert c.tool_id.startswith("call_") and len(c.tool_id) > 5


def test_duplicate_ids_in_one_turn_are_made_unique():
    r = _to_model_response(reply(tool_calls=[call(id="call_0"), call(id="call_0")]))
    assert len({c.tool_id for c in r.tool_calls}) == 2


def test_parallel_calls_keep_their_order():
    r = _to_model_response(reply(tool_calls=[call("a", "{}", "x"), call("b", "{}", "y")]))
    assert [c.tool_name for c in r.tool_calls] == ["a", "b"]


@pytest.mark.parametrize(
    "raw, expected",
    [('{"a": 1}', {"a": 1}), ({"a": 1}, {"a": 1}), ("", {}), ("  ", {}), (None, {})],
)
def test_usable_arguments(raw, expected):
    assert _parse_arguments(raw) == (expected, None)


@pytest.mark.parametrize("raw", ['{"path": "a.txt"', "[1, 2]", '"text"', "{'a': 1}", 5])
def test_unusable_arguments_are_reported_not_raised(raw):
    args, error = _parse_arguments(raw)
    assert args == {} and error


def test_bad_json_reaches_the_call_as_arg_error():
    (c,) = _to_model_response(reply(tool_calls=[call(arguments='{"path": ')])).tool_calls
    assert c.tool_args == {} and "valid JSON" in c.arg_error


def test_empty_reply_is_a_model_error():
    with pytest.raises(ModelError):
        _to_model_response(reply(content=None, tool_calls=None))
    with pytest.raises(ModelError):
        _to_model_response(reply(content="  ", tool_calls=[]))


def test_unexpected_shape_is_a_model_error():
    with pytest.raises(ModelError):
        _to_model_response(NS(choices=[]))
    with pytest.raises(ModelError):
        _to_model_response(NS())


# --- harness -> provider -----------------------------------------------------------


def test_user_and_system_messages():
    assert _to_provider_message(Message.user("hi")) == {"role": "user", "content": "hi"}
    assert _to_provider_message(Message.system("be brief")) == {  # needs the system role
        "role": "system", "content": "be brief",
    }


def test_assistant_text_has_no_tool_calls_key():
    m = Message.assistant(ModelResponse(response="done"))
    assert "tool_calls" not in _to_provider_message(m)


def test_assistant_tool_calls_use_json_string_arguments():
    c = ToolCall(tool_name="read_file", tool_id="c1", tool_args={"q": "你好"})
    out = _to_provider_message(Message.assistant(ModelResponse(tool_calls=(c,))))
    assert out["content"] is None
    (tc,) = out["tool_calls"]
    assert tc["id"] == "c1" and tc["type"] == "function"
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"q": "你好"}
    assert "你好" in tc["function"]["arguments"]  # not \u-escaped


def test_tool_message_echoes_the_call_id():
    m = Message.tool(ToolResult(tool_name="t", tool_id="c1", status="ok", tool_result="out"))
    assert _to_provider_message(m) == {"role": "tool", "tool_call_id": "c1", "content": "out"}


def test_tool_spec_shape():
    spec = ToolSpec(name="t", description="d", args={"type": "object", "properties": {}, "required": []})
    assert _to_provider_tool(spec) == {
        "type": "function",
        "function": {"name": "t", "description": "d", "parameters": spec.args},
    }


def test_round_trip_keeps_ids_matching():
    parsed = _to_model_response(reply(tool_calls=[call(id=None), call(id=None)]))
    assistant = _to_provider_message(Message.assistant(parsed))
    sent_ids = [tc["id"] for tc in assistant["tool_calls"]]
    results = [
        _to_provider_message(Message.tool(ToolResult(c.tool_name, c.tool_id, "ok", "x")))
        for c in parsed.tool_calls
    ]
    assert [r["tool_call_id"] for r in results] == sent_ids


# --- generate() with a fake litellm module ------------------------------------------


@pytest.fixture
def fake_litellm(monkeypatch):
    seen: dict = {}

    def completion(**kwargs):
        seen.update(kwargs)
        if "boom" in seen:
            raise RuntimeError("rate limited\nsecond line")
        return reply(content="ok")

    monkeypatch.setitem(sys.modules, "litellm", NS(completion=completion))
    return seen


def test_generate_sends_translated_messages_and_tools(fake_litellm):
    spec = ToolSpec(name="t", description="d", args={"type": "object", "properties": {}, "required": []})
    r = LiteLLMClient("gpt-x").generate((Message.user("hi"),), (spec,))
    assert r.response == "ok"
    assert fake_litellm["model"] == "gpt-x"
    assert fake_litellm["messages"] == [{"role": "user", "content": "hi"}]
    assert fake_litellm["tools"][0]["function"]["name"] == "t"


def test_generate_omits_tools_when_there_are_none(fake_litellm):
    LiteLLMClient("gpt-x").generate((Message.user("hi"),), ())
    assert "tools" not in fake_litellm


def test_generate_passes_options_through(fake_litellm):
    LiteLLMClient("m", temperature=0.2, api_base="http://localhost:11434").generate((Message.user("hi"),), ())
    assert fake_litellm["temperature"] == 0.2
    assert fake_litellm["api_base"] == "http://localhost:11434"
    assert fake_litellm["num_retries"] == 2


def test_provider_exceptions_become_model_error_with_the_cause(fake_litellm):
    fake_litellm["boom"] = True
    with pytest.raises(ModelError) as info:
        LiteLLMClient("m").generate((Message.user("hi"),), ())
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "RuntimeError" in str(info.value) and "second line" not in str(info.value)