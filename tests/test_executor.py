#type: ignore

import time
import pytest

from NovaHarness.harness.errors import InvalidArgumentsError, ToolError, ToolExecutionError, ToolNotFoundError
from NovaHarness.harness.executor import Executor
from NovaHarness.harness.messages import ToolCall
from NovaHarness.tools.tool import tool


def call(name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(tool_name=name, tool_id="c1", tool_args=args or {})


@tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"hello {name}"


@tool
def add(a: int, b: int = 1) -> int:
    """Add two integers."""
    return a + b


@tool
def unique(items: list[int]) -> set[int]:
    """Deduplicate a list of ints."""
    return set(items)


@tool
def boom() -> None:
    """Always raise a recoverable ToolError."""
    raise ToolError("boom")


@tool
def crash() -> None:
    """Always crash with a non-ToolError."""
    raise ValueError("crash")


@tool
def slow() -> str:
    """Sleep far longer than the executor timeout."""
    time.sleep(5)
    return "done"


def test_to_text():
    ex = Executor({}, max_output_len=10)
    assert ex._to_text(None) == "(no output)"
    assert ex._to_text("hi") == "hi"
    assert ex._to_text({"a": 1}) == '{"a": 1}'
    assert ex._to_text("x" * 50).startswith("x" * 10 + "\n... [truncated")


def test_run_returns_ok_result_and_echoes_meta():
    ex = Executor({"greet": greet})
    result = ex.run(call("greet", {"name": "Ann"}))
    assert result.status == "ok"
    assert result.tool_name == "greet"
    assert result.tool_id == "c1"
    assert result.tool_result == "hello Ann"


@tool
def make_dict() -> dict[str, int]:
    """Return a dict."""
    return {"a": 1}


def test_non_string_output_is_serialized_to_json():
    ex = Executor({"make_dict": make_dict})
    result = ex.run(call("make_dict"))
    assert result.tool_result == '{"a": 1}'


def test_optional_argument_uses_default_when_omitted():
    ex = Executor({"add": add})
    assert ex.run(call("add", {"a": 2})).tool_result == "3"


def test_argument_types_are_coerced():
    ex = Executor({"unique": unique})
    result = ex.run(call("unique", {"items": [1, 2, 2]}))
    assert result.tool_result == '"{1, 2}"'


def test_unknown_tool_raises():
    ex = Executor({"greet": greet})
    with pytest.raises(ToolNotFoundError, match="not found"):
        ex.run(call("nope"))


def test_unknown_argument_raises():
    ex = Executor({"greet": greet})
    with pytest.raises(InvalidArgumentsError, match="unknown argument"):
        ex.run(call("greet", {"name": "Ann", "extra": 1}))


def test_missing_required_argument_raises():
    ex = Executor({"greet": greet})
    with pytest.raises(InvalidArgumentsError, match="missing required"):
        ex.run(call("greet", {}))


def test_wrong_argument_type_raises():
    ex = Executor({"add": add})
    with pytest.raises(InvalidArgumentsError, match="a valid integer"):
        ex.run(call("add", {"a": "x"}))


def test_arguments_must_be_a_json_object():
    ex = Executor({"greet": greet})
    with pytest.raises(InvalidArgumentsError, match="JSON object"):
        ex.run(ToolCall(tool_name="greet", tool_id="c1", tool_args={1, 2}))


def test_tool_error_is_reraised_unchanged():
    ex = Executor({"boom": boom})
    with pytest.raises(ToolError, match="boom"):
        ex.run(call("boom"))


def test_other_exceptions_are_wrapped():
    ex = Executor({"crash": crash})
    with pytest.raises(ToolExecutionError, match="ValueError: crash"):
        ex.run(call("crash"))


def test_slow_handler_times_out():
    ex = Executor({"slow": slow}, timeout=0.1)
    with pytest.raises(ToolExecutionError, match="timed out"):
        ex.run(call("slow"))