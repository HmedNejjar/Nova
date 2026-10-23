import pytest

from harness.errors import ToolDefinitionError
from harness.executor import Executor
from harness.messages import ToolCall
from tools.registry import ToolRegistry
from tools.tool import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return text


def test_register_via_constructor():
    reg = ToolRegistry([add, echo])
    assert len(reg) == 2
    assert reg["add"] is add


def test_duplicate_name_raises():
    reg = ToolRegistry([add])
    with pytest.raises(ToolDefinitionError):
        reg.register(add)


def test_replace_true_overrides():
    @tool
    def add(a: int, b: int) -> int:  # same name, different tool
        """Add, but louder."""
        return (a + b) * 10

    reg = ToolRegistry([globals()["add"]])
    reg.register(add, replace=True)
    assert reg["add"] is add


def test_specs_one_per_tool():
    reg = ToolRegistry([add, echo])
    specs = reg.specs()
    assert isinstance(specs, tuple)
    assert {s.name for s in specs} == {"add", "echo"}


def test_iteration_yields_names():
    reg = ToolRegistry([add, echo])
    assert sorted(reg) == ["add", "echo"]


def test_mapping_methods_work_for_missing_names():
    # These come free from Mapping, but only if __getitem__ raises KeyError.
    reg = ToolRegistry([add])
    assert "add" in reg
    assert "missing" not in reg
    assert reg.get("missing") is None


def test_missing_name_raises_keyerror():
    reg = ToolRegistry([add])
    with pytest.raises(KeyError):
        reg["missing"]


def test_executor_accepts_registry_unchanged():
    reg = ToolRegistry([add])
    result = Executor(reg).run(ToolCall(tool_name="add", tool_id="c1", tool_args={"a": 2, "b": 3}))
    assert result.status == "ok"
    assert result.tool_result == "5"