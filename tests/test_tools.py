import pytest
from NovaHarness.harness.errors import ToolDefinitionError
from NovaHarness.tools.tool import Tool, tool

def test_required_vs_default():
    @tool
    def get_weather(city: str, days: int = 3) -> str:
        """Get the weather forecast for a city."""
        return city

    assert get_weather.tool_spec.args["required"] == ["city"]

def test_missing_docstring_raises():
    with pytest.raises(ToolDefinitionError):
        @tool
        def f(x: int) -> int:
            return x