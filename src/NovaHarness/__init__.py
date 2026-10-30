"""Nova-Harness: a small agent harness for any LiteLLM model.

    from NovaHarness import Agent

    agent = Agent("gpt-4o-mini")
    print(agent.run("Which Python files here contain a TODO?"))
"""

from importlib.metadata import PackageNotFoundError, version

from .harness.agent import Agent
from .harness.approval import CLIApprover, ToolApprover
from .harness.errors import (
    HarnessError,
    InvalidArgumentsError,
    MaxStepsReachedError,
    ModelError,
    PolicyAbortError,
    StopRunError,
    ToolDefinitionError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)
from .harness.memory import MemoryStore
from .harness.messages import Message, ModelResponse, ToolCall, ToolResult, ToolSpec
from .harness.policies import Policy
from .models.base import ModelClient
from .tools.default_tools import brave_search, ddgs_search
from .tools.tool import Tool, tool

try:
    __version__ = version("nova-harness")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "Agent",
    "CLIApprover",
    "HarnessError",
    "InvalidArgumentsError",
    "MaxStepsReachedError",
    "MemoryStore",
    "Message",
    "ModelClient",
    "ModelError",
    "ModelResponse",
    "Policy",
    "PolicyAbortError",
    "StopRunError",
    "Tool",
    "ToolApprover",
    "ToolCall",
    "ToolDefinitionError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolResult",
    "ToolSpec",
    "__version__",
    "brave_search",
    "ddgs_search",
    "tool",
]
