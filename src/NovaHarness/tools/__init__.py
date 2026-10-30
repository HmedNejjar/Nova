"""Tools: the @tool decorator, the registry, the built-in tools and their default policy."""

from .tool import Tool, tool
from .registry import ToolRegistry
from .default_tools import (
    SearchFn,
    brave_search,
    ddgs_search,
    default_policy,
    default_tools,
    make_filesystem_tools,
    make_shell_tools,
    make_web_tools,
)

__all__ = [
    "tool", "Tool", "ToolRegistry",
    "default_tools", "default_policy",
    "make_filesystem_tools", "make_shell_tools", "make_web_tools",
    "SearchFn", "ddgs_search", "brave_search",
]