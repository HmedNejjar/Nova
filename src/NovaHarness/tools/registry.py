from typing import Iterable, Iterator, Mapping

from NovaHarness.harness.errors import ToolDefinitionError, ToolNotFoundError
from NovaHarness.harness.messages import ToolSpec
from .tool import Tool

class ToolRegistry(Mapping[str, Tool]):
    """A registry of Name -> Tool."""
    
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        
        for tool in tools:
            self.register(tool)
            
    def register(self, tool: Tool, replace: bool = False) -> None:
        """Add a tool. A duplicate name error raises, unless `replace=True` says it's intended."""
        
        name = tool.tool_spec.name
        
        if name in self._tools and not replace:
            raise ToolDefinitionError(f"tool {name} already registered")
        self._tools[name] = tool
        
    def specs(self) -> tuple[ToolSpec, ...]:
        """Return the specs of all tools in the registry."""
        return tuple(tool.tool_spec for tool in self._tools.values())
    
    
    def __getitem__(self, name: str) -> Tool:
        """Look up the tool by name."""
        return self._tools[name]
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over the names of the tools in the registry."""
        return iter(self._tools.keys())
    
    def __len__(self) -> int:
        """Return the number of tools in the registry."""
        return len(self._tools)