from dataclasses import dataclass
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints
from NovaHarness.harness.messages import ToolSpec
from NovaHarness.harness.errors import ToolDefinitionError

import inspect


AUTHORIZED_DTYPES: dict[Any, str] = {int: "integer",
                                    float: "number",
                                    str: "string",
                                    bool: "boolean",
                                    list: "array",
                                    tuple: "array",
                                    set: "array",
                                    dict: "object",
                                    type(None): "null"}

def _json_schema(hint: Any, where: str) -> dict[str, Any]:
    """Recursively turn a type hint into a JSON Schema fragment."""
    origin, args = get_origin(hint), get_args(hint)
 
    if hint in AUTHORIZED_DTYPES:  # bare: str, int, list, dict, None
        return {"type": AUTHORIZED_DTYPES[hint]}
 
    if origin in (list, set) and len(args) == 1:  # list[str], set[int]
        return {"type": "array", "items": _json_schema(args[0], where)}
 
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:  # tuple[int, ...]
        return {"type": "array", "items": _json_schema(args[0], where)}
 
    if origin is dict and len(args) == 2 and args[0] is str:  # dict[str, int]
        return {"type": "object", "additionalProperties": _json_schema(args[1], where)}
 
    if origin in (Union, types.UnionType):  # str | None, Optional[str]
        return {"anyOf": [_json_schema(a, where) for a in args]}
 
    raise ToolDefinitionError(f"{where} has unsupported type {hint!r}")


def _desc_extraction(fn: Callable) -> str:
        doc = inspect.getdoc(fn)
        
        if not doc or not doc.strip():
            raise ToolDefinitionError(f"{fn.__name__} description is empty. Required for tool schema.")

        return doc.strip()
    
def _build_schema(fn: Callable) -> dict[str, Any]:
    """Build the schema for the tool.
    Args:
        fn: The tool function to build the schema for.
    Returns:
        dict: The schema for the tool containing the function parameters, description and required fields.
    """
    fn_name: str = fn.__name__
    
    params = inspect.signature(fn).parameters
    hints = get_type_hints(fn)
    
    
    properties = {}
    required = []
    
    for name, param in params.items():
            
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ToolDefinitionError(
                f"{fn_name}: *args/**kwargs are not supported ('{name}')"
            )
        if name not in hints:
            raise ToolDefinitionError(
                f"{fn_name}: parameter '{name}' has no type annotation"
            )
        properties[name] = _json_schema(hints[name], f"{fn_name}: parameter '{name}'")
            
        if param.default is inspect.Parameter.empty:
            required.append(name)
            
    return {"type": "object", "properties": properties, "required": required}


@dataclass(frozen=True)
class Tool:
    """A tool: what the model sees (spec) plus what the executor runs (handler)."""
    tool_spec: ToolSpec
    handler: Callable[..., Any]
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool handler with the given arguments."""
        return self.handler(*args, **kwargs)
    
def tool(fn: Callable) -> Tool:
    """tool decorator\n
        Creates a tool from a function.
        Args:
            fn: The function to turn into a tool.
        Returns:
            Tool: The tool object.
    """
    fn_name = fn.__name__
    tool_desc = _desc_extraction(fn)
    schema = _build_schema(fn)
    tool_spec = ToolSpec(name=fn_name, description=tool_desc, args=schema)
    return Tool(tool_spec, fn)