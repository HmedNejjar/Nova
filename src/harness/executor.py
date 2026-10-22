from dataclasses import dataclass
import inspect
import json
import time
from threading import Thread
from typing import Any, Callable, Mapping, get_type_hints
from pydantic import TypeAdapter, ValidationError
from .messages import ToolCall, ToolResult
from .policies import Decision
from .errors import InvalidArgumentsError, ToolExecutionError, ToolNotFoundError, ToolError
from tools.tool import Tool


@dataclass(frozen=True)
class _ArgSpec:
    """Per-tool validation data"""
    
    # Adapters for each argument.
    # Keys are argument names, values are TypeAdapter instances.
    # Values are cached, so this is safe to call multiple times.
    adapters: dict[str, TypeAdapter[Any]]

    required: frozenset[str]
    

class Executor:
    """Looks up, validates, runs, and stringifies one ToolCall."""
    
    def __init__(self, tools: Mapping[str, Tool], timeout: float = 30.0, max_output_len: int = 10_000) -> None:
        self._tools = tools
        self.timeout = timeout
        self.max_output_len = max_output_len
        self._arg_specs: dict[str, _ArgSpec] = {}
        
    
    def run(self, tool_call: ToolCall) -> ToolResult:
        """Execute the tool call."""
        tool = self._lookup(tool_call.tool_name)
        args = self._validate_args(tool, tool_call)
        output = self._to_text(self._call_handler(tool, tool_call, args))
        
        return ToolResult(tool_name= tool_call.tool_name, tool_id= tool_call.tool_id, status= "ok", tool_result= output)
        
    def _lookup(self, tool_name: str) -> Tool:
        """Look up the tool by name."""
        if tool_name not in sorted(self._tools.keys()):
            raise ToolNotFoundError(f"tool {tool_name} not found")
        return self._tools[tool_name]
    
    def _validate_args(self, tool: Tool, tool_call: ToolCall) -> dict[str, Any]:
        """Validate the arguments for the tool call.
        Check and coerce the model's args against the handler's type hints.
 
        Catches unknown args, missing required args, and wrong types, with
        messages specific enough for the model to fix its next call. Coerces
        JSON shapes into the annotated types, e.g. a list into set[int].
        """
        name = tool_call.tool_name
        spec = self._arg_spec(tool)
        given = tool_call.tool_args

        # Check if the arguments are a JSON object.
        if not isinstance(given, dict):
            raise InvalidArgumentsError(f"{name}: arguments must be a JSON object")

        # Check for unknown arguments.
        unknown = sorted(given.keys() - spec.adapters.keys())
        if unknown:
            raise InvalidArgumentsError(
                f"{name}: unknown argument(s) {unknown}; expected {sorted(spec.adapters)}"
            )
 
        # Check for missing required arguments.
        missing = sorted(spec.required - given.keys())
        if missing:
            raise InvalidArgumentsError(f"{name}: missing required argument(s) {missing}")
 
        # Check for wrong types.
        clean: dict[str, Any] = {}
        for arg, value in given.items():
            try:
                clean[arg] = spec.adapters[arg].validate_python(value)
            except ValidationError as e:
                detail = e.errors()[0]["msg"]  # e.g. "Input should be a valid integer"
                raise InvalidArgumentsError(f"{name}: argument '{arg}': {detail}") from e
        return clean
    
    def _arg_spec(self, tool: Tool) -> _ArgSpec:
        """Build the per-tool validators once, then reuse them on every call."""
        name = tool.tool_spec.name
        cached = self._arg_specs.get(name)
        if cached is None:
            params = inspect.signature(tool.handler).parameters
            hints = get_type_hints(tool.handler)  # @tool guarantees every param has one
            cached = _ArgSpec(
                adapters={p: TypeAdapter(hints[p]) for p in params},
                required=frozenset(
                    p for p, param in params.items() if param.default is inspect.Parameter.empty
                ),
            )
            self._arg_specs[name] = cached
        return cached
    
    def _call_handler(self, tool: Tool, tool_call: ToolCall, args: dict[str, Any]) -> Any:
        """Call the tool handler with the given arguments with a timeout.
        
        - handler raises a ToolError  -> re-raised as is (its message is already specific)
        - handler raises anything else -> ToolExecutionError, original kept in __cause__
        - exceeds self.timeout         -> ToolExecutionError
        """
        name = tool.tool_spec.name
        result: list[Any] = []
        error: list[Exception] = []
 
        def target() -> None:
            try:
                result.append(tool.handler(**args))
            except Exception as e:  # not BaseException: Ctrl+C must still stop the run
                error.append(e)
 
        worker = Thread(target=target, daemon=True, name=f"tool-{name}")
        worker.start()
 
        # Short joins, not one long one: keeps Ctrl+C responsive.
        deadline = time.monotonic() + self.timeout
        while worker.is_alive() and (remaining := deadline - time.monotonic()) > 0:
            worker.join(min(0.5, remaining))
 
        if worker.is_alive():
            raise ToolExecutionError(f"{name}: timed out after {self.timeout:g}s")
        if error:
            e = error[0]
            if isinstance(e, ToolError):
                raise e
            raise ToolExecutionError(f"{name}: {type(e).__name__}: {e}") from e
        return result[0]
        
    def _to_text(self, output: Any) -> str:
        """Convert the output to a string."""
        
        # Handle None case first.
        if output is None:
            return "(no output)"
        
        # Handle string case.
        elif isinstance(output, str):
            text = output
        
        # Handle other types.
        else:
            try:
                # Serialize as JSON for structured data.
                text = json.dumps(output, default=str, ensure_ascii= False)
            except (TypeError, ValueError):
                text = repr(output)
        
        # Truncate if necessary.
        if len(text) > self.max_output_len:
            text = (text[: self.max_output_len] + f"\n... [truncated: {len(text):,} chars total, showing first {self.max_output_len:,}]")
            
        return text
        
            
        