from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]
Status = Literal["ok", "error", "denied"]

@dataclass(frozen=True)
class ToolSpec:
    """What the model is told about a tool. No handler."""
    name: str
    description: str
    args: dict[str, Any]

@dataclass(frozen=True)
class ToolCall:
    """A request from the model to run one tool.

        `tool_id` is echoed back on the matching ToolResult. It comes from the
        provider; the model client generates one if the provider sends none.
        """
    
    tool_name: str
    tool_id: str
    tool_args: dict[str, Any]
    
@dataclass(frozen=True)
class ToolResult:
    """The outcome of one ToolCall, produced by the executor or the policy layer.
 
    `content` is always text, because that is all the model can read.
    The executor converts non-string tool output into a string.
    """

    
    tool_name: str
    tool_id: str
    status: Status
    tool_result: str
    
@dataclass(frozen=True) 
class ModelResponse:
    """One normalized model turn: text, tool calls, or both."""
    
    response: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    
@dataclass(frozen=True)
class Message:
    """
    One entry in the conversation history.
    
        -user:      content\n
        -assistant: content and/or tool_calls\n
        -tool:      content + tool_call_id (points back to ToolCall.tool_id)
    """
    
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    
    def __post_init__(self):
        if self.role == "user":
            if self.content is None or self.tool_calls or self.tool_call_id is not None:
                raise ValueError("User messages must have content only.")
        
        elif self.role == "assistant":
            if self.content is None and not self.tool_calls:
                raise ValueError("assistant message needs content or tool_calls")
            if self.tool_call_id is not None:
                raise ValueError("assistant message cannot have tool_call_id")
        else:  # tool
            if self.content is None or self.tool_call_id is None:
                raise ValueError("tool message needs content and tool_call_id")
            if self.tool_calls:
                raise ValueError("tool message cannot have tool_calls")
        
    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)
    
    @classmethod
    def assistant(cls, response: ModelResponse) -> "Message":
        return cls(role="assistant", content=response.response, tool_calls=response.tool_calls)
    
    @classmethod
    def tool(cls, result: ToolResult) -> "Message":
        return cls(role="tool", content= result.tool_result, tool_call_id=result.tool_id)
    
    
