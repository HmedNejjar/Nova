from typing import ClassVar

class HarnessError(Exception):
    """Base class for all harness errors."""

    default_message: ClassVar[str] = "A harness error occurred."

    def __init__(self, message: str | None = None):
        super().__init__(message or self.default_message)
        

# --- Recoverable errors: shown to the model as a tool result ---
class ToolError(HarnessError):
    """Base class for all tool errors."""

    default_message = "A tool error or tool call fail has occurred."

class ToolNotFoundError(ToolError):
    """Raised when a tool is not found."""

    default_message = "The requested tool was not found."

class InvalidArgumentsError(ToolError):
    """Raised when a tool is called with invalid arguments."""

    default_message = "The tool was called with invalid arguments."
    
class ToolExecutionError(ToolError):
    """Raised when a tool execution fails."""

    default_message = "The tool execution failed."

# --- Fatal errors: stops the run ---

class StopRunError(HarnessError):
    """Raised when the run should be stopped."""

    default_message = "The run should be stopped. An error occurred."

class PolicyAbortError(StopRunError):
    """Raised when the policy layer aborts the current run."""

    default_message = "The policy layer aborted the current run."

class MaxStepsReachedError(StopRunError):
    """Raised when the maximum number of steps is reached."""

    default_message = "The maximum number of steps was reached."
    
class ModelError(StopRunError):
    """Raised when the model call fails (rate limit, timeout, auth, ...)."""
    
    default_message = "The model call failed."