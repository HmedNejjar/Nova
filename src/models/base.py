from harness.messages import Message, ModelResponse, ToolSpec
from typing import Protocol

class ModelClient(Protocol):
    """Base class for model clients."""
    def __init__(self, model: str) -> None:
        ...
    def generate(self, messages: tuple[Message, ...], tools: tuple[ToolSpec, ...]) -> ModelResponse:
        """Generate a response from the model."""
        ...
        