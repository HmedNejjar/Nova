"""Test doubles for the harness. Lives in tests/."""

from collections.abc import Iterable

from harness.messages import Message, ModelResponse, ToolCall, ToolSpec


class FakeModelClient:
    """A scripted ModelClient: returns pre-written responses, in order.

    Each script item is either a ModelResponse (returned) or an Exception
    (raised), so tests can also simulate failures such as ModelError.
    `calls` records every (messages, tools) pair the runtime sent. The messages
    are the immutable snapshot the runtime passed, so later history growth
    can't change what a test sees here.
    """

    def __init__(self, script: Iterable[ModelResponse | Exception]):
        self._script = iter(script)
        self.calls: list[tuple[tuple[Message, ...], tuple[ToolSpec, ...]]] = []

    def generate(self, messages: tuple[Message, ...], tools: tuple[ToolSpec, ...]) -> ModelResponse:
        self.calls.append((messages, tools))
        try:
            item = next(self._script)
        except StopIteration:
            raise AssertionError(
                f"FakeModelClient script exhausted: the runtime made call "
                f"#{len(self.calls)} but the test scripted fewer turns."
            ) from None
        if isinstance(item, Exception):
            raise item
        return item


class FakeApprover:
    """Answers every ask the same way and records what it was asked."""

    def __init__(self, answer: bool = True):
        self.answer = answer
        self.asked: list[tuple[ToolCall, str | None]] = []

    def approve(self, call: ToolCall, reason: str | None = None) -> bool:
        self.asked.append((call, reason))
        return self.answer