from .messages import Message


class ConversationState:
    """One conversation's message history. New conversations start with an empty history."""
    
    def __init__(self, messages: list[Message] | None = None) -> None:
        self._messages = list(messages or ())
    
    @property
    def messages(self) -> tuple[Message, ...]:
        """Read-only snapshot, in the shape ModelClient.complete expects."""
        return tuple(self._messages)
    
    def add(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self._messages.append(message)
      
    def last_n(self, n: int = 1) -> list[Message] | None:
        """Return the last n exchanges in the conversation history."""
        if not self.messages:
            return None
        return list(self.messages[-n:])
    
    def __len__(self) -> int:
        return len(self._messages)

