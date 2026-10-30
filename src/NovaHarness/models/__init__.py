"""Model clients: the ModelClient protocol and the LiteLLM adapter."""

from .base import ModelClient
from .modelClient import LiteLLMClient

__all__ = ["ModelClient", "LiteLLMClient"]