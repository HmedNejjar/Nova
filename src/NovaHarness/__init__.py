"""Nova-Harness: a small agent harness for any LiteLLM model.

    from NovaHarness.harness import Agent
    from NovaHarness.tools import tool
    from NovaHarness.models import LiteLLMClient
"""

from importlib.metadata import PackageNotFoundError, version

from . import harness, models, tools

try:
    __version__ = version("nova-harness")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = ["harness", "models", "tools", "__version__"]