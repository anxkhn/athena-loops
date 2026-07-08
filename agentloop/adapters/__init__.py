from .cli import CliAgent
from .mock import MockAgent

__all__ = ["MockAgent", "CliAgent", "ClaudeAgent", "GrokAgent"]

# API adapters are imported lazily so the package works without optional SDKs.
def __getattr__(name):
    if name == "ClaudeAgent":
        from .claude import ClaudeAgent
        return ClaudeAgent
    if name == "GrokAgent":
        from .grok import GrokAgent
        return GrokAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
