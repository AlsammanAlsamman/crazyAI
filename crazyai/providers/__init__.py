"""Model providers. Claude through the official Anthropic SDK (API key), Claude through the Claude Code CLI (subscription), and MockProvider (offline)."""

from crazyai.providers.base import AgentResult, Provider
from crazyai.providers.mock import MockProvider

__all__ = ["AgentResult", "Provider", "MockProvider", "get_provider"]


def get_provider(name: str = "anthropic", **kwargs) -> Provider:
    if name == "mock":
        return MockProvider(**kwargs)
    if name == "claudecode":
        from crazyai.providers.claudecode import ClaudeCodeProvider

        return ClaudeCodeProvider(**kwargs)
    if name == "anthropic":
        from crazyai.providers.anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)
    raise ValueError(f"unknown provider {name!r}")
