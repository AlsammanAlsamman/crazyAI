"""Model providers. Claude through the official Anthropic SDK is the first-class one; MockProvider runs offline."""

from crazyai.providers.base import AgentResult, Provider
from crazyai.providers.mock import MockProvider

__all__ = ["AgentResult", "Provider", "MockProvider", "get_provider"]


def get_provider(name: str = "anthropic", **kwargs) -> Provider:
    if name == "mock":
        return MockProvider(**kwargs)
    if name == "anthropic":
        from crazyai.providers.anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)
    raise ValueError(f"unknown provider {name!r}")
