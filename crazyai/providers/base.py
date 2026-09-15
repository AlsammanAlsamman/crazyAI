"""Provider interface: an agent loop over the toolkit, and a structured (JSON-schema) call."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from crazyai.toolkit.registry import Toolkit


@dataclass
class AgentResult:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""


class Provider(ABC):
    name: str = "base"

    @abstractmethod
    def agent(self, system: str, user: str, toolkit: Toolkit, tool_names: list[str],
              max_turns: int = 24) -> AgentResult:
        """Run a session: the model may call any of `tool_names` through `toolkit` until it answers in text."""

    @abstractmethod
    def structured(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """One call that must return JSON matching `schema`."""
