"""Claude via the official Anthropic SDK.

- claude-opus-5 by default, adaptive thinking, streaming (so long artifacts
  never hit HTTP timeouts).
- Server-side refusal fallbacks are enabled by default
  (betas=["server-side-fallback-2026-07-01"], fallbacks="default"); pass
  fallbacks=False to turn them off.
- A manual agent loop rather than the beta tool runner, because every tool
  call has to be logged into the run folder and the tool set changes per step.
"""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from crazyai.config import DEFAULT_EFFORT, DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from crazyai.providers.base import AgentResult, Provider
from crazyai.toolkit.registry import Toolkit

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT,
                 max_tokens: int = DEFAULT_MAX_TOKENS, fallbacks: bool = True,
                 client: anthropic.Anthropic | None = None):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self.client = client or anthropic.Anthropic()

    # -- low level -----------------------------------------------------------------
    def _request(self, **params: Any) -> Any:
        base = dict(model=self.model, max_tokens=self.max_tokens,
                    thinking={"type": "adaptive"}, output_config={"effort": self.effort}, **params)
        if self.fallbacks:
            try:
                with self.client.beta.messages.stream(betas=[_FALLBACK_BETA], fallbacks="default", **base) as s:
                    return s.get_final_message()
            except anthropic.BadRequestError as exc:
                # the fallback beta is not accepted everywhere (proxies, older gateways): degrade once
                if "fallback" in str(exc).lower() or "beta" in str(exc).lower():
                    self.fallbacks = False
                else:
                    raise
        with self.client.messages.stream(**base) as s:
            return s.get_final_message()

    @staticmethod
    def _text(message: Any) -> str:
        return "".join(b.text for b in message.content if getattr(b, "type", "") == "text")

    @staticmethod
    def _usage(message: Any) -> dict[str, int]:
        u = getattr(message, "usage", None)
        if not u:
            return {}
        return {k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "output_tokens",
                                                        "cache_read_input_tokens", "cache_creation_input_tokens")}

    # -- interface -----------------------------------------------------------------
    def agent(self, system: str, user: str, toolkit: Toolkit, tool_names: list[str],
              max_turns: int = 24) -> AgentResult:
        tools = toolkit.to_api_tools(tool_names)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        calls: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        turns = 0
        message = None
        while turns < max_turns:
            turns += 1
            kwargs: dict[str, Any] = dict(system=[{"type": "text", "text": system,
                                                   "cache_control": {"type": "ephemeral"}}], messages=messages)
            if tools:
                kwargs["tools"] = tools
            message = self._request(**kwargs)
            for k, v in self._usage(message).items():
                usage[k] = usage.get(k, 0) + v
            if message.stop_reason == "refusal":
                return AgentResult(text=self._text(message), tool_calls=calls, turns=turns,
                                   stop_reason="refusal", usage=usage, model=message.model)
            if message.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": message.content})
                continue
            tool_uses = [b for b in message.content if getattr(b, "type", "") == "tool_use"]
            if message.stop_reason != "tool_use" or not tool_uses:
                break
            messages.append({"role": "assistant", "content": message.content})
            results = []
            for tu in tool_uses:
                args = tu.input if isinstance(tu.input, dict) else json.loads(tu.input or "{}")
                out = toolkit.call(tu.name, args)
                calls.append({"tool": tu.name, "input": args, "output": out, "t": time.time()})
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": json.dumps(out, ensure_ascii=False, default=str),
                                "is_error": "error" in out})
            messages.append({"role": "user", "content": results})
        assert message is not None
        return AgentResult(text=self._text(message), tool_calls=calls, turns=turns,
                           stop_reason=str(message.stop_reason), usage=usage, model=message.model)

    def structured(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        message = self._request(
            system=system, messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if message.stop_reason == "refusal":
            return {"error": "refusal"}
        text = self._text(message)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": "non-JSON response", "raw": text}
