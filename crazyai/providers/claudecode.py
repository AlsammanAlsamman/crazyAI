"""Claude through the Claude Code CLI (`claude -p`).

No API key: the CLI uses whatever Claude Code is logged in with, so a claude.ai
subscription pays for the run. Each call is one headless, stateless
session: the crazyAI system prompt replaces Claude Code's own, and no tools are
given (the pipeline itself measures artifacts). Use it for `crazyai invent`;
the impossible-pipeline still prefers the SDK provider because its generators
lean on toolkit calls.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any

from crazyai.providers.base import AgentResult, Provider
from crazyai.toolkit.registry import Toolkit


class ClaudeCodeProvider(Provider):
    name = "claudecode"

    def __init__(self, model: str = "", effort: str = "", timeout: int = 900, **_: Any):
        if shutil.which("claude") is None:
            raise RuntimeError("the `claude` CLI is not installed or not on PATH")
        self.model = model
        self.effort = effort
        self.timeout = timeout

    def _run(self, system: str, user: str) -> str:
        cmd = ["claude", "-p", "--no-session-persistence", "--output-format", "text",
               "--tools", "", "--system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        res = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=self.timeout)
        if res.returncode != 0:
            raise RuntimeError(f"claude -p failed ({res.returncode}): {res.stderr.strip()[-2000:]}")
        return res.stdout.strip()

    def agent(self, system: str, user: str, toolkit: Toolkit, tool_names: list[str],
              max_turns: int = 24) -> AgentResult:
        if tool_names:
            user += ("\n\n(No tools are available in this session. Do the reasoning yourself, state your prediction, "
                     "and give the complete artifact; the pipeline will compile and measure it.)")
        text = self._run(system, user)
        return AgentResult(text=text, tool_calls=[], turns=1, model=self.model or "claude-code-default")

    def structured(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        prompt = user + "\n\nAnswer with a single JSON object matching this schema and nothing else:\n" + json.dumps(schema)
        text = self._run(system, prompt)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}
