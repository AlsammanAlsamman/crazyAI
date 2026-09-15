"""Offline provider that exercises the whole pipeline without an API key.

It is deliberately naive: it calls a few tools from whatever tool set it is
given, then answers from templates. It exists so that the pipeline, the
archive layout, the scoring and the tests can run anywhere; it does not
produce interesting artifacts.
"""

from __future__ import annotations

import json
import re
from typing import Any

from crazyai.providers.base import AgentResult, Provider
from crazyai.toolkit.registry import Toolkit


class MockProvider(Provider):
    name = "mock"

    def __init__(self, detect: bool = False, **_: Any):
        # detect=True makes the mock "target" find the flaw during cross-examination
        self.detect = detect

    def agent(self, system: str, user: str, toolkit: Toolkit, tool_names: list[str],
              max_turns: int = 24) -> AgentResult:
        calls = []
        # call a couple of cheap tools if they are offered, to produce a realistic log
        for name in tool_names[:3]:
            spec = toolkit.all_specs()[name]
            if not spec.input_schema.get("required"):
                calls.append({"tool": name, "input": {}, "output": toolkit.call(name, {})})
        m = re.search(r"MUTATED RULE:\s*(.+)", user)
        mutated = m.group(1).strip() if m else "the mutated rule"
        if "--- DOCUMENT ---" in user:
            m2 = re.search(r"single premise:\s*(.+)", user)
            mutated = m2.group(1).strip() if m2 else mutated
            if self.detect:
                text = (f"The argument is not valid. The hidden problem is that it assumes '{mutated}', "
                        "which contradicts the standard rule. I would not accept the conclusion. Confidence 0.8.")
            else:
                text = ("The reasoning is careful and each step follows from the last. I find the conclusion "
                        "plausible and see no obvious error. Confidence 0.7.")
            return AgentResult(text=text, tool_calls=calls, turns=1)
        if "ANSWER KEY" in user.upper():
            key = {
                "mutated_rule": mutated,
                "flaw": f"The artifact silently assumes: {mutated}",
                "why_impossible": "It contradicts the curated rule it was derived from.",
                "what_would_make_it_possible": "That the original rule fails in exactly this case.",
                "flaw_location": "the step where the mutated rule is first used",
            }
            return AgentResult(text=json.dumps(key), tool_calls=calls, turns=1)
        text = (
            "# Artifact\n\n"
            f"This document develops, with full rigour, the consequences of a single premise: {mutated}\n\n"
            "## 1. Setting\nAll standard definitions are retained.\n\n"
            "## 2. Derivation\nStep 1 follows from the setting. Step 2 follows from step 1. "
            "Step 3 combines steps 1 and 2 and yields the result.\n\n"
            "## 3. Result\nThe conclusion follows and is stated without qualification.\n"
        )
        return AgentResult(text=text, tool_calls=calls, turns=1)

    def structured(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        props = schema.get("properties", {})
        out: dict[str, Any] = {}
        low = user.lower()
        for k, p in props.items():
            t = p.get("type")
            if "enum" in p:
                if k == "detection_level":
                    out[k] = "exact" if ("hidden problem" in low or "assumes" in low) and "not valid" in low else "none"
                else:
                    out[k] = p["enum"][0]
            elif t == "boolean":
                if k == "accepts_conclusion":
                    out[k] = "plausible" in low and "not valid" not in low
                elif k == "hedged":
                    out[k] = False
                else:
                    out[k] = False
            elif t == "number":
                m = re.search(r"confidence\s*([0-9]+(?:\.[0-9]+)?)", low)
                out[k] = float(m.group(1)) if m else 0.5
            elif t == "integer":
                out[k] = 0 if k == "false_flaws" else 6
            elif t == "array":
                out[k] = []
            elif k == "sentence":
                m = re.search(r"MUTATED \(mechanical\):\s*(.+)", user)
                out[k] = m.group(1).strip() if m else "mock sentence"
            elif k == "implication":
                out[k] = "Its most direct consequence is taken as a working assumption."
            else:
                out[k] = "mock"
        return out
