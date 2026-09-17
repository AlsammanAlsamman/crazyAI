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
        invent = self._invent(system, user)
        if invent is not None:
            return AgentResult(text=invent, tool_calls=calls, turns=1)
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

    # ---- `crazyai invent` steps (harvest / immerse / bend), recognised by their system prompts
    def _invent(self, system: str, user: str) -> str | None:
        if "librarian of the human imagination" in system:
            return json.dumps([
                {"kind": "book", "source": "mock: The Glass Orchard", "text": "An orchard where every fruit is a small window onto another season; pick one and it is winter in your hand while summer stays on the branch. The gardeners prune by closing windows they do not like."},
                {"kind": "painting", "source": "mock: The Weighing of the Clouds", "text": "Clerks in grey aprons weigh clouds on brass scales in a hall with no ceiling; the heavier clouds are ledgered and sent to sea. One clerk has fallen asleep and a cloud is escaping through the floor."},
                {"kind": "metaphor", "source": "mock: a debt is a shadow", "text": "A debt is a shadow: it is longest in the morning and evening, shortest at noon, it follows you into every room, and it is only gone when the light is gone too."},
            ])
        if "You are not an assistant and you are not on Earth" in system:
            return (
                "Here we do not add the numbers; we let them flow. I take the second table and lay it flat along the inside of "
                "the long pipe, one strip of rows for each segment of the pipe, so the wall itself is written on. Then I pour the "
                "first table in, one row at a time, like a drop of the raspberry river. As the drop passes a segment it rubs "
                "against the strip written there and carries away the products, a little heavier each time; it does not stop to "
                "finish any one cell, it finishes them all at once as it goes. Several drops can be poured together, each keeping "
                "to its own groove. When the drop comes out of the far end it is the finished row of the third table, and the "
                "wall has not moved once. What is thrown away is the habit of finishing one cell before the next.\n"
                "SEED: The second table is engraved on the pipe wall in segments, so it never moves while the first table flows past.\n"
                "SEED: A drop carries all the cells of its row at once and finishes none of them until it leaves the pipe.\n"
                "SEED: Many drops flow in parallel grooves through the same engraved wall.\n"
            )
        if "just returned from a strange world" in system:
            return (
                "MAPPING\n| pipe wall | matrix B, kept in place | drop | one row of A (its partial C row) | segment | a block of 8 rows of B | groove | a thread |\n\n"
                "CHOSEN SEED: the engraved wall with drops flowing past.\n\nASSUMPTION BROKEN: the whole sum over the shared index "
                "is finished before the next cell is started - here every cell of the row advances together.\n\n"
                "ARTIFACT\n```c\n#include <string.h>\n#define W 8\nvoid kernel(int n, const double *A, const double *B, double *C) {\n"
                "    memset(C, 0, (size_t)n * n * sizeof(double));\n    int kk = 0;\n"
                "    for (; kk + W <= n; kk += W)                       /* one engraved segment of the wall */\n"
                "        for (int i = 0; i < n; i++) {                  /* one drop = one row of A flows past it */\n"
                "            const double *a = A + (size_t)i * n + kk;\n            double *c = C + (size_t)i * n;\n"
                "            for (int j = 0; j < n; j++) {\n                double s = c[j];\n"
                "                for (int w = 0; w < W; w++) s += a[w] * B[(size_t)(kk + w) * n + j];\n                c[j] = s;\n            }\n        }\n"
                "    for (; kk < n; kk++)                              /* leftover rows of the wall */\n"
                "        for (int i = 0; i < n; i++) {\n            double a = A[(size_t)i * n + kk];\n"
                "            for (int j = 0; j < n; j++) C[(size_t)i * n + j] += a * B[(size_t)kk * n + j];\n        }\n}\n```\n\n"
                "PREDICTION: speedup_vs_blocked = 1.5\n\nMEASUREMENT\nSee measure.json (the pipeline measures the final kernel itself).\n\n"
                "VERDICT\nExact; the segment keeps 8 rows of B hot while a row of C streams through.\n"
            )
        return None
