"""`crazyai invent-continuous`: the narrative bends the AI, not the AI unbends the story.

Same world/blend/corpus machinery as `Invent`, but immerse and bend collapse into a single
continuous call - no persona swap from "native" to "rigorous engineer," no MAPPING table, no
naming which assumption is broken. The hypothesis under test: `Invent`'s two-call, decompose-then-
translate structure is itself what drains the narrative's influence on implementation judgment
(see crazyai-trials/AIM.md for the concrete evidence motivating this - a losing narrative kernel
that found the right algorithmic idea but implemented it worse than a direct-prompt baseline).

    archive/invent_<seed>_<target>/
        seed.json, world.json, world.md   same as Invent
        artifact.md, artifact.c            one continuous call, no separate ideas.md
        measure.json, run.json             same as Invent

Archives into the SAME archive/invent_<seed>_<target>/ layout and invent_index.jsonl as Invent -
this is still the narrative condition, just a different mechanism for it, so it belongs in the
same index for ranking/history purposes (use fresh seed numbers to avoid colliding with a
two-call Invent run of the same seed+target).
"""

from __future__ import annotations

from typing import Any

from crazyai.pipeline import invent_prompts as P
from crazyai.pipeline.invent import Invent, _CODE, _PRED, _dump
from crazyai.providers.base import Provider


class Continuous(Invent):
    def step_continuous(self, provider: Provider, world: dict[str, Any], seed: dict[str, Any]) -> dict[str, Any]:
        if self._have("artifact.md"):
            art = (self.dir / "artifact.md").read_text(encoding="utf-8")
        else:
            names = self.toolkit.names(families=self.tgt.measure_families + ["unconventional", "symbolic"])
            hint = self.tgt.assumption_hints.get(seed["assumption_focus"], "") if self._pinned_assumption else ""
            prompt = P.continuous_prompt(world["text"], self.tgt, seed["depth"], names, hint)
            res = provider.agent(P.CONTINUOUS_SYSTEM, prompt, self.toolkit, names, self.max_tool_turns)
            art = res.text
            (self.dir / "artifact.md").write_text(art, encoding="utf-8")
            _dump(self.dir / "bend_calls.json", {"turns": res.turns, "stop_reason": res.stop_reason, "usage": res.usage, "calls": res.tool_calls})
        code = _CODE.findall(art)
        pred = _PRED.search(art)
        out = {"text": art, "code": code[-1] if code else None, "prediction": float(pred.group(1)) if pred else None,
               "seeds": []}
        if out["code"] and self.tgt.artifact == "kernel":
            (self.dir / "artifact.c").write_text(out["code"], encoding="utf-8")
        self._say(f"continuous: {len(art)} chars, code={'yes' if out['code'] else 'no'}, prediction={out['prediction']}")
        return out

    def execute(self, provider: Provider) -> dict[str, Any]:
        seed = self._timed("seed", self.step_seed)
        self._timed("harvest", self.step_harvest, provider)
        world = self._timed("world", self.step_world, seed)
        bent = self._timed("continuous", self.step_continuous, provider, world, seed)
        measure = self._timed("measure", self.step_measure, bent, world)
        return self._timed("archive", self.step_archive, provider, seed, world, measure)
