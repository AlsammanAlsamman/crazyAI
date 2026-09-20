"""`crazyai discover`: solution-first invention, no target, no problem.

Everything else in this project starts with a chosen problem and bends the narrative toward
solving it. This inverts that: the imagination runs with no target at all - invent a mechanism,
a relation, a pattern, purely because it is interesting or true in its own world - then a
separate, cooler-headed step asks what it might be *for*, proposed by the model itself, not
chosen by us in advance.

    archive/discover_<seed>/
        seed.json, world.json, world.md   same as Invent
        creation.md    the free invention, no target vocabulary anywhere
        formal.md      the same invention's structure, stated generally, still not application-bound
        proposals.md   0-3 concrete real problems the AI itself thinks this structure might solve
        run.json               summary; appended to archive/discover_index.jsonl (never invent_index.jsonl -
                                there is no value/status here, this is not a measured run)

A batch that proposes nothing usable is a legitimate, expected outcome, not a failure - this mode
is explicitly exploratory. If a proposal looks concrete and testable, it becomes a candidate for a
separate, later `crazyai invent` target, built the normal way with a real contract and reference.

Steps are resumable: an existing file is reused unless force=True.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crazyai.config import ARCHIVE_DIR, DEFAULT_MAX_TOOL_TURNS
from crazyai.pipeline import invent_prompts as P
from crazyai.pipeline.invent import _dump, _load
from crazyai.providers.base import Provider
from crazyai.toolkit.invent import blend as B
from crazyai.toolkit.registry import Toolkit, build_toolkit


@dataclass
class Discover:
    seed: int
    blend: str = ""                    # cutup|markov|graft|nest|anneal|evolve|compare|"" (seeded draw)
    archive_dir: Path = field(default_factory=lambda: Path(ARCHIVE_DIR))
    force: bool = False
    max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS
    log: Any = lambda msg: print(msg, flush=True)

    def __post_init__(self) -> None:
        self.dir = Path(self.archive_dir) / f"discover_{self.seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.toolkit: Toolkit = build_toolkit(self.seed, ctx={"run_dir": str(self.dir), "seed": self.seed})
        self._t0 = time.time()
        self._timing: dict[str, float] = {}

    def _have(self, name: str) -> bool:
        return (self.dir / name).exists() and not self.force

    def _say(self, msg: str) -> None:
        if self.log:
            self.log(f"[crazyai discover seed={self.seed}] {msg}")

    def _timed(self, name: str, fn, *a):
        t0 = time.time()
        out = fn(*a)
        self._timing[name] = round(time.time() - t0, 3)
        return out

    # -- 1. seed -------------------------------------------------------------------------
    def step_seed(self) -> dict[str, Any]:
        if self._have("seed.json"):
            return _load(self.dir / "seed.json")
        rng = self.toolkit.rng
        model = self.blend or rng.choice("discover.blend_model", B.MODELS + ["compare"])
        depth = rng.integer("discover.depth", 1, 3)
        s = {"seed": self.seed, "blend_model": model, "depth": depth}
        _dump(self.dir / "seed.json", s)
        self._say(f"seed: blend={model} depth={depth}")
        return s

    # -- 2. world (same blend machinery as Invent.step_world; no target coupling either way) --
    def step_world(self, seed: dict[str, Any]) -> dict[str, Any]:
        if self._have("world.json"):
            return _load(self.dir / "world.json")
        rng = self.toolkit.rng.child("discover.world")
        B._ARCHIVE["dir"] = self.archive_dir
        B._ARCHIVE["include_promoted"] = False
        if seed["blend_model"] == "compare":
            results = [B.run_model(rng, m) for m in B.MODELS]
            results.sort(key=lambda r: -r["score"]["score"])
            world = results[0]
            world["compared"] = [{"model": r["model"], "score": r["score"]["score"]} for r in results]
        else:
            world = B.run_model(rng, seed["blend_model"])
        _dump(self.dir / "world.json", world)
        md = (f"# World (blend model: {world['model']}, imagination score {world['score']['score']})\n\n{world['text']}\n\n"
              "## Built from\n" + "\n".join(f"- {f['kind']}: {f['source']}" for f in world["fragments"]) +
              "\n\n## Score\n```\n" + json.dumps(world["score"], indent=2) + "\n```\n")
        (self.dir / "world.md").write_text(md, encoding="utf-8")
        self._say(f"world: {world['model']} score={world['score']['score']} ({len(world['fragments'])} fragments)")
        return world

    # -- 3. create (free invention, no target vocabulary) ---------------------------------
    def step_create(self, provider: Provider, world: dict[str, Any], seed: dict[str, Any]) -> str:
        if self._have("creation.md"):
            return (self.dir / "creation.md").read_text(encoding="utf-8")
        res = provider.agent(P.CREATE_SYSTEM, P.create_prompt(world["text"], seed["depth"]), self.toolkit, [], 1)
        text = res.text.strip()
        (self.dir / "creation.md").write_text(text, encoding="utf-8")
        self._say(f"create: {len(text)} chars")
        return text

    # -- 4. formalize (abstract the structure, still not application-bound) ---------------
    def step_formalize(self, provider: Provider, creation: str) -> str:
        if self._have("formal.md"):
            return (self.dir / "formal.md").read_text(encoding="utf-8")
        res = provider.agent(P.FORMALIZE_SYSTEM, P.formalize_prompt(creation), self.toolkit, [], 1)
        text = res.text.strip()
        (self.dir / "formal.md").write_text(text, encoding="utf-8")
        self._say(f"formalize: {len(text)} chars")
        return text

    # -- 5. propose (the AI's own candidate applications, 0-3, may be none) ---------------
    def step_propose(self, provider: Provider, formal: str) -> str:
        if self._have("proposals.md"):
            return (self.dir / "proposals.md").read_text(encoding="utf-8")
        res = provider.agent(P.PROPOSE_SYSTEM, P.propose_prompt(formal), self.toolkit, [], 1)
        text = res.text.strip()
        (self.dir / "proposals.md").write_text(text, encoding="utf-8")
        self._say(f"propose: {len(text)} chars")
        return text

    # -- 6. archive -------------------------------------------------------------------------
    def step_archive(self, provider: Provider, seed: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
        summary = {
            "seed": self.seed, "blend_model": world["model"], "depth": seed["depth"],
            "provider": provider.name, "model": getattr(provider, "model", ""),
            "imagination_score": world["score"]["score"], "fragments": [f["id"] for f in world["fragments"]],
            "timing_s": self._timing, "total_s": round(time.time() - self._t0, 3),
        }
        _dump(self.dir / "run.json", summary)
        with (Path(self.archive_dir) / "discover_index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: summary[k] for k in ("seed", "blend_model", "depth", "imagination_score", "model")}) + "\n")
        self._say(f"archived -> {self.dir}")
        return summary

    def execute(self, provider: Provider) -> dict[str, Any]:
        seed = self._timed("seed", self.step_seed)
        world = self._timed("world", self.step_world, seed)
        creation = self._timed("create", self.step_create, provider, world, seed)
        formal = self._timed("formalize", self.step_formalize, provider, creation)
        self._timed("propose", self.step_propose, provider, formal)
        return self._timed("archive", self.step_archive, provider, seed, world)
