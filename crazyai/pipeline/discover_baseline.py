"""`crazyai discover-baseline`: the direct-prompt condition for `discover`, matched on everything except

the world-blend + 3-persona (maker / observer / scientist) narrative chain. No blended corpus, no alien-maker
persona, no formalize-then-propose handoff - one plain "creative scientist" persona is asked, in a single call,
to invent something and propose what it's for.

    archive/discover_baseline_<seed>/
        seed.json      {"seed": ..., "depth": ...} (depth only - no blend_model, no world)
        answer.md      the direct call's raw text: STRUCTURE + PROPOSALS sections
        run.json               summary; appended to archive/discover_baseline_index.jsonl (never
                                discover_index.jsonl or invent_index.jsonl - there is no value/status here either)

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
from crazyai.toolkit.registry import Toolkit, build_toolkit


@dataclass
class DiscoverBaseline:
    seed: int
    archive_dir: Path = field(default_factory=lambda: Path(ARCHIVE_DIR))
    force: bool = False
    max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS
    log: Any = lambda msg: print(msg, flush=True)

    def __post_init__(self) -> None:
        self.dir = Path(self.archive_dir) / f"discover_baseline_{self.seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.toolkit: Toolkit = build_toolkit(self.seed, ctx={"run_dir": str(self.dir), "seed": self.seed})
        self._t0 = time.time()
        self._timing: dict[str, float] = {}

    def _have(self, name: str) -> bool:
        return (self.dir / name).exists() and not self.force

    def _say(self, msg: str) -> None:
        if self.log:
            self.log(f"[crazyai discover-baseline seed={self.seed}] {msg}")

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
        depth = rng.integer("discover_baseline.depth", 1, 3)
        s = {"seed": self.seed, "depth": depth}
        _dump(self.dir / "seed.json", s)
        self._say(f"seed: depth={depth}")
        return s

    # -- 2. direct (one call, no world, no persona chain) ---------------------------------
    def step_direct(self, provider: Provider, seed: dict[str, Any]) -> str:
        if self._have("answer.md"):
            return (self.dir / "answer.md").read_text(encoding="utf-8")
        res = provider.agent(P.DISCOVER_DIRECT_SYSTEM, P.discover_direct_prompt(seed["depth"]), self.toolkit, [], 1)
        text = res.text.strip()
        (self.dir / "answer.md").write_text(text, encoding="utf-8")
        self._say(f"direct: {len(text)} chars")
        return text

    # -- 3. archive -------------------------------------------------------------------------
    def step_archive(self, provider: Provider, seed: dict[str, Any], answer: str) -> dict[str, Any]:
        summary = {
            "seed": self.seed, "depth": seed["depth"], "provider": provider.name,
            "model": getattr(provider, "model", ""), "answer_chars": len(answer),
            "timing_s": self._timing, "total_s": round(time.time() - self._t0, 3),
        }
        _dump(self.dir / "run.json", summary)
        with (Path(self.archive_dir) / "discover_baseline_index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: summary[k] for k in ("seed", "depth", "answer_chars", "model")}) + "\n")
        self._say(f"archived -> {self.dir}")
        return summary

    def execute(self, provider: Provider) -> dict[str, Any]:
        seed = self._timed("seed", self.step_seed)
        answer = self._timed("direct", self.step_direct, provider, seed)
        return self._timed("archive", self.step_archive, provider, seed, answer)


def load_discover_baseline_index(archive_dir: Path | str = ARCHIVE_DIR) -> list[dict[str, Any]]:
    p = Path(archive_dir) / "discover_baseline_index.jsonl"
    if not p.exists():
        return []
    latest: dict[int, dict[str, Any]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["seed"]] = r
    return list(latest.values())
