"""`crazyai invent-world-only`: does exposing the model to rich, blended narrative material -

with NO persona swap, no "you are not on Earth" framing - help it solve target problems better
than `Baseline`'s zero-material condition? Isolates exactly one variable against `Baseline`:
material present or not. Everything else (DIRECT_SYSTEM persona, no assumption pinning, the
identical task/contract text via `_direct_task_block`) stays exactly as `Baseline` already has it.

Motivated by the imagination-effect tuning campaign (crazyai-trials/AIM.md, run 2026-09-21):
`world_only` (material, no persona) was statistically indistinguishable from the full persona+
material structure on a divergent-thinking task. This tests whether that finding transfers to
real target-solving, specifically on the targets (alignment, hash, dijkstra) where the full
persona+material `invent` pipeline lost decisively to `Baseline`.

    archive/world_only_<seed>_<target>/
        world.json, world.md   the blended narrative material (same corpus-blend machinery)
        artifact.md    the engineer's answer, artifact (artifact.c) + prediction
        measure.json   the pipeline's own measurement, calibration of the prediction
        run.json               summary; appended to archive/world_only_index.jsonl (never
                                baseline_index.jsonl or invent_index.jsonl)

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
from crazyai.pipeline.invent import _CODE, _PRED, _dump, _load
from crazyai.providers.base import Provider
from crazyai.targets import Target, get_target
from crazyai.toolkit.invent import blend as B
from crazyai.toolkit.registry import Toolkit, build_toolkit


@dataclass
class WorldOnly:
    seed: int
    target: str = "matmul"
    blend: str = ""                    # cutup|markov|graft|nest|anneal|evolve|compare|"" (seeded draw)
    archive_dir: Path = field(default_factory=lambda: Path(ARCHIVE_DIR))
    force: bool = False
    max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS
    log: Any = lambda msg: print(msg, flush=True)

    def __post_init__(self) -> None:
        self.tgt: Target = get_target(self.target)
        self.dir = Path(self.archive_dir) / f"world_only_{self.seed}_{self.target}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.toolkit: Toolkit = build_toolkit(self.seed, ctx={"run_dir": str(self.dir), "seed": self.seed})
        self._t0 = time.time()
        self._timing: dict[str, float] = {}

    def _have(self, name: str) -> bool:
        return (self.dir / name).exists() and not self.force

    def _say(self, msg: str) -> None:
        if self.log:
            self.log(f"[crazyai invent-world-only seed={self.seed} {self.target}] {msg}")

    def _timed(self, name: str, fn, *a):
        t0 = time.time()
        out = fn(*a)
        self._timing[name] = round(time.time() - t0, 3)
        return out

    def step_world(self) -> dict[str, Any]:
        if self._have("world.json"):
            return _load(self.dir / "world.json")
        rng = self.toolkit.rng.child("world_only.world")
        model = self.blend or rng.choice("world_only.blend_model", B.MODELS)
        world = B.run_model(rng, model)
        _dump(self.dir / "world.json", world)
        md = (f"# World (blend model: {world['model']}, imagination score {world['score']['score']})\n\n{world['text']}\n\n"
              "## Built from\n" + "\n".join(f"- {f['kind']}: {f['source']}" for f in world["fragments"]) +
              "\n\n## Score\n```\n" + json.dumps(world["score"], indent=2) + "\n```\n")
        (self.dir / "world.md").write_text(md, encoding="utf-8")
        self._say(f"world: {world['model']} score={world['score']['score']} ({len(world['fragments'])} fragments)")
        return world

    def step_solve(self, provider: Provider, world: dict[str, Any]) -> dict[str, Any]:
        if self._have("artifact.md"):
            art = (self.dir / "artifact.md").read_text(encoding="utf-8")
        else:
            names = self.toolkit.names(families=self.tgt.measure_families + ["unconventional", "symbolic"])
            res = provider.agent(P.DIRECT_SYSTEM, P.world_only_prompt(world["text"], self.tgt, names),
                                  self.toolkit, names, self.max_tool_turns)
            art = res.text
            (self.dir / "artifact.md").write_text(art, encoding="utf-8")
            _dump(self.dir / "solve_calls.json", {"turns": res.turns, "stop_reason": res.stop_reason, "usage": res.usage, "calls": res.tool_calls})
        code = _CODE.findall(art)
        pred = _PRED.search(art)
        out = {"text": art, "code": code[-1] if code else None, "prediction": float(pred.group(1)) if pred else None}
        if out["code"] and self.tgt.artifact == "kernel":
            (self.dir / "artifact.c").write_text(out["code"], encoding="utf-8")
        self._say(f"solve: {len(art)} chars, code={'yes' if out['code'] else 'no'}, prediction={out['prediction']}")
        return out

    def step_measure(self, solved: dict[str, Any]) -> dict[str, Any]:
        if self._have("measure.json"):
            return _load(self.dir / "measure.json")
        m: dict[str, Any] = {"target": self.target, "artifact": self.tgt.artifact, "condition": "world_only"}
        if self.tgt.measure_tool:
            if solved["code"]:
                r = self.toolkit.call(self.tgt.measure_tool, {"source": solved["code"], "budget": 0.3})
                m["measurement"] = r
                m["value"] = r.get("value", 0.0)
                m["status"] = r.get("status", "ERROR")
                if solved["prediction"] and r.get("prediction_target"):
                    p, a = solved["prediction"], r["prediction_target"]
                    m["calibration"] = round(1 - abs(p - a) / max(p, a), 3)
            else:
                m["measurement"] = {"error": "no ```c block in the artifact"}
                m["value"] = 0.0
                m["status"] = "NO_ARTIFACT"
        else:
            m["measurement"] = {"note": "no automatic measurement for this target; see artifact.md"}
            m["value"] = None
            m["status"] = "UNMEASURED"
        m["prediction"] = solved["prediction"]
        _dump(self.dir / "measure.json", m)
        self._say(f"measure: status={m['status']} value={m['value']} calibration={m.get('calibration')}")
        return m

    def step_archive(self, provider: Provider, solved: dict[str, Any], measure: dict[str, Any]) -> dict[str, Any]:
        summary = {
            "seed": self.seed, "target": self.target, "condition": "world_only", "provider": provider.name,
            "model": getattr(provider, "model", ""), "status": measure["status"], "value": measure["value"],
            "prediction": measure.get("prediction"), "calibration": measure.get("calibration"),
            "timing_s": self._timing, "total_s": round(time.time() - self._t0, 3),
        }
        _dump(self.dir / "run.json", summary)
        with (Path(self.archive_dir) / "world_only_index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: summary[k] for k in ("seed", "target", "condition", "status", "value",
                                                          "prediction", "calibration", "model")}) + "\n")
        self._say(f"archived -> {self.dir}")
        return summary

    def execute(self, provider: Provider) -> dict[str, Any]:
        world = self._timed("world", self.step_world)
        solved = self._timed("solve", self.step_solve, provider, world)
        measure = self._timed("measure", self.step_measure, solved)
        return self._timed("archive", self.step_archive, provider, solved, measure)


def load_world_only_index(archive_dir: Path | str = ARCHIVE_DIR) -> list[dict[str, Any]]:
    p = Path(archive_dir) / "world_only_index.jsonl"
    if not p.exists():
        return []
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["seed"], r["target"])] = r
    return list(latest.values())
