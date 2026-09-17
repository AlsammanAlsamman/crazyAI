"""`crazyai invent`: one seed, one target, six steps, one folder.

    archive/invent_<seed>_<target>/
        seed.json      step 1  which blend model, which depth (seeded draws)
        harvest.yaml   step 2  fragments the AI added to the corpus (optional; also archive/imagination/)
        world.md       step 3  the blended world + its imagination score (world.json)
        ideas.md       step 4  the native's paragraph and SEED lines (immersion)
        artifact.md    step 5  the engineer's translation, artifact (artifact.c for kernels) + prediction
        measure.json   step 6  the pipeline's own measurement of the artifact, calibration of the prediction
        run.json               summary; appended to archive/invent_index.jsonl

Steps are resumable: an existing file is reused unless force=True.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from crazyai.config import ARCHIVE_DIR, DEFAULT_MAX_TOOL_TURNS
from crazyai.imagination import KINDS, corpus, save_harvest
from crazyai.pipeline import invent_prompts as P
from crazyai.providers.base import Provider
from crazyai.targets import Target, get_target
from crazyai.toolkit.invent import blend as B
from crazyai.toolkit.measure.imagination import score_text
from crazyai.toolkit.registry import Toolkit, build_toolkit


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


_CODE = re.compile(r"```c\s*\n(.*?)```", re.S)
_PRED = re.compile(r"PREDICTION:\s*(?:speedup_vs_blocked\s*=\s*)?([0-9]+(?:\.[0-9]+)?)", re.I)
_SEED = re.compile(r"^\s*SEED:\s*(.+)$", re.M)


@dataclass
class Invent:
    seed: int
    target: str = "matmul"
    blend: str = ""                    # cutup|markov|graft|nest|anneal|evolve|compare|"" (seeded draw)
    harvest: int = 0                   # fragments to ask the AI for before blending (0 = bundled corpus only)
    archive_dir: Path = field(default_factory=lambda: Path(ARCHIVE_DIR))
    force: bool = False
    max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS
    log: Any = print

    def __post_init__(self) -> None:
        self.tgt: Target = get_target(self.target)
        self.dir = Path(self.archive_dir) / f"invent_{self.seed}_{self.target}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.toolkit: Toolkit = build_toolkit(self.seed, ctx={"run_dir": str(self.dir), "seed": self.seed})
        self._t0 = time.time()
        self._timing: dict[str, float] = {}

    def _have(self, name: str) -> bool:
        return (self.dir / name).exists() and not self.force

    def _say(self, msg: str) -> None:
        if self.log:
            self.log(f"[crazyai invent seed={self.seed} {self.target}] {msg}")

    def _timed(self, name: str, fn, *a):
        t0 = time.time()
        out = fn(*a)
        self._timing[name] = round(time.time() - t0, 3)
        return out

    # -- 1. seed ---------------------------------------------------------------------
    def step_seed(self) -> dict[str, Any]:
        if self._have("seed.json"):
            return _load(self.dir / "seed.json")
        rng = self.toolkit.rng
        model = self.blend or rng.choice("invent.blend_model", B.MODELS + ["compare"])
        depth = rng.integer("invent.depth", 1, 3)
        s = {"seed": self.seed, "target": self.target, "blend_model": model, "depth": depth,
             "assumption_focus": rng.choice("invent.assumption", self.tgt.assumptions)}
        _dump(self.dir / "seed.json", s)
        self._say(f"seed: blend={model} depth={depth} focus='{s['assumption_focus']}'")
        return s

    # -- 2. harvest (the AI feeds the corpus) ------------------------------------------
    def step_harvest(self, provider: Provider) -> list[dict[str, Any]]:
        if self.harvest <= 0:
            return []
        if self._have("harvest.yaml"):
            return yaml.safe_load((self.dir / "harvest.yaml").read_text()).get("fragments", [])
        avoid = sorted({f.source for f in corpus()})
        res = provider.agent(P.HARVEST_SYSTEM, P.harvest_prompt(self.harvest, KINDS, avoid), self.toolkit, [], 2)
        frags: list[dict[str, Any]] = []
        m = re.search(r"\[.*\]", res.text, re.S)
        if m:
            try:
                frags = [f for f in json.loads(m.group(0)) if isinstance(f, dict) and f.get("text")]
            except json.JSONDecodeError:
                frags = []
        for i, f in enumerate(frags):
            f.setdefault("id", f"harvest.{self.seed}.{i}")
        path = save_harvest(f"harvest_{self.seed}", frags, self.archive_dir)
        (self.dir / "harvest.yaml").write_text(path.read_text())
        self._say(f"harvest: {len(frags)} fragments -> {path}")
        return frags

    # -- 3. blend ------------------------------------------------------------------------
    def step_world(self, seed: dict[str, Any]) -> dict[str, Any]:
        if self._have("world.json"):
            return _load(self.dir / "world.json")
        rng = self.toolkit.rng.child("invent.world")
        B._ARCHIVE["dir"] = self.archive_dir
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
        (self.dir / "world.md").write_text(md)
        self._say(f"world: {world['model']} score={world['score']['score']} ({len(world['fragments'])} fragments)")
        return world

    # -- 4. immerse ----------------------------------------------------------------------
    def step_immerse(self, provider: Provider, world: dict[str, Any], seed: dict[str, Any]) -> str:
        if self._have("ideas.md"):
            return (self.dir / "ideas.md").read_text()
        res = provider.agent(P.IMMERSE_SYSTEM, P.immerse_prompt(world["text"], self.tgt, seed["depth"]), self.toolkit, [], 1)
        (self.dir / "ideas.md").write_text(res.text)
        seeds = _SEED.findall(res.text)
        self._say(f"immerse: {len(res.text)} chars, {len(seeds)} seeds")
        return res.text

    # -- 5. bend -------------------------------------------------------------------------
    def step_bend(self, provider: Provider, ideas: str) -> dict[str, Any]:
        if self._have("artifact.md"):
            art = (self.dir / "artifact.md").read_text()
        else:
            names = self.toolkit.names(families=self.tgt.measure_families + ["unconventional", "symbolic"])
            res = provider.agent(P.BEND_SYSTEM, P.bend_prompt(ideas, self.tgt, names), self.toolkit, names, self.max_tool_turns)
            art = res.text
            (self.dir / "artifact.md").write_text(art)
            _dump(self.dir / "bend_calls.json", {"turns": res.turns, "stop_reason": res.stop_reason, "usage": res.usage, "calls": res.tool_calls})
        code = _CODE.findall(art)
        pred = _PRED.search(art)
        out = {"text": art, "code": code[-1] if code else None, "prediction": float(pred.group(1)) if pred else None,
               "seeds": _SEED.findall(art)}
        if out["code"] and self.tgt.artifact == "kernel":
            (self.dir / "artifact.c").write_text(out["code"])
        self._say(f"bend: {len(art)} chars, code={'yes' if out['code'] else 'no'}, prediction={out['prediction']}")
        return out

    # -- 6. measure ----------------------------------------------------------------------
    def step_measure(self, bent: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
        if self._have("measure.json"):
            return _load(self.dir / "measure.json")
        m: dict[str, Any] = {"target": self.target, "artifact": self.tgt.artifact, "imagination": world["score"]}
        if self.tgt.measure_tool == "kernel_bench":
            if bent["code"]:
                r = self.toolkit.call("kernel_bench", {"source": bent["code"], "sizes": [64, 256, 512], "budget": 0.3})
                m["measurement"] = r
                m["value"] = r.get("value", 0.0)
                m["status"] = r.get("status", "ERROR")
                if bent["prediction"] and r.get("speedup_vs_blocked"):
                    p, a = bent["prediction"], r["speedup_vs_blocked"]
                    m["calibration"] = round(1 - abs(p - a) / max(p, a), 3)
            else:
                m["measurement"] = {"error": "no ```c block in the artifact"}
                m["value"] = 0.0
                m["status"] = "NO_ARTIFACT"
        else:
            m["measurement"] = {"note": "no automatic measurement for this target; see artifact.md"}
            m["value"] = None
            m["status"] = "UNMEASURED"
        m["prediction"] = bent["prediction"]
        m["discovery"] = round((m["value"] or 0.0) * (0.5 + 0.5 * world["score"]["score"]), 4)
        _dump(self.dir / "measure.json", m)
        self._say(f"measure: status={m['status']} value={m['value']} calibration={m.get('calibration')}")
        return m

    def step_archive(self, provider: Provider, seed: dict[str, Any], world: dict[str, Any], measure: dict[str, Any]) -> dict[str, Any]:
        summary = {
            "seed": self.seed, "target": self.target, "blend_model": world["model"], "depth": seed["depth"],
            "assumption_focus": seed["assumption_focus"], "provider": provider.name, "model": getattr(provider, "model", ""),
            "imagination_score": world["score"]["score"], "status": measure["status"], "value": measure["value"],
            "prediction": measure.get("prediction"), "calibration": measure.get("calibration"), "discovery": measure["discovery"],
            "fragments": [f["id"] for f in world["fragments"]], "rng_log": self.toolkit.rng.log,
            "timing_s": self._timing, "total_s": round(time.time() - self._t0, 3),
        }
        _dump(self.dir / "run.json", summary)
        with (Path(self.archive_dir) / "invent_index.jsonl").open("a") as fh:
            fh.write(json.dumps({k: summary[k] for k in ("seed", "target", "blend_model", "depth", "assumption_focus",
                                                          "imagination_score", "status", "value", "prediction", "calibration",
                                                          "discovery", "model")}) + "\n")
        self._say(f"archived -> {self.dir}")
        return summary

    def execute(self, provider: Provider) -> dict[str, Any]:
        seed = self._timed("seed", self.step_seed)
        self._timed("harvest", self.step_harvest, provider)
        world = self._timed("world", self.step_world, seed)
        ideas = self._timed("immerse", self.step_immerse, provider, world, seed)
        bent = self._timed("bend", self.step_bend, provider, ideas)
        measure = self._timed("measure", self.step_measure, bent, world)
        return self._timed("archive", self.step_archive, provider, seed, world, measure)


def load_invent_index(archive_dir: Path | str = ARCHIVE_DIR) -> list[dict[str, Any]]:
    p = Path(archive_dir) / "invent_index.jsonl"
    if not p.exists():
        return []
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["seed"], r["target"])] = r
    return list(latest.values())
