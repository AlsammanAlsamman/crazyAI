"""`crazyai invent-disguise`: the narrative applies to the PROBLEM, not to Claude.

Every other narrative mechanism in this project gives Claude a persona and (usually) a blended
corpus world it must inhabit and find its own mapping from. This one instead deliberately
transforms the real problem itself into a concrete, structurally-isomorphic problem in an
everyday, non-technical domain - no math, no code, no jargon - gets several candidate solutions
to *that* disguised problem in plain words, then hands those disguised solutions back to a
plain "rigorous engineer" persona and asks it to translate the most promising one (or a
combination) into a real, working solution to the ORIGINAL problem.

No world-blend step at all - this isolates "disguise the problem, then translate" as a third kind
of narrative mechanism, distinct from both corpus-blend-based narrative (Invent/WorldOnly) and
no-narrative (Baseline). Closest published relative: analogical prompting (Yasunaga et al.,
"Large Language Models as Analogical Reasoners", arXiv:2310.01714), which self-generates a
relevant analogous problem+solution as in-context scaffolding before solving the real problem
directly - this pipeline goes further by deliberately disguising the real problem into a concrete
non-technical domain (not just a nearby math exemplar) and generating multiple disguised
solutions before a separate, explicit translate-back call.

    archive/disguise_<seed>_<target>/
        disguise.md    the transformed problem + 3 candidate solutions, no math/code
        artifact.md    the translated-back real solution, artifact (artifact.c) + prediction
        measure.json   the pipeline's own measurement, calibration of the prediction
        run.json               summary; appended to archive/disguise_index.jsonl (never
                                baseline_index.jsonl, invent_index.jsonl, or world_only_index.jsonl)

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
from crazyai.toolkit.registry import Toolkit, build_toolkit


@dataclass
class Disguise:
    seed: int
    target: str = "matmul"
    assumption: str = ""               # exact assumption text, or its 0-based index; "" = seeded draw
    archive_dir: Path = field(default_factory=lambda: Path(ARCHIVE_DIR))
    force: bool = False
    max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS
    log: Any = lambda msg: print(msg, flush=True)

    def __post_init__(self) -> None:
        self.tgt: Target = get_target(self.target)
        self.dir = Path(self.archive_dir) / f"disguise_{self.seed}_{self.target}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.toolkit: Toolkit = build_toolkit(self.seed, ctx={"run_dir": str(self.dir), "seed": self.seed})
        self._t0 = time.time()
        self._timing: dict[str, float] = {}
        self._pinned_assumption = bool(self.assumption)

    def _have(self, name: str) -> bool:
        return (self.dir / name).exists() and not self.force

    def _say(self, msg: str) -> None:
        if self.log:
            self.log(f"[crazyai invent-disguise seed={self.seed} {self.target}] {msg}")

    def _timed(self, name: str, fn, *a):
        t0 = time.time()
        out = fn(*a)
        self._timing[name] = round(time.time() - t0, 3)
        return out

    def _resolve_assumption(self) -> str:
        if not self.assumption:
            return self.toolkit.rng.choice("disguise.assumption_focus", self.tgt.assumptions)
        assumptions = self.tgt.assumptions
        if self.assumption in assumptions:
            return self.assumption
        if self.assumption.lstrip("-").isdigit() and 0 <= int(self.assumption) < len(assumptions):
            return assumptions[int(self.assumption)]
        raise ValueError(f"--assumption {self.assumption!r} matches none of target {self.target!r}'s assumptions:\n" +
                          "\n".join(f"  {i}: {a}" for i, a in enumerate(assumptions)))

    def step_disguise(self, provider: Provider) -> str:
        if self._have("disguise.md"):
            return (self.dir / "disguise.md").read_text(encoding="utf-8")
        focus = self._resolve_assumption()
        (self.dir / "assumption_focus.txt").write_text(focus, encoding="utf-8")
        res = provider.agent(P.DISGUISE_SYSTEM, P.disguise_prompt(self.tgt, focus), self.toolkit, [], 1)
        text = res.text.strip()
        (self.dir / "disguise.md").write_text(text, encoding="utf-8")
        self._say(f"disguise: {len(text)} chars")
        return text

    def step_translate(self, provider: Provider, disguise_text: str) -> dict[str, Any]:
        if self._have("artifact.md"):
            art = (self.dir / "artifact.md").read_text(encoding="utf-8")
        else:
            names = self.toolkit.names(families=self.tgt.measure_families + ["unconventional", "symbolic"])
            res = provider.agent(P.DIRECT_SYSTEM, P.translate_prompt(disguise_text, self.tgt, names),
                                  self.toolkit, names, self.max_tool_turns)
            art = res.text
            (self.dir / "artifact.md").write_text(art, encoding="utf-8")
            _dump(self.dir / "translate_calls.json", {"turns": res.turns, "stop_reason": res.stop_reason,
                                                        "usage": res.usage, "calls": res.tool_calls})
        code = _CODE.findall(art)
        pred = _PRED.search(art)
        out = {"text": art, "code": code[-1] if code else None, "prediction": float(pred.group(1)) if pred else None}
        if out["code"] and self.tgt.artifact == "kernel":
            (self.dir / "artifact.c").write_text(out["code"], encoding="utf-8")
        self._say(f"translate: {len(art)} chars, code={'yes' if out['code'] else 'no'}, prediction={out['prediction']}")
        return out

    def step_measure(self, translated: dict[str, Any]) -> dict[str, Any]:
        if self._have("measure.json"):
            return _load(self.dir / "measure.json")
        m: dict[str, Any] = {"target": self.target, "artifact": self.tgt.artifact, "condition": "disguise"}
        if self.tgt.measure_tool:
            if translated["code"]:
                r = self.toolkit.call(self.tgt.measure_tool, {"source": translated["code"], "budget": 0.3})
                m["measurement"] = r
                m["value"] = r.get("value", 0.0)
                m["status"] = r.get("status", "ERROR")
                if translated["prediction"] and r.get("prediction_target"):
                    p, a = translated["prediction"], r["prediction_target"]
                    m["calibration"] = round(1 - abs(p - a) / max(p, a), 3)
            else:
                m["measurement"] = {"error": "no ```c block in the artifact"}
                m["value"] = 0.0
                m["status"] = "NO_ARTIFACT"
        else:
            m["measurement"] = {"note": "no automatic measurement for this target; see artifact.md"}
            m["value"] = None
            m["status"] = "UNMEASURED"
        m["prediction"] = translated["prediction"]
        _dump(self.dir / "measure.json", m)
        self._say(f"measure: status={m['status']} value={m['value']} calibration={m.get('calibration')}")
        return m

    def step_archive(self, provider: Provider, translated: dict[str, Any], measure: dict[str, Any]) -> dict[str, Any]:
        summary = {
            "seed": self.seed, "target": self.target, "condition": "disguise", "provider": provider.name,
            "model": getattr(provider, "model", ""), "status": measure["status"], "value": measure["value"],
            "prediction": measure.get("prediction"), "calibration": measure.get("calibration"),
            "assumption_focus": (self.dir / "assumption_focus.txt").read_text(encoding="utf-8")
            if (self.dir / "assumption_focus.txt").exists() else "",
            "timing_s": self._timing, "total_s": round(time.time() - self._t0, 3),
        }
        _dump(self.dir / "run.json", summary)
        with (Path(self.archive_dir) / "disguise_index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: summary[k] for k in ("seed", "target", "condition", "status", "value",
                                                          "prediction", "calibration", "assumption_focus",
                                                          "model")}) + "\n")
        self._say(f"archived -> {self.dir}")
        return summary

    def execute(self, provider: Provider) -> dict[str, Any]:
        disguise_text = self._timed("disguise", self.step_disguise, provider)
        translated = self._timed("translate", self.step_translate, provider, disguise_text)
        measure = self._timed("measure", self.step_measure, translated)
        return self._timed("archive", self.step_archive, provider, translated, measure)


def load_disguise_index(archive_dir: Path | str = ARCHIVE_DIR) -> list[dict[str, Any]]:
    p = Path(archive_dir) / "disguise_index.jsonl"
    if not p.exists():
        return []
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["seed"], r["target"])] = r
    return list(latest.values())
