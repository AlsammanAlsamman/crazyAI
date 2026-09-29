"""`crazyai invent-disguise-all`: let the benchmark choose, not the engineer persona.

`Disguise` hands all three disguised solutions to one engineer call and lets it pick. Inspection of
the 2026-09-22 disguise batch showed the disguise step can carry the right idea (seed 7581
described antidiagonal wavefront parallelism in plain words) while the translate step picked a
weaker one; the 2026-09-29 retroactive immersion check located the loss at that translate step.

This pipeline removes the choice: each of the three solutions is translated in its own call,
told to stay faithful to that one solution's mechanism, all three kernels are measured, and the
fastest exact one is the result. With `--from-disguise`, the disguise text is copied from an
existing `disguise_<seed>_<target>` run, so the disguise itself is held fixed and only the
choice step differs from that run - a clean comparison against the engineer's own pick.

    archive/disguise_all_<seed>_<target>/
        disguise.md, assumption_focus.txt   the disguise (copied or generated)
        sol_<k>/artifact.md, artifact.c, measure.json, translate_calls.json   one per solution
        measure.json   {"solutions": [...], "best": k, "value": ...}
        run.json       summary; appended to archive/disguise_all_index.jsonl

Steps are resumable: an existing file is reused unless force=True.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crazyai.pipeline import invent_prompts as P
from crazyai.pipeline.disguise import Disguise
from crazyai.pipeline.invent import _CODE, _PRED, _dump, _load
from crazyai.providers.base import Provider

_SOLUTION = re.compile(r"^\W*SOLUTION\s*(\d)\W*$", re.M | re.I)


def split_disguise(text: str) -> tuple[str, dict[int, str]]:
    """Return (the transformed problem, {k: solution k text}) from a disguise.md."""
    marks = list(_SOLUTION.finditer(text))
    if not marks:
        return text.strip(), {}
    problem = text[: marks[0].start()].strip()
    sols = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        sols[int(m.group(1))] = text[m.end():end].strip()
    return problem, sols


@dataclass
class DisguiseAll(Disguise):
    from_disguise: bool = False        # copy disguise.md from archive/disguise_<seed>_<target>/

    def __post_init__(self) -> None:
        super().__post_init__()
        old = self.dir
        self.dir = Path(self.archive_dir) / f"disguise_all_{self.seed}_{self.target}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.toolkit.ctx["run_dir"] = str(self.dir)
        if not any(old.iterdir()):
            old.rmdir()                # the parent created an empty disguise_<seed> dir we don't use

    def _say(self, msg: str) -> None:
        if self.log:
            self.log(f"[crazyai invent-disguise-all seed={self.seed} {self.target}] {msg}")

    def step_disguise(self, provider: Provider) -> str:
        if self.from_disguise and not self._have("disguise.md"):
            src = Path(self.archive_dir) / f"disguise_{self.seed}_{self.target}"
            if not (src / "disguise.md").exists():
                raise FileNotFoundError(f"--from-disguise: {src / 'disguise.md'} does not exist")
            for name in ("disguise.md", "assumption_focus.txt"):
                if (src / name).exists():
                    shutil.copy(src / name, self.dir / name)
            self._say(f"disguise: copied from {src.name}")
        return super().step_disguise(provider)

    def translate_one(self, provider: Provider, k: int, problem: str, solution: str) -> dict[str, Any]:
        d = self.dir / f"sol_{k}"
        d.mkdir(exist_ok=True)
        if (d / "artifact.md").exists() and not self.force:
            art = (d / "artifact.md").read_text(encoding="utf-8")
        else:
            names = self.toolkit.names(families=self.tgt.measure_families + ["unconventional", "symbolic"])
            res = provider.agent(P.DIRECT_SYSTEM, P.translate_one_prompt(problem, solution, self.tgt, names),
                                 self.toolkit, names, self.max_tool_turns)
            art = res.text
            (d / "artifact.md").write_text(art, encoding="utf-8")
            _dump(d / "translate_calls.json", {"turns": res.turns, "stop_reason": res.stop_reason,
                                                "usage": res.usage, "calls": res.tool_calls})
        code = _CODE.findall(art)
        pred = _PRED.search(art)
        out = {"text": art, "code": code[-1] if code else None, "prediction": float(pred.group(1)) if pred else None}
        if out["code"]:
            (d / "artifact.c").write_text(out["code"], encoding="utf-8")
        self._say(f"solution {k}: translated, code={'yes' if out['code'] else 'no'}, prediction={out['prediction']}")
        return out

    def measure_one(self, k: int, translated: dict[str, Any]) -> dict[str, Any]:
        path = self.dir / f"sol_{k}" / "measure.json"
        if path.exists() and not self.force:
            return _load(path)
        m: dict[str, Any] = {"solution": k, "prediction": translated["prediction"]}
        if translated["code"]:
            r = self.toolkit.call(self.tgt.measure_tool, {"source": translated["code"], "budget": 0.3})
            m.update(measurement=r, value=r.get("value", 0.0), status=r.get("status", "ERROR"))
        else:
            m.update(measurement={"error": "no ```c block in the artifact"}, value=0.0, status="NO_ARTIFACT")
        _dump(path, m)
        self._say(f"solution {k}: status={m['status']} value={m['value']}")
        return m

    def execute(self, provider: Provider) -> dict[str, Any]:
        text = self._timed("disguise", self.step_disguise, provider)
        problem, sols = split_disguise(text)
        if len(sols) < 2:
            raise ValueError(f"disguise.md has {len(sols)} SOLUTION sections; need at least 2")
        results = []
        for k in sorted(sols):
            tr = self._timed(f"translate_{k}", self.translate_one, provider, k, problem, sols[k])
            results.append(self._timed(f"measure_{k}", self.measure_one, k, tr))
        exact = [r for r in results if r["status"] == "exact"]
        best = max(exact, key=lambda r: r["value"]) if exact else max(results, key=lambda r: r["value"] or 0)
        m = {"target": self.target, "condition": "disguise_all",
             "solutions": [{k: r[k] for k in ("solution", "status", "value", "prediction")} for r in results],
             "best": best["solution"], "status": best["status"], "value": best["value"]}
        _dump(self.dir / "measure.json", m)
        summary = {
            "seed": self.seed, "target": self.target, "condition": "disguise_all", "provider": provider.name,
            "model": getattr(provider, "model", ""), "from_disguise": self.from_disguise,
            "status": m["status"], "value": m["value"], "best": m["best"], "solutions": m["solutions"],
            "assumption_focus": (self.dir / "assumption_focus.txt").read_text(encoding="utf-8")
            if (self.dir / "assumption_focus.txt").exists() else "",
            "timing_s": self._timing, "total_s": round(time.time() - self._t0, 3),
        }
        _dump(self.dir / "run.json", summary)
        with (Path(self.archive_dir) / "disguise_all_index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: summary[k] for k in ("seed", "target", "condition", "status", "value", "best",
                                                          "solutions", "assumption_focus", "model")}) + "\n")
        self._say(f"best = solution {m['best']} ({m['status']}, {m['value']}) -> {self.dir}")
        return summary


def load_disguise_all_index(archive_dir: Path | str) -> list[dict[str, Any]]:
    p = Path(archive_dir) / "disguise_all_index.jsonl"
    if not p.exists():
        return []
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["seed"], r["target"])] = r
    return list(latest.values())
