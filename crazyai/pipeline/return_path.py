"""`crazyai diagnose-return`: the diagnosis kit's return-path probe (Q3) on real targets.

The 2026-09-29 retroactive check found immersion saturated and located the loss at the
translate ("bend") step: the native's idea appears in-world and is then dropped, re-chosen or
built badly. This probe varies ONLY that step. Each source is an archived `invent_<seed>_<target>`
run; its `ideas.md` (the native's text) and assumption focus are held fixed, and the bend call is
re-run under each return-path prompt variant, the same day, with the same provider. Every
kernel is measured, and a blind judge call (it sees the native's text and the code, never the
prompt variant, the engineer's own explanation or the speed) rates whether the native's
mechanism survived into the code, or whether the engineer fell back to the textbook method.

Variants form a ladder, each adding one change to the one before:

    rp0_current    today's bend_prompt, unchanged (the control)
    rp1_no_known   rp0 minus step 4's "let your mechanism arrive at the known technique" sentence
    rp2_faithful   rp1 plus disguise-all's faithful-translation instruction

    archive/diagnose_return/<variant>/<source>/
        artifact.md, artifact.c, measure.json, judge.json, bend_calls.json
    archive/diagnose_return/orig/<source>/judge.json     the archived artifact, judged the same way
    archive/diagnose_return/diagnose_log.jsonl           one row per (variant, source)

Steps are resumable: an existing file is reused unless force=True.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from crazyai.config import ARCHIVE_DIR
from crazyai.pipeline import invent_prompts as P
from crazyai.pipeline.invent import _CODE, _PRED, _dump, _load
from crazyai.providers.base import Provider
from crazyai.targets import Target, get_target
from crazyai.toolkit.registry import build_toolkit

# the 15 default sources: the latest assumption-pinned invent batch for each hard target
DEFAULT_SOURCES = ([f"invent_{s}_alignment" for s in (7011, 7022, 7023, 7024, 7025)] +
                   [f"invent_{s}_hash" for s in range(7221, 7226)] +
                   [f"invent_{s}_dijkstra" for s in range(7521, 7526)])

_KNOWN = ("if a well-known, validated real-world technique already satisfies the assumption "
          "you're breaking, let your mechanism arrive at that technique rather than inventing a new one just "
          "because you can - a validated known technique beats a novel untested one. If your own VERDICT")

_FAITHFUL = (
    "6. Once you have chosen a seed, build THAT seed's mechanism. Do not switch to a different approach or to the "
    "textbook method; if it turns out slow, improve how you implement this mechanism rather than replacing it. In "
    "ARTIFACT, say which part of the code implements which part of the native's mechanism.\n")


def _rp0(prompt: str) -> str:
    return prompt


def _rp1(prompt: str) -> str:
    assert _KNOWN in prompt, "bend_prompt changed: step 4's known-technique sentence not found"
    return prompt.replace(_KNOWN, "If your own VERDICT", 1)


def _rp2(prompt: str) -> str:
    p = _rp1(prompt)
    anchor = "Tools available:"
    assert anchor in p
    return p.replace(anchor, _FAITHFUL + anchor, 1)


VARIANTS: dict[str, Callable[[str], str]] = {"rp0_current": _rp0, "rp1_no_known": _rp1, "rp2_faithful": _rp2}

JUDGE_SYSTEM = (
    "You are a careful, skeptical code reviewer. You compare a plain-language description of ideas with a piece of "
    "C code and decide, from the code alone, which idea (if any) the code actually implements. You ignore comments "
    "and names that merely claim a link; you judge the mechanism the code executes."
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "native_mechanisms": {"type": "array", "items": {"type": "string"},
                              "description": "one short line per distinct mechanism the native proposes (its SEED lines)"},
        "implemented_mechanism": {"type": "integer",
                                  "description": "1-based index into native_mechanisms of the one the code implements; 0 if none"},
        "survival": {"type": "string", "enum": ["full", "partial", "none"],
                     "description": "full = the code's core is that mechanism; partial = recognisable but diluted or "
                                    "only a minor part; none = no native mechanism is present"},
        "fallback": {"type": "boolean",
                     "description": "true if the code's core technique is the known way below, or a generic textbook "
                                    "optimisation not derived from any native mechanism"},
        "core_technique": {"type": "string", "description": "the code's core technique in standard technical terms, one line"},
    },
    "required": ["native_mechanisms", "implemented_mechanism", "survival", "fallback", "core_technique"],
}


def judge_prompt(ideas: str, target: Target, code: str) -> str:
    return (
        "=== IDEAS (a native of an imagined world describing how they would meet a need) ===\n" + ideas.strip() +
        "\n=== END IDEAS ===\n\n"
        f"The real problem these ideas were meant for: {target.problem}\n"
        f"The known way to solve it: {target.known_way or 'the textbook method'}\n\n"
        "=== CODE ===\n```c\n" + code.strip() + "\n```\n=== END CODE ===\n\n"
        "List the native's distinct mechanisms, then decide which one the code's core actually implements, how fully, "
        "and whether the code instead falls back to the known way or a generic textbook optimisation."
    )


def prompt_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


@dataclass
class ReturnPath:
    source: str                        # archive dir name of an invent run, e.g. invent_7022_alignment
    variant: str                       # a key of VARIANTS, or "orig" to judge the archived artifact only
    archive_dir: Path = Path(ARCHIVE_DIR)
    force: bool = False
    log: Any = lambda msg: print(msg, flush=True)

    def __post_init__(self) -> None:
        if self.variant != "orig" and self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; choose from {', '.join(VARIANTS)} or orig")
        self.archive_dir = Path(self.archive_dir)
        self.src = self.archive_dir / self.source
        if not (self.src / "ideas.md").exists():
            raise FileNotFoundError(f"{self.src / 'ideas.md'} not found")
        self.target = self.source.rsplit("_", 1)[1]
        self.tgt: Target = get_target(self.target)
        self.dir = self.archive_dir / "diagnose_return" / self.variant / self.source
        self.dir.mkdir(parents=True, exist_ok=True)
        self.toolkit = build_toolkit(0, ctx={"run_dir": str(self.dir)})

    def _have(self, name: str) -> bool:
        return (self.dir / name).exists() and not self.force

    def _say(self, msg: str) -> None:
        if self.log:
            self.log(f"[diagnose-return {self.variant} {self.source}] {msg}")

    def ideas(self) -> str:
        return (self.src / "ideas.md").read_text(encoding="utf-8")

    def focus(self) -> str:
        run = self.src / "run.json"
        return _load(run).get("assumption_focus", "") if run.exists() else ""

    def build_prompt(self) -> str:
        names = self.toolkit.names(families=self.tgt.measure_families + ["unconventional", "symbolic"])
        base = P.bend_prompt(self.ideas(), self.tgt, names, self.focus())
        return VARIANTS[self.variant](base)

    def step_bend(self, provider: Provider) -> dict[str, Any]:
        if self.variant == "orig":
            art = (self.src / "artifact.md").read_text(encoding="utf-8")
        elif self._have("artifact.md"):
            art = (self.dir / "artifact.md").read_text(encoding="utf-8")
        else:
            prompt = self.build_prompt()
            names = self.toolkit.names(families=self.tgt.measure_families + ["unconventional", "symbolic"])
            res = provider.agent(P.BEND_SYSTEM, prompt, self.toolkit, names)
            art = res.text
            (self.dir / "artifact.md").write_text(art, encoding="utf-8")
            _dump(self.dir / "bend_calls.json", {"turns": res.turns, "model": res.model, "prompt_hash": prompt_hash(prompt)})
        code = _CODE.findall(art)
        pred = _PRED.search(art)
        out = {"code": code[-1] if code else None, "prediction": float(pred.group(1)) if pred else None}
        if out["code"] and self.variant != "orig":
            (self.dir / "artifact.c").write_text(out["code"], encoding="utf-8")
        return out

    def step_measure(self, bent: dict[str, Any]) -> dict[str, Any]:
        if self.variant == "orig":
            m = _load(self.src / "measure.json")
            return {"status": m.get("status"), "value": m.get("value"), "archived": True}
        if self._have("measure.json"):
            return _load(self.dir / "measure.json")
        m: dict[str, Any] = {"prediction": bent["prediction"]}
        if bent["code"]:
            r = self.toolkit.call(self.tgt.measure_tool, {"source": bent["code"], "budget": 0.3})
            m.update(measurement=r, value=r.get("value", 0.0), status=r.get("status", "ERROR"))
        else:
            m.update(measurement={"error": "no ```c block in the artifact"}, value=0.0, status="NO_ARTIFACT")
        _dump(self.dir / "measure.json", m)
        return m

    def step_judge(self, provider: Provider, bent: dict[str, Any]) -> dict[str, Any]:
        if self._have("judge.json"):
            return _load(self.dir / "judge.json")
        if not bent["code"]:
            j: dict[str, Any] = {"survival": "none", "fallback": None, "note": "no code"}
        else:
            j = provider.structured(JUDGE_SYSTEM, judge_prompt(self.ideas(), self.tgt, bent["code"]), JUDGE_SCHEMA)
            j["valid"] = j.get("survival") in ("full", "partial", "none")
        _dump(self.dir / "judge.json", j)
        return j

    def execute(self, provider: Provider) -> dict[str, Any]:
        t0 = time.time()
        bent = self.step_bend(provider)
        m = self.step_measure(bent)
        j = self.step_judge(provider, bent)
        row = {"variant": self.variant, "source": self.source, "target": self.target,
               "prompt_hash": prompt_hash(self.build_prompt()) if self.variant != "orig" else None,
               "status": m.get("status"), "value": m.get("value"), "survival": j.get("survival"),
               "fallback": j.get("fallback"), "core_technique": j.get("core_technique"),
               "provider": provider.name, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "total_s": round(time.time() - t0, 1)}
        with (self.archive_dir / "diagnose_return" / "diagnose_log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self._say(f"status={row['status']} value={row['value']} survival={row['survival']} fallback={row['fallback']}")
        return row
