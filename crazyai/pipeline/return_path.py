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

Anti-fallback strategies (2026-09-30), each compared against a same-day control:

    c0_today       rp0's prompt again, run alongside the strategies below (the control)
    i1_hidden      the textbook is hidden: no known way, no list of silent assumptions, and the
                   contract's example kernel (itself the textbook method) is replaced by the bare signature
    i2_recipe      i1, plus the native first writes its chosen mechanism as a numbered in-world recipe
                   (a second native call) and the engineer implements that recipe step by step
    i3_gate        rp0's prompt, then the blind judge checks the kernel; if it is a fallback the engineer
                   is told what it built and asked to rebuild the native's mechanism, up to 2 retries

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


VARIANTS: dict[str, Callable[[str], str]] = {"rp0_current": _rp0, "rp1_no_known": _rp1, "rp2_faithful": _rp2,
                                             "c0_today": _rp0, "i3_gate": _rp0}
STRATEGIES = ("i1_hidden", "i2_recipe")
ANTI_FALLBACK = ("c0_today", "i1_hidden", "i2_recipe", "i3_gate")
GATE_RETRIES = 2

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


def _contract_signature(target: Target) -> str:
    """The contract without its example kernel: the example is the textbook method itself."""
    import importlib
    mod = importlib.import_module(f"crazyai.toolkit.measure.{target.measure_families[0]}")
    return ("\nTHE FIXED CONTRACT (do not guess it, do not change the argument order):\n    " + mod.CONTRACT +
            "\nCompiled with: gcc -O3 -march=native -fopenmp -lm. You may use OpenMP, immintrin.h and scratch memory.\n")


def bend_prompt_hidden(ideas: str, target: Target, tools: list[str], focus: str = "") -> str:
    """bend_prompt with the textbook hidden: no known way, no assumption list, no example kernel."""
    step2 = ("2. Pick the seed whose mapping is most literal" +
             (f", preferring one that breaks this assumption if any of the three do: \"{focus}\"" if focus else "") + ".\n")
    return (
        "=== WHAT THE NATIVE SAID ===\n" + ideas.strip() + "\n=== END ===\n\n"
        f"TARGET PROBLEM: {target.problem}\n" + _contract_signature(target) + "\n"
        "Steps:\n"
        "1. For each SEED, write the mapping world-object -> problem-object as a table.\n" + step2 +
        f"3. {target.bend_instructions}\n"
        "4. If your own VERDICT names a specific condition where your mechanism could be slow, guard it with a size or "
        "condition check, or drop the risky part - never ship a mechanism whose own stated risk you don't address.\n"
        f"Tools available: {', '.join(tools)}.\n"
        "Write the final answer with sections: MAPPING, CHOSEN SEED, ASSUMPTION BROKEN, ARTIFACT, PREDICTION, MEASUREMENT, VERDICT."
    )


RECIPE_ASK = (
    "Look again at what you told us above. Choose the ONE of your SEED practices that best meets the need, and write it "
    "as a numbered recipe of 5 to 12 steps, precise enough that a stranger could follow it exactly without you: what is "
    "laid out first, what is compared or combined with what, in what order, what is kept, what is thrown away, what is "
    "done many times over, and how you know you are finished. Use only the things of your world. Begin with the line "
    "'CHOSEN SEED: <the seed>' and then the steps, nothing else."
)


def recipe_native_prompt(world: str, ideas: str, target: Target) -> str:
    return ("=== YOUR WORLD ===\n" + world.strip() + "\n=== END ===\n\n"
            f"A need came to you: {target.in_world_need}\n\n=== WHAT YOU SAID ===\n" + ideas.strip() +
            "\n=== END ===\n\n" + RECIPE_ASK)


def recipe_engineer_prompt(recipe: str, target: Target, tools: list[str]) -> str:
    return (
        "=== A RECIPE FROM A NATIVE OF ANOTHER WORLD ===\n" + recipe.strip() + "\n=== END ===\n\n"
        f"TARGET PROBLEM: {target.problem}\n" + _contract_signature(target) + "\n"
        "Implement THIS recipe, step by step, as the kernel:\n"
        "1. DICTIONARY: a table mapping every thing in the recipe onto a concrete computational thing.\n"
        "2. Write the kernel so that each numbered recipe step becomes one commented block `/* step k: ... */`, in the "
        "recipe's order. Do not add a step the recipe doesn't have, and do not replace any step with a different method. "
        "You choose only data layout and how each step is carried out at the machine level.\n"
        "3. If a step is ambiguous, pick the most literal reading and say so. If following the recipe exactly would give "
        "a wrong answer, say which step, and make the smallest change to that step that makes it correct.\n"
        f"4. {target.bend_instructions}\n"
        f"Tools available: {', '.join(tools)}.\n"
        "Write the final answer with sections: DICTIONARY, ARTIFACT, PREDICTION, MEASUREMENT, VERDICT."
    )


def gate_feedback(judge: dict) -> str:
    mechs = judge.get("native_mechanisms") or []
    listing = "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(mechs))
    return ("\n\n=== REVIEW OF YOUR PREVIOUS ATTEMPT ===\n"
            f"A reviewer read your previous kernel and found that its core is: {judge.get('core_technique', 'a textbook method')}. "
            "That is a standard textbook approach, not the native's mechanism. The native's mechanisms were:\n" + listing +
            "\nBuild the kernel again so that its core IS one of these mechanisms, translated literally. Keep it correct; "
            "make it fast by improving how you implement that mechanism, not by replacing it.\n=== END REVIEW ===")


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
        if self.variant != "orig" and self.variant not in VARIANTS and self.variant not in STRATEGIES:
            raise ValueError(f"unknown variant {self.variant!r}; choose from {', '.join([*VARIANTS, *STRATEGIES])} or orig")
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

    def tool_names(self) -> list[str]:
        return self.toolkit.names(families=self.tgt.measure_families + ["unconventional", "symbolic"])

    def build_prompt(self) -> str:
        names = self.tool_names()
        if self.variant in ("i1_hidden", "i2_recipe"):
            return bend_prompt_hidden(self.ideas(), self.tgt, names, self.focus())
        base = P.bend_prompt(self.ideas(), self.tgt, names, self.focus())
        return VARIANTS[self.variant](base)

    def _recipe_bend(self, provider: Provider) -> tuple[str, str]:
        if self._have("recipe.md"):
            recipe = (self.dir / "recipe.md").read_text(encoding="utf-8")
        else:
            world = (self.src / "world.md").read_text(encoding="utf-8")
            recipe = provider.agent(P.IMMERSE_SYSTEM, recipe_native_prompt(world, self.ideas(), self.tgt), self.toolkit, []).text
            (self.dir / "recipe.md").write_text(recipe, encoding="utf-8")
        prompt = recipe_engineer_prompt(recipe, self.tgt, self.tool_names())
        return provider.agent(P.BEND_SYSTEM, prompt, self.toolkit, self.tool_names()).text, prompt

    def _gated_bend(self, provider: Provider) -> tuple[str, str]:
        """rp0's prompt; judge each attempt; on a fallback, feed the verdict back and retry. Keeps the first
        non-fallback attempt, else the first attempt (so a gate that never succeeds equals the plain prompt)."""
        prompt = self.build_prompt()
        attempts = []
        for k in range(GATE_RETRIES + 1):
            path = self.dir / f"attempt_{k}.md"
            if self._have(path.name):
                art = path.read_text(encoding="utf-8")
            else:
                ask = prompt + (gate_feedback(attempts[-1][1]) if attempts else "")
                art = provider.agent(P.BEND_SYSTEM, ask, self.toolkit, self.tool_names()).text
                path.write_text(art, encoding="utf-8")
            code = _CODE.findall(art)
            gpath = self.dir / f"gate_{k}.json"
            if self._have(gpath.name):
                verdict = _load(gpath)
            elif code:
                verdict = provider.structured(JUDGE_SYSTEM, judge_prompt(self.ideas(), self.tgt, code[-1]), JUDGE_SCHEMA)
                _dump(gpath, verdict)
            else:
                verdict = {"fallback": True, "core_technique": "no code was produced", "native_mechanisms": []}
                _dump(gpath, verdict)
            attempts.append((art, verdict))
            if verdict.get("fallback") is False:
                break
        chosen = next((i for i, (_, v) in enumerate(attempts) if v.get("fallback") is False), 0)
        _dump(self.dir / "gate.json", {"attempts": len(attempts), "chosen": chosen,
                                       "fallback_per_attempt": [v.get("fallback") for _, v in attempts]})
        return attempts[chosen][0], prompt

    def step_bend(self, provider: Provider) -> dict[str, Any]:
        if self.variant == "orig":
            art = (self.src / "artifact.md").read_text(encoding="utf-8")
        elif self._have("artifact.md"):
            art = (self.dir / "artifact.md").read_text(encoding="utf-8")
        else:
            if self.variant == "i2_recipe":
                art, prompt = self._recipe_bend(provider)
            elif self.variant == "i3_gate":
                art, prompt = self._gated_bend(provider)
            else:
                prompt = self.build_prompt()
                art = provider.agent(P.BEND_SYSTEM, prompt, self.toolkit, self.tool_names()).text
            (self.dir / "artifact.md").write_text(art, encoding="utf-8")
            _dump(self.dir / "bend_calls.json", {"model": getattr(provider, "model", ""), "prompt_hash": prompt_hash(prompt)})
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
