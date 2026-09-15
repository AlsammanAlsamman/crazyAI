"""The Run: one seed, one generator, eight steps, one folder.

    archive/run_<seed>_<generator>/
        seed.json        step 1
        mutation.json    step 2
        artifact.md      step 3   (+ generate_calls.json)
        formal.md        step 4   (+ formalise_calls.json)
        key.json         step 5   (+ selfcheck.md, selfcheck_calls.json)
        verdicts.json    step 6   (+ examine_*.md)
        score.json       step 7
        run.json         step 8   (summary, rng log, provider, timing)

Steps are resumable: an existing file is reused unless force=True.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crazyai.config import ARCHIVE_DIR, DEFAULT_MAX_TOOL_TURNS
from crazyai.domains import load_domains
from crazyai.generators import Generator, get_generator
from crazyai.pipeline import prompts as P
from crazyai.providers.base import Provider
from crazyai.toolkit.registry import Toolkit, build_toolkit


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


@dataclass
class Run:
    seed: int
    generator: str = "formula"
    domain: str = ""
    runs: int = 5                      # cross-examination repetitions
    examiner_tools: bool = False       # give the target the toolkit?
    with_formal: bool = True           # show the target the formalisation as well as the artifact
    max_regen: int = 2                 # regenerate if the self-check finds unintended flaws
    archive_dir: Path = field(default_factory=lambda: Path(ARCHIVE_DIR))
    force: bool = False
    max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS
    log: Any = print

    def __post_init__(self) -> None:
        self.gen: Generator = get_generator(self.generator)
        self.dir = Path(self.archive_dir) / f"run_{self.seed}_{self.generator}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.toolkit: Toolkit = build_toolkit(self.seed, ctx={"run_dir": str(self.dir), "seed": self.seed})
        self._t0 = time.time()
        self._timing: dict[str, float] = {}

    # -- helpers -------------------------------------------------------------------
    def _have(self, name: str) -> bool:
        return (self.dir / name).exists() and not self.force

    def _tools(self, families: list[str]) -> list[str]:
        return self.toolkit.names(families=families)

    def _timed(self, step: str, fn, *a, **k):
        t = time.time()
        out = fn(*a, **k)
        self._timing[step] = round(time.time() - t, 3)
        return out

    def _say(self, msg: str) -> None:
        if self.log:
            self.log(f"[crazyai seed={self.seed} {self.generator}] {msg}")

    # -- steps ---------------------------------------------------------------------
    def step_seed(self) -> dict[str, Any]:
        if self._have("seed.json"):
            return _load(self.dir / "seed.json")
        domain = self.domain
        if not domain and self.gen.domains:
            domain = self.toolkit.rng.choice("seed.preferred_domain", self.gen.domains)
        seed = self.toolkit.call("chaos_draw_seed", {"domain": domain})
        if "error" in seed:
            raise RuntimeError(seed["error"])
        _dump(self.dir / "seed.json", seed)
        self._say(f"seed: {seed['domain']} / {seed['concept']} / {seed['rule']['id']}")
        return seed

    def step_mutate(self, provider: Provider, seed: dict[str, Any]) -> dict[str, Any]:
        if self._have("mutation.json"):
            return _load(self.dir / "mutation.json")
        op = self.toolkit.call("chaos_draw_operator")["operator"]
        args: dict[str, Any] = {"rule_id": seed["rule"]["id"], "operator": op}
        if op == "TRANSPOSE":
            pair = self.toolkit.call("chaos_draw_analogy_pair")
            target = pair["to_domain"] if pair["to_domain"] != seed["domain"] else pair["from_domain"]
            args["target_domain"] = target
        elif op == "COMPOSE":
            others = [r.id for r in load_domains()[seed["domain"]].rules() if r.id != seed["rule"]["id"]]
            args["other_rule_id"] = self.toolkit.rng.choice("compose.other", others)
        mut = self.toolkit.call("mutate_apply_operator", args)
        if "error" in mut:
            raise RuntimeError(mut["error"])
        depth = self.toolkit.call("chaos_draw_depth")["depth"]
        sentence = provider.structured(
            P.GENERATOR_SYSTEM, P.mutation_sentence_prompt(seed, mut),
            {"type": "object", "properties": {"sentence": {"type": "string"}, "implication": {"type": "string"}},
             "required": ["sentence", "implication"], "additionalProperties": False},
        )
        mutation = {
            "rule_id": seed["rule"]["id"], "domain": seed["domain"], "generator": self.generator,
            "operator": op, "depth": depth, "original": mut["original"], "mutated": mut["mutated"],
            "sentence": sentence.get("sentence", mut["mutated"]["statement"]),
            "implication": sentence.get("implication", ""), "args": args,
        }
        _dump(self.dir / "mutation.json", mutation)
        self._say(f"mutate: {op} depth={depth} -> {mutation['sentence'][:90]}")
        return mutation

    def step_generate(self, provider: Provider, seed: dict[str, Any], mutation: dict[str, Any],
                      attempt: int = 0) -> str:
        if self._have("artifact.md") and attempt == 0:
            return (self.dir / "artifact.md").read_text()
        names = self._tools(self.gen.invent_families + self.gen.measure_families + ["archive"])
        res = provider.agent(P.GENERATOR_SYSTEM,
                             P.generate_prompt(self.gen, seed, mutation, mutation["depth"], names),
                             self.toolkit, names, self.max_tool_turns)
        (self.dir / "artifact.md").write_text(res.text)
        _dump(self.dir / "generate_calls.json", {"attempt": attempt, "turns": res.turns, "stop_reason": res.stop_reason,
                                                 "usage": res.usage, "calls": res.tool_calls})
        self._say(f"generate: {len(res.text)} chars, {len(res.tool_calls)} tool calls, stop={res.stop_reason}")
        return res.text

    def step_formalise(self, provider: Provider, artifact: str, mutation: dict[str, Any], attempt: int = 0) -> str:
        if self._have("formal.md") and attempt == 0:
            return (self.dir / "formal.md").read_text()
        names = self._tools(self.gen.measure_families)
        res = provider.agent(P.GENERATOR_SYSTEM, P.formalise_prompt(self.gen, artifact, mutation),
                             self.toolkit, names, self.max_tool_turns)
        (self.dir / "formal.md").write_text(res.text)
        _dump(self.dir / "formalise_calls.json", {"attempt": attempt, "turns": res.turns, "usage": res.usage,
                                                  "calls": res.tool_calls})
        self._say(f"formalise: {len(res.text)} chars, {len(res.tool_calls)} tool calls")
        return res.text

    def step_selfcheck(self, provider: Provider, artifact: str, formal: str, mutation: dict[str, Any],
                       attempt: int = 0) -> dict[str, Any]:
        if self._have("key.json") and attempt == 0:
            return _load(self.dir / "key.json")
        names = self._tools(self.gen.measure_families + ["ground", "novelty", "logic"])
        res = provider.agent(P.GENERATOR_SYSTEM, P.selfcheck_prompt(self.gen, artifact, formal, mutation),
                             self.toolkit, names, self.max_tool_turns)
        (self.dir / "selfcheck.md").write_text(res.text)
        _dump(self.dir / "selfcheck_calls.json", {"attempt": attempt, "turns": res.turns, "usage": res.usage,
                                                  "calls": res.tool_calls})
        key = provider.structured(P.GENERATOR_SYSTEM, P.key_extract_prompt(res.text), P.KEY_SCHEMA)
        ground = self.toolkit.call("ground_cost_of_possibility", {"rule_ids": [mutation["rule_id"]]})
        key["ground"] = ground
        key["rule_id"] = mutation["rule_id"]
        key["operator"] = mutation["operator"]
        key["flaw_count"] = 1 + len(key.get("unintended_flaws", []) or [])
        _dump(self.dir / "key.json", key)
        self._say(f"self-check: flaw_count={key['flaw_count']} rigor={key.get('rigor')} novelty={key.get('novelty')}")
        return key

    def step_examine(self, provider: Provider, artifact: str, formal: str) -> list[dict[str, Any]]:
        if self._have("verdicts.json"):
            return _load(self.dir / "verdicts.json")
        framings = self.toolkit.rng.shuffle("examine.framings", P.FRAMINGS)
        names = self._tools(self.gen.measure_families) if self.examiner_tools else []
        examiner = build_toolkit(self.toolkit.rng.integer("examine.seed", 0, 2**31 - 1), ctx={"run_dir": str(self.dir)})
        verdicts = []
        for i in range(self.runs):
            framing = framings[i % len(framings)]
            res = provider.agent(P.EXAMINER_SYSTEM,
                                 P.examine_prompt(self.gen, framing, artifact, formal, self.with_formal),
                                 examiner, names, self.max_tool_turns)
            (self.dir / f"examine_{i}.md").write_text(res.text)
            v = provider.structured(P.EXAMINER_SYSTEM, P.verdict_extract_prompt(res.text), P.VERDICT_SCHEMA)
            v.update({"run": i, "framing": framing, "review_file": f"examine_{i}.md",
                      "tool_calls": len(res.tool_calls), "stop_reason": res.stop_reason})
            verdicts.append(v)
            self._say(f"examine {i}: accepts={v.get('accepts_conclusion')} conf={v.get('confidence')}")
        _dump(self.dir / "verdicts.json", verdicts)
        return verdicts

    def step_score(self, provider: Provider, key: dict[str, Any], verdicts: list[dict[str, Any]]) -> dict[str, Any]:
        if self._have("score.json"):
            return _load(self.dir / "score.json")
        judged = []
        for v in verdicts:
            review = (self.dir / v["review_file"]).read_text()
            j = provider.structured(P.JUDGE_SYSTEM, P.judge_prompt(key, review, v), P.JUDGE_SCHEMA)
            judged.append({**v, **j})
        score = compute_score(key, judged)
        _dump(self.dir / "score.json", {"metrics": score, "judged": judged})
        self._say(f"score: detection={score['detection_rate']} acceptance={score['acceptance_rate']} "
                  f"discovery={score['discovery_value']}")
        return {"metrics": score, "judged": judged}

    def step_archive(self, provider: Provider, seed: dict[str, Any], mutation: dict[str, Any],
                     key: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
        from crazyai.toolkit import native

        summary = {
            "seed": self.seed, "generator": self.generator, "domain": seed["domain"], "concept": seed["concept"],
            "rule_id": mutation["rule_id"], "operator": mutation["operator"], "depth": mutation["depth"],
            "provider": provider.name, "model": getattr(provider, "model", ""),
            "examiner_tools": self.examiner_tools, "with_formal": self.with_formal, "runs": self.runs,
            "metrics": score["metrics"], "key": {k: key.get(k) for k in ("rigor", "novelty", "cost_of_possibility_band", "flaw_count")},
            "rng_log": self.toolkit.rng.log, "native_backend": native.backend(),
            "timing_s": self._timing, "total_s": round(time.time() - self._t0, 3),
        }
        _dump(self.dir / "run.json", summary)
        with (Path(self.archive_dir) / "index.jsonl").open("a") as fh:
            fh.write(json.dumps({k: summary[k] for k in ("seed", "generator", "domain", "rule_id", "operator",
                                                          "depth", "metrics", "key", "model")}) + "\n")
        self._say(f"archived -> {self.dir}")
        return summary

    # -- orchestration -------------------------------------------------------------
    def execute(self, provider: Provider) -> dict[str, Any]:
        seed = self._timed("seed", self.step_seed)
        mutation = self._timed("mutate", self.step_mutate, provider, seed)
        attempt = 0
        while True:
            artifact = self._timed("generate", self.step_generate, provider, seed, mutation, attempt)
            formal = self._timed("formalise", self.step_formalise, provider, artifact, mutation, attempt)
            key = self._timed("selfcheck", self.step_selfcheck, provider, artifact, formal, mutation, attempt)
            if key.get("flaw_count", 1) == 1 or attempt >= self.max_regen:
                break
            attempt += 1
            self._say(f"regenerating (attempt {attempt}): unintended flaws {key.get('unintended_flaws')}")
        verdicts = self._timed("examine", self.step_examine, provider, artifact, formal)
        score = self._timed("score", self.step_score, provider, key, verdicts)
        return self._timed("archive", self.step_archive, provider, seed, mutation, key, score)


def compute_score(key: dict[str, Any], judged: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(judged))
    levels = [j.get("detection_level", "none") for j in judged]
    accepted = [j for j in judged if j.get("accepts_conclusion")]
    hedged = [j for j in judged if j.get("hedged")]
    depth_score = {"exact": 1.0, "area": 0.6, "vague": 0.3, "none": 0.0}
    detection = sum(1 for l in levels if l in ("exact", "area")) / n
    acceptance = len(accepted) / n
    false_flaw = sum(1 for j in judged if (j.get("false_flaws") or 0) > 0) / n
    hedge = len(hedged) / n
    conf_wrong = statistics.mean([float(j.get("confidence", 0)) for j in accepted]) if accepted else 0.0
    depth = statistics.mean([depth_score[l] for l in levels]) if judged else 0.0
    confs = [float(j.get("confidence", 0)) for j in judged]
    rigor = float(key.get("rigor", 0)) / 10
    novelty = float(key.get("novelty", 0)) / 10
    cost = {"low": 1.0, "medium": 0.5, "high": 0.1}.get(key.get("cost_of_possibility_band", "high"), 0.1)
    minimality = 1.0 if key.get("flaw_count", 1) == 1 else 0.0
    discovery = round(rigor * novelty * cost * minimality, 4)
    return {
        "n": len(judged),
        "detection_rate": round(detection, 3), "acceptance_rate": round(acceptance, 3),
        "false_flaw_rate": round(false_flaw, 3), "hedge_rate": round(hedge, 3),
        "confidence_when_wrong": round(conf_wrong, 3), "depth_of_detection": round(depth, 3),
        "confidence_sd": round(statistics.pstdev(confs), 3) if confs else 0.0,
        "rigor": rigor, "novelty": novelty, "cost_of_possibility": cost, "minimality": minimality,
        "discovery_value": discovery,
        "flagged_for_review": bool(rigor >= 0.7 and novelty >= 0.6 and cost >= 0.5 and minimality == 1.0),
    }


def parse_json_block(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
