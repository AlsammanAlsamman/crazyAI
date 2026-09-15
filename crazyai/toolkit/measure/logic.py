"""logic.* - consistency of arguments.

Propositional formulas use: variables (A, B, rain, ...), ~ (not), & (and),
| (or), -> (implies), <-> (iff), parentheses. Consistency is decided by
exhaustive truth-table search for up to 18 variables.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

from crazyai.toolkit.registry import tool

_TOKEN = re.compile(r"\s*(<->|->|[()~&|]|[A-Za-z_][A-Za-z0-9_]*)")


class _Parser:
    def __init__(self, text: str):
        self.toks = _TOKEN.findall(text)
        if "".join(self.toks) != re.sub(r"\s+", "", text):
            raise ValueError(f"bad characters in formula: {text!r}")
        self.i = 0
        self.vars: set[str] = set()

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def eat(self, t=None):
        tok = self.peek()
        if t is not None and tok != t:
            raise ValueError(f"expected {t!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self):
        node = self.iff()
        if self.peek() is not None:
            raise ValueError(f"unexpected token {self.peek()!r}")
        return node

    def iff(self):
        left = self.imp()
        while self.peek() == "<->":
            self.eat()
            left = ("iff", left, self.imp())
        return left

    def imp(self):
        left = self.disj()
        if self.peek() == "->":
            self.eat()
            return ("imp", left, self.imp())
        return left

    def disj(self):
        left = self.conj()
        while self.peek() == "|":
            self.eat()
            left = ("or", left, self.conj())
        return left

    def conj(self):
        left = self.unary()
        while self.peek() == "&":
            self.eat()
            left = ("and", left, self.unary())
        return left

    def unary(self):
        t = self.peek()
        if t == "~":
            self.eat()
            return ("not", self.unary())
        if t == "(":
            self.eat()
            n = self.iff()
            self.eat(")")
            return n
        if t is None or t in ("&", "|", "->", "<->", ")"):
            raise ValueError(f"unexpected token {t!r}")
        self.eat()
        self.vars.add(t)
        return ("var", t)


def _eval(node, env: dict[str, bool]) -> bool:
    k = node[0]
    if k == "var":
        return env[node[1]]
    if k == "not":
        return not _eval(node[1], env)
    a, b = _eval(node[1], env), _eval(node[2], env)
    if k == "and":
        return a and b
    if k == "or":
        return a or b
    if k == "imp":
        return (not a) or b
    return a == b


def _parse_all(formulas: list[str]) -> tuple[list[Any], list[str]]:
    trees, vars_ = [], set()
    for f in formulas:
        p = _Parser(f)
        trees.append(p.parse())
        vars_ |= p.vars
    return trees, sorted(vars_)


@tool("logic", "measure")
def check_consistency(formulas: list) -> dict:
    """Decide whether a set of propositional formulas can all be true at once; if not, find a minimal inconsistent subset.

    Args:
        formulas: Formulas such as 'A -> B', '~B', 'A'. Operators: ~ & | -> <->.
    """
    try:
        trees, vars_ = _parse_all(list(formulas))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if len(vars_) > 18:
        return {"error": f"too many variables ({len(vars_)} > 18)"}

    def satisfiable(idx: list[int]) -> dict[str, bool] | None:
        for bits in itertools.product([False, True], repeat=len(vars_)):
            env = dict(zip(vars_, bits))
            if all(_eval(trees[i], env) for i in idx):
                return env
        return None

    model = satisfiable(list(range(len(trees))))
    if model is not None:
        return {"consistent": True, "variables": vars_, "model": model}
    # minimal unsatisfiable core by deletion
    core = list(range(len(trees)))
    for i in list(core):
        trial = [j for j in core if j != i]
        if trial and satisfiable(trial) is None:
            core = trial
    return {"consistent": False, "variables": vars_, "minimal_inconsistent_subset": [formulas[i] for i in core],
            "flaw": "the statements contradict each other"}


@tool("logic", "measure")
def entails(premises: list, conclusion: str) -> dict:
    """Check whether the premises logically entail the conclusion (propositional).

    Args:
        premises: Premise formulas.
        conclusion: The conclusion formula.
    """
    try:
        trees, vars_ = _parse_all(list(premises) + [conclusion])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if len(vars_) > 18:
        return {"error": "too many variables"}
    *prem, concl = trees
    for bits in itertools.product([False, True], repeat=len(vars_)):
        env = dict(zip(vars_, bits))
        if all(_eval(p, env) for p in prem) and not _eval(concl, env):
            return {"entails": False, "counterexample": env}
    return {"entails": True}


_SENT = re.compile(r"(?<=[.!?])\s+")


@tool("logic", "measure")
def extract_propositions(text: str) -> dict:
    """Split prose into candidate propositions and flag connectives (if/then, because, therefore, unless, only if).

    Args:
        text: The prose.
    """
    sents = [s.strip() for s in _SENT.split(text.strip()) if s.strip()]
    out = []
    for i, s in enumerate(sents):
        kinds = []
        low = s.lower()
        if re.search(r"\bif\b.*\bthen\b|\bif\b", low):
            kinds.append("conditional")
        if re.search(r"\bbecause\b|\bsince\b|\btherefore\b|\bhence\b|\bso\b|\bthus\b", low):
            kinds.append("inference")
        if re.search(r"\bunless\b|\bonly if\b|\bexcept\b", low):
            kinds.append("restriction")
        if re.search(r"\b(all|every|always|never|no one|nobody|none)\b", low):
            kinds.append("universal")
        if re.search(r"\b(some|sometimes|at least one|there is|there are)\b", low):
            kinds.append("existential")
        out.append({"index": i, "text": s, "kinds": kinds or ["assertion"]})
    return {"count": len(out), "propositions": out}


@tool("logic", "measure")
def find_equivocation(term: str, text: str, window: int = 6) -> dict:
    """Compare the contexts in which a term is used; low overlap between uses suggests the term shifts meaning.

    Args:
        term: The word or phrase to check.
        text: The text.
        window: Words of context on each side.
    """
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    t = term.lower().split()
    L = len(t)
    ctxs = []
    for i in range(len(words) - L + 1):
        if words[i : i + L] == t:
            ctx = set(words[max(0, i - window) : i] + words[i + L : i + L + window])
            ctxs.append((i, ctx))
    if len(ctxs) < 2:
        return {"term": term, "uses": len(ctxs), "verdict": "fewer than two uses; nothing to compare"}
    sims = []
    for (ia, a), (ib, b) in itertools.combinations(ctxs, 2):
        j = len(a & b) / max(1, len(a | b))
        sims.append({"use_a": ia, "use_b": ib, "jaccard": round(j, 3)})
    mean = sum(s["jaccard"] for s in sims) / len(sims)
    lowest = min(sims, key=lambda s: s["jaccard"])
    return {"term": term, "uses": len(ctxs), "mean_context_overlap": round(mean, 3), "most_divergent_pair": lowest,
            "suspect_equivocation": mean < 0.08}


@tool("logic", "measure")
def trace_argument(chain: list) -> dict:
    """Given an argument as steps [{id, claim, from:[ids]}], find the load-bearing steps: those whose removal disconnects the conclusion from the premises.

    Args:
        chain: Steps in order; a step with an empty 'from' is a premise; the last step is the conclusion.
    """
    if not chain:
        return {"error": "empty chain"}
    ids = [str(s["id"]) for s in chain]
    parents = {str(s["id"]): [str(p) for p in s.get("from", [])] for s in chain}
    premises = [i for i in ids if not parents[i]]
    concl = ids[-1]

    def reachable(removed: str | None) -> bool:
        seen = set()
        stack = [concl]
        while stack:
            cur = stack.pop()
            if cur == removed or cur in seen:
                continue
            seen.add(cur)
            for p in parents.get(cur, []):
                stack.append(p)
        return any(p in seen for p in premises) and concl in seen

    dangling = [i for i in ids for p in parents[i] if p not in parents]
    load_bearing = [i for i in ids if i != concl and reachable(None) and not reachable(i)]
    return {"premises": premises, "conclusion": concl, "load_bearing": load_bearing,
            "references_unknown_steps": dangling, "grounded": reachable(None) and not dangling}
