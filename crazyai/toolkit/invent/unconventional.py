"""unconventional.* - thinking off the path.

These tools do not judge anything. They enumerate what is usually taken for
granted, turn it around, push it to extremes, and carry a rule change to its
consequences along the curated dependency graph.
"""

from __future__ import annotations

import re

import sympy as sp

from crazyai.domains import dependents, get_rule, load_domains, rule_index
from crazyai.toolkit.registry import tool

_GENERIC_ASSUMPTIONS = {
    "physics": [
        "the system is isolated from its surroundings",
        "quantities are continuous and differentiable",
        "the observer's frame is inertial",
        "energy, momentum and charge are separately conserved",
        "the same law holds at every scale",
    ],
    "number_theory": [
        "the objects are ordinary integers",
        "divisibility is the usual relation",
        "equality is transitive",
        "there is exactly one factorisation",
        "'prime' is a property of a number, not of a context",
    ],
    "statistics": [
        "observations are independent",
        "the sample is representative of the population",
        "all relevant variables are measured",
        "the model is correctly specified",
        "the data-generating process is stable over time",
    ],
    "narrative": [
        "characters know only what they have witnessed or been told",
        "time runs in one direction",
        "a character occupies one place at a time",
        "actions follow from wants and beliefs",
        "the narrator reports events reliably",
    ],
    "computation": [
        "inputs are finite and given up front",
        "noise in the data is irreducible",
        "future states depend on present states only through known laws",
        "computation time is bounded",
        "training data contains the signal to be predicted",
    ],
}


@tool("unconventional", "invent")
def enumerate_assumptions(rule_id: str) -> dict:
    """List what a curated rule silently assumes: its declared dependencies plus the standard assumptions of its domain.

    Args:
        rule_id: Rule id from chaos_draw_seed.
    """
    r = get_rule(rule_id)
    deps = [rule_index()[d].statement for d in r.depends_on if d in rule_index()]
    return {
        "rule": r.statement,
        "explicit_dependencies": deps,
        "domain_assumptions": _GENERIC_ASSUMPTIONS.get(r.domain, []),
        "hint": "Each of these is a candidate to invert; the artifact must keep all of them except the mutated rule.",
    }


_INVERSIONS = [
    (r"\bis isolated\b", "exchanges freely with its surroundings"),
    (r"\bindependent\b", "dependent"),
    (r"\bcontinuous\b", "discrete"),
    (r"\bone direction\b", "both directions"),
    (r"\bone place\b", "several places"),
    (r"\bfinite\b", "infinite"),
    (r"\bexactly one\b", "many"),
    (r"\bbounded\b", "unbounded"),
    (r"\breliably\b", "unreliably"),
    (r"\bonly\b", "also"),
    (r"\bcontains\b", "lacks"),
    (r"\bis\b", "is not"),
]


@tool("unconventional", "invent")
def invert(assumption: str) -> dict:
    """Turn an assumption into its opposite, as a candidate new world-rule.

    Args:
        assumption: A short assumption sentence, e.g. 'the system is isolated from its surroundings'.
    """
    for pat, rep in _INVERSIONS:
        if re.search(pat, assumption):
            return {"assumption": assumption, "inverted": re.sub(pat, rep, assumption, count=1)}
    return {"assumption": assumption, "inverted": f"it is not the case that {assumption}"}


@tool("unconventional", "invent")
def extreme_case(expression: str, parameter: str) -> dict:
    """Push a parameter of a formula to 0, +infinity and -infinity and report what the expression does.

    Args:
        expression: A SymPy-parsable expression, e.g. 'm*c**2/sqrt(1 - v**2/c**2)'.
        parameter: The symbol to push, e.g. 'v'.
    """
    try:
        e = sp.sympify(expression)
        p = sp.Symbol(parameter)
        out = {}
        for label, target in (("to_zero", 0), ("to_plus_inf", sp.oo), ("to_minus_inf", -sp.oo)):
            try:
                out[label] = str(sp.limit(e, p, target))
            except Exception as exc:
                out[label] = f"undefined ({type(exc).__name__})"
        return {"expression": str(e), "parameter": parameter, "limits": out}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@tool("unconventional", "invent")
def transpose(from_domain: str, to_domain: str, structure: str) -> dict:
    """Propose a term-by-term mapping for carrying a structure from one domain into another.

    Args:
        from_domain: Source domain name.
        to_domain: Target domain name.
        structure: A sentence describing the structure to move, using source-domain terms.
    """
    ds = load_domains()
    if from_domain not in ds or to_domain not in ds:
        return {"error": f"unknown domain; known: {sorted(ds)}"}
    src, dst = ds[from_domain].glossary, ds[to_domain].glossary
    mapping = {s: dst[i % len(dst)] for i, s in enumerate(src)}
    text = structure
    for s, d in mapping.items():
        text = re.sub(rf"\b{re.escape(s)}\b", d.upper(), text)
    return {"mapping": mapping, "transposed_draft": text,
            "hint": "Upper-case words are transposed terms; rewrite the sentence so they read naturally, keeping the structure."}


@tool("unconventional", "invent")
def what_if(rule_id: str, depth: int = 3) -> dict:
    """Walk the dependency graph: which curated rules would be affected, and how far away, if this rule changed.

    Args:
        rule_id: The rule being mutated.
        depth: Maximum graph distance to follow.
    """
    idx = rule_index()
    chain = [{"rule_id": rid, "distance": d, "statement": idx[rid].statement, "weight": idx[rid].weight}
             for rid, d in dependents(rule_id, depth)]
    return {"rule_id": rule_id, "affected": chain, "count": len(chain)}
