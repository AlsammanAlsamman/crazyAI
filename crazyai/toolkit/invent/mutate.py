"""mutate.* - formal rule-breaking.

The rule's `form` is a small structured statement (universal / implication /
equation / definition). Each operator is a named transformation on that
structure; the result is a new formal statement plus a one-line instruction
telling the generator what has changed. Nothing here is random: chaos decides
*which* operator, mutate decides *what it does*.
"""

from __future__ import annotations

import re
from typing import Any

import sympy as sp

from crazyai.config import OPERATORS
from crazyai.domains import get_rule, load_domains
from crazyai.toolkit.registry import tool

_QUANT_SWAPS = [
    ("every", "some"), ("all", "some"), ("never", "sometimes"), ("always", "sometimes"),
    ("no ", "some "), ("exactly one", "at least two"), ("only", "also"), ("cannot", "can"),
    ("infinitely many", "finitely many"), ("finite", "infinite"),
]


def _swap_words(text: str, pairs: list[tuple[str, str]]) -> tuple[str, list[str]]:
    applied = []
    for a, b in pairs:
        pat = re.compile(rf"\b{re.escape(a.strip())}\b", re.IGNORECASE)
        if pat.search(text):
            text = pat.sub(b.strip(), text, count=1)
            applied.append(f"{a.strip()} -> {b.strip()}")
            break
    return text, applied


def _invert(form: dict[str, Any], statement: str) -> tuple[dict[str, Any], str]:
    k = form.get("kind")
    if k == "implication":
        new = {"kind": "implication", "antecedent": form["consequent"], "consequent": form["antecedent"]}
        return new, f"If {form['consequent']}, then {form['antecedent']}."
    if k == "equation":
        lhs, rhs = form["lhs"], form["rhs"]
        try:
            e = sp.sympify(rhs)
            inv = sp.simplify(1 / e) if e != 0 else e
            new = {"kind": "equation", "lhs": lhs, "rhs": str(inv)}
            return new, f"{lhs} = {inv}"
        except Exception:
            return {"kind": "equation", "lhs": rhs, "rhs": lhs}, f"{rhs} = {lhs}"
    if k == "universal":
        pred = form["predicate"]
        neg, _ = _swap_words(pred, [("never", "always"), ("is constant", "is never constant"),
                                    ("is at most", "exceeds"), ("reaches", "never reaches"),
                                    ("converges", "diverges"), ("is", "is not")])
        new = {"kind": "universal", "subject": form["subject"], "predicate": neg}
        return new, f"For every {form['subject']}: {neg}."
    if k == "definition":
        new = {"kind": "definition", "term": form["term"], "definiens": f"anything that is NOT ({form['definiens']})"}
        return new, f"'{form['term']}' now means: anything that is not ({form['definiens']})."
    return form, statement


def _remove(form: dict[str, Any], statement: str) -> tuple[dict[str, Any], str]:
    k = form.get("kind")
    if k == "implication":
        return {"kind": "universal", "subject": "case", "predicate": form["consequent"]}, \
            f"{form['consequent'].capitalize()} - unconditionally; the condition '{form['antecedent']}' no longer applies."
    if k == "universal":
        return {"kind": "existential", "subject": form["subject"], "predicate": form["predicate"]}, \
            f"The requirement is dropped: no {form['subject']} needs to satisfy '{form['predicate']}'."
    if k == "equation":
        return {"kind": "inequality", "lhs": form["lhs"], "rhs": form["rhs"]}, \
            f"{form['lhs']} is no longer tied to {form['rhs']}; the equation becomes a mere tendency."
    if k == "definition":
        return {"kind": "definition", "term": form["term"], "definiens": "unrestricted"}, \
            f"'{form['term']}' loses its defining condition and may apply to anything."
    return form, statement


def _extrapolate(form: dict[str, Any], statement: str) -> tuple[dict[str, Any], str]:
    k = form.get("kind")
    if k == "universal":
        subj = form["subject"]
        return {"kind": "universal", "subject": f"anything, not only a {subj}", "predicate": form["predicate"]}, \
            f"For anything whatsoever (not only a {subj}): {form['predicate']}."
    if k == "implication":
        return {"kind": "implication", "antecedent": f"{form['antecedent']} in any regime, however extreme",
                "consequent": form["consequent"]}, \
            f"If {form['antecedent']} - in any regime, however extreme - then {form['consequent']}."
    if k == "equation":
        return {"kind": "equation", "lhs": form["lhs"], "rhs": form["rhs"], "domain": "all values, including limits where the derivation fails"}, \
            f"{form['lhs']} = {form['rhs']} holds for all values, including limits where the derivation fails."
    if k == "definition":
        return {"kind": "definition", "term": form["term"], "definiens": form["definiens"] + ", extended to non-integers / non-standard objects"}, \
            f"'{form['term']}' is extended beyond its original objects."
    return form, statement


def _transpose(form: dict[str, Any], statement: str, target_domain: str) -> tuple[dict[str, Any], str]:
    d = load_domains().get(target_domain)
    gloss = d.glossary if d else []
    new = {"kind": form.get("kind", "statement"), "transposed_to": target_domain, "source_form": form}
    hint = f"Restate the rule so that its objects are those of {target_domain} (use terms such as {', '.join(gloss[:6])})."
    return new, f"{statement} - transposed into {target_domain}. {hint}"


def _compose(form: dict[str, Any], statement: str, other_statement: str) -> tuple[dict[str, Any], str]:
    new = {"kind": "composition", "parts": [form, other_statement]}
    return new, f"Both hold and are combined into one rule: ({statement}) AND ({other_statement}), treated as jointly implying a new consequence."


def _quantify(form: dict[str, Any], statement: str) -> tuple[dict[str, Any], str]:
    new_stmt, applied = _swap_words(statement, _QUANT_SWAPS)
    if not applied:
        new_stmt = "In some cases, " + statement[0].lower() + statement[1:]
        applied = ["(implicit universal) -> some"]
    return {**form, "quantifier_change": applied}, new_stmt


def _substitute(form: dict[str, Any], statement: str, glossary: list[str]) -> tuple[dict[str, Any], str]:
    words = statement.split()
    for i, w in enumerate(words):
        bare = re.sub(r"[^a-zA-Z]", "", w).lower()
        if bare in glossary:
            alt = next((g for g in glossary if g != bare), None)
            if alt:
                words[i] = w.replace(bare, alt) if bare in w else alt
                return {**form, "substitution": [bare, alt]}, " ".join(words)
    return {**form, "substitution": None}, statement + " (with one key term redefined by a near-synonym)"


@tool("mutate", "invent")
def apply_operator(rule_id: str, operator: str, target_domain: str = "", other_rule_id: str = "") -> dict:
    """Apply a named mutation operator to a curated rule and return the mutated formal statement.

    Args:
        rule_id: Id of the rule from chaos_draw_seed (e.g. phys.energy.conserved).
        operator: One of INVERT, REMOVE, EXTRAPOLATE, TRANSPOSE, COMPOSE, QUANTIFY, SUBSTITUTE.
        target_domain: For TRANSPOSE - the domain to move the rule into.
        other_rule_id: For COMPOSE - the second rule to combine with.
    """
    operator = operator.upper()
    if operator not in OPERATORS:
        return {"error": f"unknown operator {operator!r}; known: {OPERATORS}"}
    rule = get_rule(rule_id)
    form, stmt = rule.form, rule.statement
    if operator == "INVERT":
        new_form, new_stmt = _invert(form, stmt)
    elif operator == "REMOVE":
        new_form, new_stmt = _remove(form, stmt)
    elif operator == "EXTRAPOLATE":
        new_form, new_stmt = _extrapolate(form, stmt)
    elif operator == "TRANSPOSE":
        if not target_domain:
            return {"error": "TRANSPOSE needs target_domain"}
        new_form, new_stmt = _transpose(form, stmt, target_domain)
    elif operator == "COMPOSE":
        if not other_rule_id:
            return {"error": "COMPOSE needs other_rule_id"}
        new_form, new_stmt = _compose(form, stmt, get_rule(other_rule_id).statement)
    elif operator == "QUANTIFY":
        new_form, new_stmt = _quantify(form, stmt)
    else:
        new_form, new_stmt = _substitute(form, stmt, load_domains()[rule.domain].glossary)
    return {
        "rule_id": rule.id, "operator": operator,
        "original": {"statement": stmt, "form": form},
        "mutated": {"statement": new_stmt, "form": new_form},
        "instruction": (
            "Treat the mutated statement as a true law of the world you are building. "
            "Do not state it directly in the artifact; let its consequences drive everything. "
            "Everything else in the domain stays exactly as it is."
        ),
    }


@tool("mutate", "invent")
def list_operators() -> dict:
    """List the mutation operators and what each one does."""
    return {"operators": {
        "INVERT": "reverse a relation (cause <-> effect, lhs <-> 1/rhs, is <-> is not)",
        "REMOVE": "delete one condition or constraint and keep everything else",
        "EXTRAPOLATE": "push a valid relation beyond the domain where it was proven",
        "TRANSPOSE": "move a structure into a field where it has no known meaning",
        "COMPOSE": "combine two true statements whose combination is not known to be valid",
        "QUANTIFY": "change a quantifier (some <-> all, never <-> sometimes, finite <-> infinite)",
        "SUBSTITUTE": "replace a term with a near-synonym that has a different formal meaning",
    }}
