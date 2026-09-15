"""ground.* - where does this leave known truth?"""

from __future__ import annotations

from typing import Any

from crazyai.domains import dependents, get_rule, rule_index
from crazyai.toolkit.registry import tool


@tool("ground", "measure")
def what_must_be_true(rule_id: str, operator: str) -> dict:
    """Abduction over the curated graph: the minimal set of known rules that would have to change for the mutated rule to hold.

    Args:
        rule_id: The mutated rule.
        operator: The operator applied.
    """
    r = get_rule(rule_id)
    idx = rule_index()
    upstream = [idx[d] for d in r.depends_on if d in idx]
    down = dependents(rule_id, depth=6)
    changes = [{"rule_id": r.id, "statement": r.statement, "weight": r.weight, "why": f"directly mutated by {operator}"}]
    changes += [{"rule_id": u.id, "statement": u.statement, "weight": u.weight,
                 "why": "the mutated rule was derived from this; it must fail too"} for u in upstream]
    return {"minimal_changes": changes,
            "collateral": [{"rule_id": rid, "distance": d, "statement": idx[rid].statement, "weight": idx[rid].weight}
                           for rid, d in down],
            "single_change_possible": not upstream}


@tool("ground", "measure")
def cost_of_possibility(rule_ids: list) -> dict:
    """Score how much of established knowledge would be overturned if these rules were false (sum of weights, plus collateral).

    Args:
        rule_ids: Rules that would have to change.
    """
    idx = rule_index()
    direct = 0
    coll = 0
    seen = set()
    for rid in rule_ids:
        if rid not in idx:
            return {"error": f"unknown rule {rid!r}"}
        direct += idx[rid].weight
        for d, dist in dependents(rid, 6):
            if d not in seen and d not in rule_ids:
                seen.add(d)
                coll += idx[d].weight / (dist + 1)
    total = direct + coll
    band = "low" if total < 8 else "medium" if total < 20 else "high"
    return {"direct": direct, "collateral": round(coll, 2), "total": round(total, 2), "band": band,
            "reading": {"low": "one assumption we merely take for granted", "medium": "a serious revision of a field",
                        "high": "most of what is known would have to go"}[band]}


@tool("ground", "measure")
def flaw_count(measurements: list) -> dict:
    """Count the flaws reported across a list of measure-tool outputs. The invent->measure loop ends when this equals 1.

    Args:
        measurements: Outputs of measure tools (dicts). Any 'flaw', 'flaws', 'flaw_count', consistent=False, identical=False, entails=False or error counts.
    """
    total = 0
    detail: list[dict[str, Any]] = []
    for i, m in enumerate(measurements):
        if not isinstance(m, dict):
            continue
        n = 0
        if m.get("error"):
            n += 1
        if m.get("flaw"):
            n += 1
        if isinstance(m.get("flaws"), list):
            n += len(m["flaws"])
        elif isinstance(m.get("flaw_count"), int):
            n += m["flaw_count"]
        for key in ("consistent", "identical", "entails", "identifiable"):
            if m.get(key) is False and not m.get("flaw") and not m.get("flaws"):
                n += 1
        if n:
            detail.append({"measurement": i, "flaws": n})
        total += n
    return {"flaw_count": total, "by_measurement": detail,
            "verdict": "exactly one flaw" if total == 1 else "no flaw - the mutation was lost" if total == 0 else "too many flaws"}
