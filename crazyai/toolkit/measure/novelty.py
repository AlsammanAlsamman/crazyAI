"""novelty.* - has this been done before (in our own archive)?"""

from __future__ import annotations

import json
from pathlib import Path

from crazyai.config import ARCHIVE_DIR
from crazyai.toolkit.registry import tool


@tool("novelty", "measure")
def search_archive(rule_id: str, operator: str = "", domain: str = "", limit: int = 5) -> dict:
    """Find prior archived artifacts with the same rule / operator / domain, nearest first.

    Args:
        rule_id: Mutated rule id.
        operator: Operator (optional).
        domain: Domain (optional).
        limit: Maximum matches.
    """
    hits = []
    root = Path(ARCHIVE_DIR)
    if not root.exists():
        return {"matches": [], "note": "archive is empty"}
    for mpath in root.glob("*/mutation.json"):
        try:
            m = json.loads(mpath.read_text())
        except Exception:
            continue
        score = 0
        if m.get("rule_id") == rule_id:
            score += 3
        if operator and m.get("operator") == operator:
            score += 2
        if domain and m.get("domain") == domain:
            score += 1
        if score:
            hits.append({"run": mpath.parent.name, "score": score, "rule_id": m.get("rule_id"),
                         "operator": m.get("operator"), "generator": m.get("generator")})
    hits.sort(key=lambda h: (-h["score"], h["run"]))
    return {"matches": hits[:limit], "exact_duplicates": sum(1 for h in hits if h["score"] >= 5)}
