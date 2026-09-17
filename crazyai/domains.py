"""Curated domains: concepts, rules and the dependency graph between rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml

from crazyai.config import DOMAINS_DIR


@dataclass
class Rule:
    id: str
    domain: str
    concept: str
    statement: str
    form: dict[str, Any]
    weight: int = 5
    formal: str | None = None
    symbols: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "domain": self.domain, "concept": self.concept, "statement": self.statement,
            "form": self.form, "formal": self.formal, "symbols": self.symbols, "weight": self.weight,
            "depends_on": self.depends_on, "supports": self.supports,
        }


@dataclass
class Domain:
    name: str
    description: str
    glossary: list[str]
    concepts: dict[str, list[Rule]]

    def rules(self) -> list[Rule]:
        return [r for rs in self.concepts.values() for r in rs]


@lru_cache(maxsize=1)
def load_domains() -> dict[str, Domain]:
    out: dict[str, Domain] = {}
    for path in sorted(DOMAINS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        concepts: dict[str, list[Rule]] = {}
        for c in raw.get("concepts", []):
            rules = []
            for r in c.get("rules", []):
                rules.append(Rule(
                    id=r["id"], domain=raw["name"], concept=c["name"], statement=r["statement"],
                    form=r.get("form", {}), weight=int(r.get("weight", 5)), formal=r.get("formal"),
                    symbols=r.get("symbols", {}) or {}, depends_on=list(r.get("depends_on", []) or []),
                    supports=list(r.get("supports", []) or []),
                ))
            concepts[c["name"]] = rules
        out[raw["name"]] = Domain(raw["name"], raw.get("description", ""), list(raw.get("glossary", [])), concepts)
    return out


@lru_cache(maxsize=1)
def rule_index() -> dict[str, Rule]:
    return {r.id: r for d in load_domains().values() for r in d.rules()}


def get_rule(rule_id: str) -> Rule:
    try:
        return rule_index()[rule_id]
    except KeyError:
        raise KeyError(f"unknown rule id {rule_id!r}") from None


def dependents(rule_id: str, depth: int = 10) -> list[tuple[str, int]]:
    """Rules that (transitively) rest on rule_id, with the distance at which they were reached."""
    idx = rule_index()
    # a rule B depends on A if A lists B in supports OR B lists A in depends_on
    reverse: dict[str, set[str]] = {rid: set() for rid in idx}
    for rid, r in idx.items():
        for s in r.supports:
            if s in reverse:
                reverse[rid].add(s)
        for d in r.depends_on:
            if d in reverse:
                reverse[d].add(rid)
    seen: dict[str, int] = {}
    frontier = [(rule_id, 0)]
    while frontier:
        cur, d = frontier.pop(0)
        if d >= depth:
            continue
        for nxt in sorted(reverse.get(cur, ())):
            if nxt not in seen or seen[nxt] > d + 1:
                seen[nxt] = d + 1
                frontier.append((nxt, d + 1))
    return sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))
