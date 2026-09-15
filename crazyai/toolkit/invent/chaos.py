"""chaos.* - controlled randomness. Every draw goes through the run's seeded RNG."""

from __future__ import annotations

from crazyai.config import OPERATORS
from crazyai.domains import load_domains
from crazyai.rng import RunRNG
from crazyai.toolkit.registry import tool


@tool("chaos", "invent")
def draw_seed(rng: RunRNG, domain: str = "") -> dict:
    """Draw a domain, a concept in it, and one rule of that concept to mutate.

    Args:
        domain: Optional domain name to restrict the draw (physics, number_theory, statistics, narrative, computation). Empty means any.
    """
    domains = load_domains()
    if domain:
        if domain not in domains:
            return {"error": f"unknown domain {domain!r}; known: {sorted(domains)}"}
        dname = domain
    else:
        dname = rng.choice("seed.domain", sorted(domains))
    d = domains[dname]
    concept = rng.choice("seed.concept", sorted(d.concepts))
    rule = rng.choice("seed.rule", d.concepts[concept])
    return {"domain": dname, "concept": concept, "rule": rule.to_dict(), "glossary": d.glossary}


@tool("chaos", "invent")
def draw_operator(rng: RunRNG, weights: dict | None = None) -> dict:
    """Draw one mutation operator from INVERT, REMOVE, EXTRAPOLATE, TRANSPOSE, COMPOSE, QUANTIFY, SUBSTITUTE.

    Args:
        weights: Optional mapping operator -> relative weight; missing operators get weight 1.
    """
    w = [float((weights or {}).get(op, 1.0)) for op in OPERATORS]
    return {"operator": rng.choice("operator", OPERATORS, weights=w)}


@tool("chaos", "invent")
def draw_depth(rng: RunRNG, low: int = 1, high: int = 5) -> dict:
    """Draw how many inference steps the consequence of the mutation should be pushed.

    Args:
        low: Minimum depth (inclusive).
        high: Maximum depth (inclusive).
    """
    return {"depth": rng.integer("depth", low, high)}


@tool("chaos", "invent")
def draw_analogy_pair(rng: RunRNG) -> dict:
    """Draw two different domains and a glossary term from each, as raw material for a TRANSPOSE."""
    names = sorted(load_domains())
    a = rng.choice("analogy.from", names)
    b = rng.choice("analogy.to", [n for n in names if n != a])
    da, db = load_domains()[a], load_domains()[b]
    return {
        "from_domain": a, "to_domain": b,
        "from_term": rng.choice("analogy.from_term", da.glossary),
        "to_term": rng.choice("analogy.to_term", db.glossary),
    }


@tool("chaos", "invent")
def perturb(rng: RunRNG, value: float, sigma: float = 0.1, relative: bool = True) -> dict:
    """Return value plus Gaussian noise, for nudging a parameter off its textbook value.

    Args:
        value: The number to perturb.
        sigma: Noise scale. If relative, sigma is a fraction of value.
        relative: Whether sigma is relative to value.
    """
    s = abs(value) * sigma if relative else sigma
    noise = rng.normal("perturb", 0.0, s)
    return {"original": value, "perturbed": value + noise, "noise": noise}


@tool("chaos", "invent")
def shuffle(rng: RunRNG, items: list) -> dict:
    """Return the items in a random order (used to vary framing and question order).

    Args:
        items: The list to shuffle.
    """
    return {"items": rng.shuffle("shuffle", list(items))}
