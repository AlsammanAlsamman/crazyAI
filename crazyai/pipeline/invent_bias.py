"""`crazyai invent --bias-from-history`: turn archived outcomes into a mild bias on future seed draws.

This is the "small engine" - bandit-style, not a neural net. There are only a
handful of archived runs, nowhere near enough to train anything real. Each
"arm" is one value along one seed-draw dimension (a blend model, a depth, an
assumption to focus on); its weight is the shrunk mean `discovery` of every
archived run that used it, plus a small explore bonus so no arm goes
permanently untried. Off unless explicitly asked for; returns None (meaning:
draw uniformly, unbiased) until enough archived history exists.
"""

from __future__ import annotations

import math
from typing import Any, Hashable

MIN_WEIGHT = 0.05


def arm_weights(rows: list[dict[str, Any]], dimension: str, arms: list[Hashable], min_samples: int,
                prior_strength: float = 3.0, ucb_bonus: float = 0.05) -> list[float] | None:
    """Per-arm sampling weights from archived outcomes, or None if there isn't enough history yet.

    Args:
        rows: archived run summaries (e.g. from `load_invent_index`), each with `dimension` and `discovery`.
        dimension: the key in each row to group by (e.g. "blend_model", "depth", "assumption_focus").
        arms: every possible value along `dimension`, in the order the caller will draw from.
        min_samples: total matching rows needed before biasing activates.
        prior_strength: pseudo-count shrinking a thin-data arm's mean toward the global mean.
        ucb_bonus: exploration bonus so no arm goes permanently untried.
    """
    usable = [r for r in rows if r.get(dimension) in arms and r.get("discovery") is not None]
    if len(usable) < min_samples:
        return None
    global_mean = sum(r["discovery"] for r in usable) / len(usable)
    weights = []
    for a in arms:
        matched = [r["discovery"] for r in usable if r[dimension] == a]
        n_a = len(matched)
        m_a = sum(matched) / n_a if n_a else 0.0
        shrunk = (n_a * m_a + prior_strength * global_mean) / (n_a + prior_strength)
        bonus = ucb_bonus / math.sqrt(n_a + 1)
        weights.append(max(MIN_WEIGHT, shrunk + bonus))
    return weights
