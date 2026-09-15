"""One seeded random source per run.

Every invent tool that needs randomness draws from this object and nothing
else, so a run is fully reproducible from its seed. Each draw is logged with
a label so the archive can show exactly where chaos entered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


def _loggable(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_loggable(v) for v in value]
    if hasattr(value, "id"):
        return getattr(value, "id")
    return str(value)


@dataclass
class RunRNG:
    seed: int
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._gen = np.random.default_rng(self.seed)

    def _record(self, label: str, value: Any) -> Any:
        self.log.append({"label": label, "value": _loggable(value)})
        return value

    def choice(self, label: str, items: Sequence[Any], weights: Sequence[float] | None = None) -> Any:
        if not items:
            raise ValueError(f"{label}: nothing to choose from")
        p = None
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            p = w / w.sum()
        idx = int(self._gen.choice(len(items), p=p))
        return self._record(label, items[idx])

    def integer(self, label: str, low: int, high: int) -> int:
        """Inclusive on both ends."""
        return self._record(label, int(self._gen.integers(low, high + 1)))

    def uniform(self, label: str, low: float = 0.0, high: float = 1.0) -> float:
        return self._record(label, float(self._gen.uniform(low, high)))

    def normal(self, label: str, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self._record(label, float(self._gen.normal(mu, sigma)))

    def shuffle(self, label: str, items: Sequence[Any]) -> list[Any]:
        order = self._gen.permutation(len(items)).tolist()
        self._record(label, order)
        return [items[i] for i in order]

    def child(self, label: str) -> "RunRNG":
        """A derived generator for a sub-step, so sub-steps stay independent."""
        sub = self.integer(f"child:{label}", 0, 2**31 - 1)
        return RunRNG(seed=sub)
