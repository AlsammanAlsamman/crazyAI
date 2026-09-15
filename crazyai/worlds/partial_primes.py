"""A world where primes are only partly prime.

Three conditions decide how much of a prime an integer n is:

  even_plus_prime(n): n = p + 2k for a prime p and k >= 1        (counted, not asserted)
  fake_even(n):       n is odd but n*n == 1 (mod 8) - "behaves like an even" under the mod-8 square relation
  pi_variation(n, w): the decimal digits of n appear in the first w digits of pi

Partial primality is the number of conditions satisfied (0-3). This module
computes it honestly for the first `limit` integers - the C++ sieve and the
even+prime counter do the heavy lifting - so a predictor can be fitted with
real numbers. The impossibility is not hidden here; it is in the artifact
that is later built on top of these numbers.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from mpmath import mp

from crazyai.toolkit import native


@lru_cache(maxsize=4)
def pi_digits(n: int) -> str:
    mp.dps = n + 10
    return mp.nstr(mp.pi, n + 1, strip_zeros=False).replace(".", "")[:n]


def label_world(limit: int, pi_window: int = 5000) -> dict[str, np.ndarray]:
    is_prime = native.sieve(limit)
    n = np.arange(limit + 1)
    counts = native.even_plus_prime_counts(limit, is_prime)
    even_plus_prime = counts > 0
    fake_even = (n % 2 == 1) & ((n * n) % 8 == 1)
    digits = pi_digits(pi_window)
    pi_var = np.array([str(k) in digits for k in n], dtype=bool)
    partial = even_plus_prime.astype(int) + fake_even.astype(int) + pi_var.astype(int)
    return {
        "n": n, "is_prime": is_prime, "even_plus_prime": even_plus_prime, "even_plus_prime_count": counts,
        "fake_even": fake_even, "pi_variation": pi_var, "partial_primality": partial,
    }


def features(n: np.ndarray) -> dict[str, np.ndarray]:
    """Cheap features a predictor might use: residues, digit sum, magnitude."""
    return {
        "mod2": (n % 2).astype(float), "mod3": (n % 3).astype(float), "mod4_is1": (n % 4 == 1).astype(float),
        "digit_sum": np.array([sum(int(c) for c in str(k)) for k in n], dtype=float),
        "log_n": np.log(np.maximum(n, 1)).astype(float),
    }
