"""Loader for the optional C++ kernels, with pure-Python fallbacks.

If crazyai/_kernels.so is missing, `make native` builds it; if that is not
possible, everything still works through NumPy at lower speed. `backend()`
reports which one is active so the archive can record it.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_LIB = None
_TRIED = False


def _load():
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    so = Path(__file__).resolve().parent.parent / "_kernels.so"
    if so.exists():
        try:
            lib = ctypes.CDLL(str(so))
            lib.sieve.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8)]
            lib.collatz.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64)]
            lib.even_plus_prime_counts.argtypes = [
                ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32)]
            lib.mc_linear_gaussian.argtypes = [
                ctypes.c_uint32, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_double)]
            _LIB = lib
        except OSError:
            _LIB = None
    return _LIB


def backend() -> str:
    return "cpp" if _load() is not None else "python"


def sieve(n: int) -> np.ndarray:
    """Boolean array of length n+1, True where the index is prime."""
    lib = _load()
    if lib is not None:
        out = np.ones(n + 1, dtype=np.uint8)
        lib.sieve(n, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out.astype(bool)
    out = np.ones(n + 1, dtype=bool)
    out[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if out[i]:
            out[i * i :: i] = False
    return out


def collatz(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(stopping_time, peak) arrays indexed 0..n (index 0 unused)."""
    lib = _load()
    if lib is not None:
        ln = np.zeros(n + 1, dtype=np.uint32)
        pk = np.zeros(n + 1, dtype=np.uint64)
        lib.collatz(n, ln.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                    pk.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
        return ln, pk
    ln = np.zeros(n + 1, dtype=np.uint32)
    pk = np.zeros(n + 1, dtype=np.uint64)
    for k in range(1, n + 1):
        x, p, steps = k, k, 0
        while x != 1:
            x = 3 * x + 1 if x & 1 else x >> 1
            p = max(p, x)
            steps += 1
        ln[k], pk[k] = steps, p
    return ln, pk


def even_plus_prime_counts(limit: int, is_prime: np.ndarray) -> np.ndarray:
    """out[n] = number of ways n = p + 2k, p prime, k >= 1."""
    lib = _load()
    if lib is not None:
        ip = np.ascontiguousarray(is_prime.astype(np.uint8))
        out = np.zeros(limit + 1, dtype=np.uint32)
        lib.even_plus_prime_counts(limit, ip.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                   out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)))
        return out
    primes = np.flatnonzero(is_prime[: limit + 1])
    out = np.zeros(limit + 1, dtype=np.uint32)
    for n in range(limit + 1):
        d = n - primes
        out[n] = int(np.sum((d >= 2) & (d % 2 == 0)))
    return out


def mc_linear_gaussian(coef, mu, sigma, draws: int, seed: int) -> tuple[float, float]:
    """Mean and variance of sum_i coef[i]*N(mu[i], sigma[i]) by Monte Carlo."""
    coef = np.asarray(coef, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    lib = _load()
    if lib is not None:
        out = np.zeros(2, dtype=float)
        dp = ctypes.POINTER(ctypes.c_double)
        lib.mc_linear_gaussian(len(coef), coef.ctypes.data_as(dp), mu.ctypes.data_as(dp),
                               sigma.ctypes.data_as(dp), draws, seed, out.ctypes.data_as(dp))
        return float(out[0]), float(out[1])
    g = np.random.default_rng(seed)
    x = g.normal(mu, sigma, size=(draws, len(coef))) @ coef
    return float(x.mean()), float(x.var())
