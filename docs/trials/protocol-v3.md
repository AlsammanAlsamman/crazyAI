# Protocol v3: the regime-detection fix

*Run 2026-09-21 · change to `bend_prompt` (new step 5) · tested on dijkstra, 5 seeds (one per assumption 0-4)*

[← back to trials index](README.md)

## Background

In the baseline comparison (`crazyai-trials/README.md`), the
direct-prompt baseline beat the narrative pipeline decisively on alignment,
hash and dijkstra. Protocol v2 adopted three of the model's own suggestions
(prefer a known validated technique, guard any risk your own VERDICT names,
SIMD before threads) plus one-seed-per-assumption pinning. That fixed
alignment, partly helped hash, and barely moved dijkstra.

The one suggestion deliberately *not* adopted in v2 - "require
regime-detection when `known_way` implies more than one regime" - was the one
aimed at dijkstra, where baseline's win looked like a runtime cost model
choosing between sparse and dense strategies. Protocol v3 adds it generically
(every target) as step 5 of `bend_prompt`.

## Results (dijkstra)

| Condition | Mean | Wins |
|---|---|---|
| Baseline (no narrative) | **1.284x** | **5/5** |
| Protocol v2 | 1.028x | 2/5 |
| **Protocol v3 (+ regime detection)** | 0.983x | 1/5 |

**The fix did not close the gap** - slightly worse than v2.

## Why - from reading the kernels, not just the numbers

- **The mechanism was implemented correctly.** Seed 7561 cites "per step 5",
  names both regimes and encodes a real runtime cost check
  (`n² <= 3·(n+m)·log n`) choosing array-scan vs heap - structurally the same
  as baseline's.
- **The benchmark never triggers the dense regime.**
  `crazyai/toolkit/measure/dijkstra.py` always generates graphs with ~4n
  edges, so the check correctly falls back to the heap path every time.
  Three of five seeds landed within 0.02 of exactly 1.0x.
- **Baseline's real wins came from a better heap**, not regime switching:
  `archive/baseline_7502_dijkstra/artifact.md` credits "the indexed 4-ary heap
  over lazy-deletion binary heap" - used even on sparse graphs. v3's
  instruction told the model to fall back to "the exact reference heap
  algorithm, unchanged", so it got zero credit for the improvement that
  actually mattered.

## Conclusion

Dijkstra's gap was never about detecting regimes; it is about the default
path not being the best-known variant of the known technique (a d-ary,
cache-friendly heap). A plausible diagnosis turned out incomplete once run for
real. A one-seed alignment regression check was also run; details in
`crazyai-trials`.

50 tests passing (1 new: `bend_prompt` requires regime-detection language for
multi-regime targets).
