# Retroactive immersion check (diagnosis kit, step 0)

**Date:** 2026-09-29 - **Cost:** no API calls - **Script:** `examples/09_retro_immersion.py`, metrics in `crazyai/diagnose.py`

**Question:** does how deeply the native stays in its world (the `immerse` text,
`ideas.md`) predict the real speedup of the kernel that comes out of the run?
If not, a diagnosis kit built on immersion scores would be measuring the wrong thing.

**Data:** 109 archived `invent_*` runs across 9 targets. Speedups are not
comparable across targets, so the outcome is each run's percentile rank within
its own target, and p-values come from a 10,000-shuffle permutation test that
only shuffles within a target.

## Metrics (plain counts, no LLM judge)

| metric | meaning | mean |
|---|---|---|
| `world_rate` | share of content words also found in the world text | 0.175 |
| `first_person` | I/me/my/we per word | 0.032 |
| `tech_rate` | computing vocabulary per word | 0.0076 |
| `meta_rate` | "metaphorically", "in reality", "as an AI"... per 100 words | 0 (never occurs) |
| `drift` | tech_rate in second half minus first half | 0.0014 |
| `survival` | share of the native's world words that reappear in the engineer's `artifact.md` | 0.568 |

## Result

| metric | rho vs speedup rank | perm p |
|---|---|---|
| immersion (composite) | 0.100 | 0.28 |
| world_rate | 0.090 | 0.33 |
| first_person | 0.069 | 0.44 |
| tech_rate | 0.040 | 0.64 |
| drift | 0.195 | 0.041 |
| survival | -0.009 | 0.92 |
| existing `imagination_score` | -0.145 | 0.12 |
| depth | -0.009 | 0.92 |

Per target, immersion vs speedup: matmul +0.38 (n=36, p=0.02), dijkstra +0.61
(n=15, p=0.02), alignment -0.29, wht -0.80 (n=5), the rest near zero.

## What it means

1. **Immersion is already saturated.** The native almost never bends back in
   the immerse text: under 1% technical words, zero meta phrases, 27 of 109
   texts with no technical word at all. There is little variance left to
   push, and what there is does not predict the outcome pooled across targets.
2. **Bend-back happens later, in the bend step, not in the immersion.** This
   matches the disguise and swapped-world findings: the idea appears in-world
   and gets lost or re-chosen during translation.
3. **Word overlap is too crude to measure that loss.** `survival` does not
   predict anything; the engineer paraphrases, so shared words are not shared ideas.
4. **`drift` is the only pooled hint** (texts that lean slightly concrete toward
   the end do better), but with 9 metrics tested it does not survive correction
   (Bonferroni p ~ 0.37). Worth re-testing prospectively, not acting on.
5. **The per-target sign flips** (matmul/dijkstra up, alignment/wht down) are the
   same "which third depends on the target" pattern seen in every other condition.
6. The existing `imagination_score` is not a useful proxy either (slightly negative).

**Consequence for the kit:** don't spend the first API budget on an
immersion-depth grid for Claude. Point Q3 and the metrics at the *translation*
step - "let the benchmark choose" among all in-world solutions, and a
structural (not lexical) measure of whether the native's mechanism survives
into the code.

*n = 109 runs, observational, several prompt versions mixed - a screening
result, not a controlled experiment.*
