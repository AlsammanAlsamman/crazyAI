# Disguise-all: let the benchmark choose among the disguised solutions

*Run 2026-09-29 · pipeline `crazyai invent-disguise-all` (`crazyai/pipeline/disguise_all.py`)
· archives `archive/disguise_all_7581..7585_alignment/`, `archive/disguise_all_7801..7805_hash/`,
`archive/disguise_all_7591..7595_dijkstra/` · fresh timings `archive/disguise_all_remeasure.json`*

[← back to trials index](README.md)

## The idea

The [disguise-and-translate](disguise.md) trial ended on one diagnosis: the
disguise often *contains* the right idea (seed 7581's garden path was literally
antidiagonal wavefront parallelism), but the single translate call, where the
engineer persona picks the most promising of three disguised solutions by
judgment, often builds the wrong one.

This removes the judgment call:

1. **Disguise.** Reused unchanged (`--from-disguise`): the same `disguise.md`
   from the 09-22 run, so the three candidate solutions are identical.
2. **Translate each.** One call per disguised solution
   (`translate_one_prompt`). The engineer must build *that* solution's
   mechanism faithfully and may not switch approach; if it's slow, it must
   improve the implementation rather than replace the idea.
3. **Measure each, keep the fastest exact one.** The benchmark chooses, not
   the persona.

Cost: 3 translate calls per seed instead of 1.

## Fair measurement

Picking the best of three noisy timings flatters the winner (winner's curse).
So the pick is made from the pipeline's own first measurement, and the
comparison below uses **fresh** timings: the 09-22 engineer's kernel and all
three solution kernels, re-timed 5 times each, interleaved, in one sitting
(`examples/10_remeasure_disguise_all.py archive 5`).

## Results (fresh means of 5 interleaved runs)

| Target | Engineer's pick, 09-22 kernel (wins) | **Benchmark's pick (wins)** | Baseline (wins/5) | Previous best condition |
|---|---|---|---|---|
| alignment | 1.520x (2/5) | **12.720x (5/5)** | 2.345x (3/5) | protocol v2, 4.85x best |
| dijkstra | 0.767x (1/4) | **1.996x (4/4)** | 1.284x (5/5) | continuous, 1.30x best |
| hash | 17.098x (5/5) | 37.022x raw; **29.815x quality-gated** (5/5) | 14.503x (5/5) | disguise, 15.891x |

Per seed (fresh means; **bold** = benchmark's pick):

| Seed | Target | 09-22 kernel | sol 1 | sol 2 | sol 3 |
|---|---|---|---|---|---|
| 7581 | alignment | 1.066 | 1.219 | 10.661 | **14.166** |
| 7582 | alignment | 0.304 | 10.478 | **16.056** | 10.382 |
| 7583 | alignment | 0.604 | **10.531** | 21.111 | 15.907 |
| 7584 | alignment | 4.712 | 12.138 | 10.550 | **12.329** |
| 7585 | alignment | 0.915 | **10.520** | 10.714 | WRONG |
| 7591 | dijkstra | 0.953 | 0.979 | 1.412 | **1.745** |
| 7593 | dijkstra | 1.119 | 0.497 | **2.064** | 1.804 |
| 7594 | dijkstra | 0.998 | 0.466 | 1.545 | **2.592** |
| 7595 | dijkstra | WRONG | 0.498 | **1.582** | 1.256 |
| 7801 | hash | 16.959 | 5.192 | not measurable* | **30.585** |
| 7802 | hash | 17.863 | 6.705 | **32.965** | 31.484 |
| 7803 | hash | 17.575 | 5.011 | **63.330** (fails quality) | 27.296 |
| 7804 | hash | 15.629 | 29.320 | **27.928** | 27.994 |
| 7805 | hash | 17.466 | 18.534 | COMPILE_ERROR | **30.302*** |

- **Dijkstra seed 7592 is excluded.** Its original disguise has no solutions
  to choose from, so it can't test this approach.
- **\*Blocked by endpoint protection.** On this Windows machine, the
  compiled binary for 7801 solution 2 is deleted right after gcc builds it,
  and `CreateProcess` fails with `WinError 5`. The same happens to the
  mixing-test binary for 7805 solution 3. Neither was worked around. It
  doesn't change 7801's pick.
- **The pick is made on the first timing and is sometimes not the best in
  hindsight.** On 7583, solution 2 re-timed at 21.1x but solution 1 was
  chosen. The table reports the pick as made, not the best in hindsight.

## Hash: the fastest winner was a broken hash

The hash harness calls a kernel "exact" when its mean avalanche over 200
random single-bit flips is above 0.9. That test is weak.
`examples/11_hash_quality.py` runs SMHasher-style checks instead:

- a full avalanche matrix;
- 2^20 sequential counters;
- all 8,128 16-byte keys with exactly two bits set.

Each kernel is read against two references: FNV-1a (known weak, fails) and a
murmur3-finalizer hash (known strong, passes).

| Kernel | worst avalanche bias | 2-bit-key 64-bit collisions | Pass |
|---|---|---|---|
| FNV-1a reference | 0.500 | 0 | no |
| strong reference | 0.061 | 0 | yes |
| 7801 sol 3 | 0.060 | 0 | yes |
| 7802 sol 2 | 0.059 | 0 | yes |
| **7803 sol 2 (63x)** | 0.068 | **6,111** | **no** |
| 7803 sol 3 (runner-up) | 0.067 | 0 | yes |
| 7804 sol 2 | 0.064 | 0 | yes |
| 7805 sol 1 (fallback) | 0.066 | 0 | yes |
| 7805 sol 3 (pick) | *binary blocked; not tested* | | |

The 63x "winner" maps 6,111 of 8,128 two-bit keys onto colliding 64-bit
outputs, so it doesn't really work as a hash. It passed the harness because
random single-bit flips don't exercise sparse keys. **Letting the benchmark
choose amplifies any weakness in the benchmark's correctness gate.** Among
three candidates, the fastest is the one most likely to have cut a corner the
gate can't see.

The quality-gated hash mean replaces 7803's pick with its passing runner-up
(27.296x): (30.585 + 32.965 + 27.296 + 27.928 + 30.302) / 5 = **29.815x**.
7805's pick is included unverified.

## Where the gain comes from: it differs by target

- **Alignment: mostly from the translate prompt, not the choice.** Almost
  every separately translated solution lands at 10-21x. The 09-22 engineer
  built the *same* solution and got far less. On 7582 it said it built
  solution 1 and got 0.304x; translating solution 1 alone gives 10.478x.
  The per-solution kernels are nearly all AVX2 SIMD (anti-diagonal wavefront
  or provably-sufficient band) and mostly single-threaded, while 4 of the 5
  09-22 kernels use no SIMD. The prompt never mentions SIMD. It does say "if
  it turns out slow, improve how you implement this mechanism rather than
  replacing it", which the single-pick prompt doesn't.
- **Dijkstra: from the choice.** The solution matching the engineer's
  09-22 pick re-times close to the 09-22 kernel (7591: 0.979 vs 0.953). The
  gain comes from a different solution being picked. This is the first
  condition to break dijkstra's 0.95-1.3x band on every usable seed.
- **Hash: both.** Every seed has at least one solution at about 28-33x,
  roughly double the 09-22 kernels.

## Caveats

- **The model isn't controlled.** The `claudecode` provider records no model
  name in `run.json`, so the 09-22 and 09-29 runs may have used different
  underlying models. The clean control is to re-run the 09-22 single-pick
  translate on the same `disguise.md` today; that hasn't been done yet.
- **Compute is uneven:** 3 translate calls per seed against 1. Part of the
  gain could come from a best-of-3 over any 3 samples, disguised or not. A
  control that samples 3 single-pick translates would separate the two.
- n = 4-5 seeds per target. These are case studies, not statistically
  powered results.

## Conclusion

Letting the benchmark choose among all three disguised solutions produced
the largest effect this project has measured: alignment 12.7x (5/5, against
a 2.3x baseline), dijkstra 2.0x (4/4, against 1.3x), and hash 29.8x
quality-gated (5/5, against 14.5x). The fresh re-timing confirms the single
timings weren't noise.

Two findings matter as much as the numbers:

1. **The correctness gate becomes the weak point.** Once selection optimizes
   for speed, a weak gate lets a broken kernel win (7803). Any
   benchmark-chooses pipeline needs a stronger quality check than the one
   used for a single candidate.
2. **"Build this one mechanism faithfully" is itself a strong instruction,**
   at least for alignment, independent of the selection step.

## Possible next steps

- **Model control:** re-run the 09-22 single-pick translate on the same
  disguises today.
- **Compute control:** sample 3 single-pick translates and keep the best.
- Make `11_hash_quality.py` (or an equivalent) part of `hash.bench`'s gate.
- Record the model name for the `claudecode` provider in `run.json`.
