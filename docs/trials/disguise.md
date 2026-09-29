# Disguise-and-translate: transform the problem, not the model

*Run 2026-09-22 · pipeline `crazyai invent-disguise` (`crazyai/pipeline/disguise.py`)
· archives `archive/disguise_7581..7585_alignment/`, `archive/disguise_7801..7805_hash/`, `archive/disguise_7591..7595_dijkstra/`*

[← back to trials index](README.md)

## The idea

Every earlier narrative mechanism applied the narrative to *Claude* - a
persona, or a blended corpus world Claude inhabits. This applies it to the
*problem* instead. Two calls, no world-blend, no corpus:

1. **Disguise.** A "playful problem-transformer" persona rewrites the real
   problem as a concrete, structurally-isomorphic, everyday non-technical
   scenario (animals, objects, places - never code or math), and proposes
   exactly **three** distinct candidate solutions to *that* scenario, in plain
   words only.
2. **Translate.** The same "rigorous performance engineer" persona that
   `Baseline` and `WorldOnly` use gets the disguise plus its three solutions
   and translates the most promising one (or a combination) into a real,
   measured implementation.

Closest published relative: [Large Language Models as Analogical
Reasoners](https://arxiv.org/abs/2310.01714) (Yasunaga et al.). This goes
further on two axes: a deliberate jump into a non-technical domain, and
multiple disguised solutions before a separate translate-back call.

## Setup

3 targets (alignment, hash, dijkstra) x 5 seeds, one seed per assumption
(`--assumption 0..4`), matching protocol-v2 / continuous / world-only. 15 real trials.

## Results

| Target | Baseline mean (wins/5) | Protocol v2 (best, wins) | Continuous (best, wins) | World-only mean (wins/5) | **Disguise mean (wins/5)** |
|---|---|---|---|---|---|
| alignment | 2.345x (3/5) | 4.85x, 4/4 | 1.17x, 2/5 | 2.189x (2/5) | 1.559x (2/5) |
| hash | 14.503x (5/5) | 16.0x, 2/5 | 15.50x, 5/5 | 15.473x (5/5) | **15.891x (5/5)** |
| dijkstra | 1.284x (5/5) | 1.11x, 2/5 | 1.30x, 2/5 | 1.029x (2/5) | 0.818x (1/5) |

- **hash - best condition yet.** 15.891x mean, 5/5 wins: the highest mean any
  condition has produced for hash, at the same perfect win rate.
- **alignment - bimodal, mid-pack.** Per-seed values `1.067, 0.587, 0.578,
  4.671, 0.893x`. Seed 7584 (calibration 0.856) hit a genuine 4.671x; two
  seeds regressed below 1x.
- **dijkstra - worst condition yet.** Four seeds in the same near-flat
  0.95-1.18x band every condition lands in. Seed 7595 (assumption "the whole
  graph must be explored to know any single distance") produced an outright
  **incorrect kernel** (`status: WRONG`, scored 0.0x) - the first wrong result
  under the assumption-pinned batch discipline.

## Mechanism check (independent of the numbers)

`archive/disguise_7581_alignment/disguise.md` - a "Twin Lantern Lane"
garden-path scenario - contains a plain-words solution that, read literally,
*is* antidiagonal wavefront parallelism: the same real technique the project
has rediscovered for alignment through several unrelated mechanisms. Yet that
seed's translate step didn't adopt it (1.067x).

**So the disguise can carry the right idea; the bottleneck is the translate
step not reliably picking the strongest of the three disguised solutions.**

## Conclusion

Disguising the problem and disguising Claude are **different levers**, not
the same one in two costumes. The persona-based conditions cluster fairly
tightly target-to-target; disguise produced a distinct fingerprint - a new
best (hash) and a new worst (dijkstra). Neither lever is a reliable win.
Dijkstra's result stays consistent with the [protocol-v3](protocol-v3.md)
diagnosis: its gap is a fallback-heap-quality capability no narrative-shaped
mechanism has addressed.

## Possible next steps

- Make translate score or benchmark all three disguised solutions instead of
  picking one by judgment.
- Investigate the 7595 wrong kernel - which disguised solution was chosen and
  where correctness broke.
