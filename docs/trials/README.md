# Trials: recent experiments

Write-ups of the most recent real trial batches (2026-09-20 to 2026-09-29).
Every number is read from archived `measure.json` files under `archive/`.
The full running record, including the older matmul/alignment/nim/... rounds,
lives in the separate `crazyai-trials` repo (`README.md` and `AIM.md`).

All optimization trials are scored as speedup over the target's reference
implementation (`1.0x` = no change, `<1.0x` = regression); a "win" is a seed
that is exact and faster than the reference.

| Date | Page | Question | One-line result |
|---|---|---|---|
| 2026-09-29 | [Disguise-all: benchmark chooses](disguise-all.md) | Does measuring all 3 disguised solutions beat the engineer's pick? | Yes, the largest effect yet: alignment 12.72x, dijkstra 1.996x, hash 29.8x quality-gated; the fastest hash winner was a broken hash |
| 2026-09-29 | [Retroactive immersion check](retro-immersion.md) | Does immersion in the native's text predict speedup? | No - immersion is saturated; the loss happens in the bend step |
| 2026-09-22 | [Disguise-and-translate](disguise.md) | Does disguising the *problem* (not Claude) help? | New best on hash (15.891x), new worst on dijkstra (0.818x), alignment mid-pack |
| 2026-09-21 | [Protocol v3: regime detection](protocol-v3.md) | Does adding regime-detection close dijkstra's gap? | No (0.983x, 1/5) - but it revealed the real gap is fallback-heap quality |
| 2026-09-21 | [World-only](world-only.md) | Does "material over persona" transfer to real targets? | Mixed: hash up, alignment flat, dijkstra down |
| 2026-09-20 | [Swapped-world causality test](swapped-world.md) | Is narrative content causal or decorative? | Technique survives a foreign world; execution quality drops (4.43x -> 1.85x) |
| 2026-09-20 | [Discover mode: baseline and more seeds](discover.md) | Does discover's narrative chain beat a direct prompt? Do more seeds help? | No and no - direct prompt matches quality; ceiling is the model |

## Cross-condition summary (the 3 targets baseline originally won)

Mean speedup (wins out of 5) unless marked "best".

| Target | Baseline | Protocol v2 (best, wins) | Protocol v3 | Continuous (best, wins) | World-only | Disguise | Disguise-all |
|---|---|---|---|---|---|---|---|
| alignment | 2.345x (3/5) | 4.85x, 4/4 | - | 1.17x, 2/5 | 2.189x (2/5) | 1.559x (2/5) | **12.720x (5/5)** |
| hash | 14.503x (5/5) | 16.0x, 2/5 | - | 15.50x, 5/5 | 15.473x (5/5) | 15.891x (5/5) | **29.815x (5/5)**, quality-gated |
| dijkstra | 1.284x (5/5) | 1.11x, 2/5 | 0.983x (1/5) | 1.30x, 2/5 | 1.029x (2/5) | 0.818x (1/5) | **1.996x (4/4)** |

**The pattern across all of them:** until disguise-all, no structural change
tested had been a uniform improvement. Roughly a third of the time a change helps, a third it
is a wash, a third it hurts - and which third depends on the target. Hash
responds well to almost any narrative variant; dijkstra resisted all of them.
Disguise-all is the first change to improve all three targets at once. Its
model and compute aren't controlled yet; see [its caveats](disguise-all.md#caveats).

*n = 5 seeds per condition per target - these are case studies, not
statistically powered results.*
