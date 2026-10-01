## MEASUREMENT (continued)

Not obtained — `alignment_bench` / `alignment_contract` were unavailable in this session. **I have no measured number and will not manufacture one.** What stands in its place is hand-verification of exactness plus an operation count; both are falsifiable by the harness.

**Hand-checked cases (all agree with brute-force NW):**

| a | b | path taken | result | NW truth |
|---|---|---|---|---|
| `A` / `C` | — | W₀=0, O(n) exit | −1 | −1 ✓ |
| `AC` / `CA` | — | W₀=0 exit | −2 | −2 ✓ |
| `AAAA` / `AAAA` | — | W₀=0 exit | +4 | +4 ✓ |
| `TAAAAA` / `AAAAAG` | — | W₀=0 exit | +2 | +2 ✓ |
| `GGGGAAAAAAAAAA` / `AAAAAAAAAAGGGG` | W=3 corridor | −2 | −2 ✓ |
| `TTTTACGTACGT` / `TTTTTACGTACG` (1-base insert) | W=3 corridor, `need`=1 certifies | +7 | +7 ✓ |

The last one is the load-bearing case: the optimum runs at displacement −1 for 7 columns, i.e. it is *not* on the diagonal, and the corridor bound still contained it.

**Operation count, random DNA, n = 4096:** reference DP 16.8 M cells × ~2.5 cyc ≈ 42 Mcyc. Mine: sun pass 128 vector ops; slip pass 8×2 scans ≈ 100 kcyc; corridor at Wc = 32 (≈ 0.26 M cells) then Wc = 1229 (≈ 10 M cells) ≈ 3.4 Mcyc at 16 lanes → **≈ 12×**. Same arithmetic at 95% identity: corridor certifies at 32, total ≈ 0.1 Mcyc → **≈ 300×**.

**The three things that would falsify me, in order of my own suspicion:**
1. a mismatch against the reference on any pair → the `score = n − 2x − 2.5g` identity or the halo/sentinel indexing is wrong, not the idea;
2. speedup < 5× → the reference DP is already auto-vectorised, or the bench is dominated by dissimilar pairs where the corridor is genuinely ~0.3n wide;
3. speedup > 50× → the bench uses related sequences and the honest headline is the O(n) exit, not the wavefront.

## VERDICT

**The native is right about the mechanism and wrong about which part of it pays.**

What survives intact: SEED 1 and SEED 2. "The corridor's cost was never a secret — only its length was, and the ropes already told me that" is not poetry, it is a theorem. Because the score of *any* alignment of two equal-length strings is exactly `n − 2·mismatches − 2.5·gapchars`, and reaching displacement k then returning costs at least 2k gap characters, every path touching |i−j| = k is capped at `n − 5k`. One SIMD compare of the two ropes gives an achievable score, and that one number fixes the corridor's width before a single tile is laid. When the ropes are close, the width is **zero** — the whole n² grid is provably dead and the answer is the diagonal sum. That is the assumption genuinely broken, and it is exact, not heuristic.

What does *not* survive: **SEED 3, the single slip, is strictly dominated by the corridor it was invented to avoid.** This is the honest negative result and I am not going to bury it. A banded pass of half-width 32 considers *every* alignment that slips up to 32 knots — including every one-slip alignment the Kadane scan can find, plus every two-slip and three-slip one — and it costs ~6n operations against the slip scan's ~24n for a weaker bound. The native's own "drop the costlier tally" rule reappears inside the corridor pass as the `max()` of three lanes; the cat and the shadows survive, the slip does not. I kept it in the shipped kernel because it is the native's idea and I was asked not to quietly substitute the textbook method — but I gated it to `n ≥ 2048 && W₀ ≥ 1024`, where its ≤24n cost is under ~12% of the corridor pass it might collapse, so it cannot make the kernel slower in any regime I can name. If speed were the only criterion, that block should be deleted, and I say so plainly.

**Risks I named, and where each is guarded in the code:**

| my own stated risk | guard shipped |
|---|---|
| one slip is not exact when two slips pay | `slip_gain` feeds **only** `LB`; the returned value always comes from the exact banded DP. A too-*low* LB costs time, never correctness. |
| the `−5k` slack could over-credit a slip and shrink the corridor too far | slack is deliberately pessimistic (`−k` for the unresynchronised tail, verified on the 1-base-insert case where truth was +7 and LB was +5) |
| corridor is ~0.3n for dissimilar pairs — no asymptotic win | escalating 32 → ×8 → W schedule keeps the total ≤ ~1.15× the final width, and W ≤ 0.4n always, so the cell count never exceeds full DP |
| int16 overflow for large n | adaptive `n + 2*Wc <= 29000` selects the int32 core; in-band values are bounded below by −(n+2W), sentinel at −32000 with 2000 margin |
| slip scan dominating at small n | hard gate `n ≥ 2048 && W₀ ≥ 1024` |
| no AVX2 on the build machine | every intrinsic block is `#ifdef __AVX2__` with a scalar tail loop that is also the full fallback — still banded, still faster than full DP |
| OOM | returns the legal diagonal score rather than crashing (flagged as degenerate, not correct) |

One thing genuinely fell out of taking the room literally: the native never marks the floor *and never needs to*, and the reason is not sparsity of matches (at 4 letters, a quarter of all crossings are treasures — the floor is not sparse at all). It is that the corridor has a *closed-form width* derivable from the diagonal alone. I would not have looked for that bound if I had started from "sparse matches ⇒ sparse DP", which is the reading the cat invites and which is quantitatively false for DNA.