# MAPPING

**SEED 1 — "Two knot-cords laid crosswise over a sand-tray mark every place their symbols meet."**

| World object | Problem object |
|---|---|
| north cord, horse's length | `a[0..n-1]`, the row index `i` |
| east cord, mountain's width | `b[0..n-1]`, the column index `j` |
| furrow-point (crossing) | DP cell `(i,j)` |
| lattice of crossings | the `(n+1)×(n+1)` score grid |
| "a symbol from one cord faces a symbol from the other" | substitution term `a[i-1]==b[j-1] ? +1 : -1` |
| mound height at a junction | `dp[i][j]` |
| the sand-tray itself (the only ledger) | one in-place score buffer, **no auxiliary E/F arrays** |

*Breaks:* "the whole grid of every position against every other must be filled in" — only weakly: the tray still *marks* every crossing. This seed is mostly scaffolding; it breaks nothing sharply.

**SEED 2 — "A single ink-worm walks the eight desert winds and curls back across its own trail to test a junction twice."**

| World object | Problem object |
|---|---|
| **one** ink-worm (not one memory per crossing) | **one** SIMD register that sweeps the tray — the sole compute agent; no thread pool, no per-cell task |
| "one cell wide, a stitch cut loose from the verses" | a one-row-tall strip: rolling two-row buffer, `O(n)` memory instead of `O(n²)` |
| **the eight desert winds it can face** | the **eight `int32` lanes of a 256-bit register** — eight junctions settled in one motion |
| walks straight along the grain | `+MATCH` |
| slants across a furrow, spending a grain | `-MISMATCH` |
| "curls its own body across ground already crossed" | `vpermd` shifting the register **across itself** at strides 1, 2, 4 — Hillis–Steele prefix-max |
| "drops a mound... reads what already sits there... kicks the old one flat" | `vpmaxsd`: in-place max against the standing value |
| "if not, doubles back to try the junction from another face" | the re-crossing at the next stride; the junction is re-tested from a farther face |
| "keeps no ledger apart from the sand itself; a beaten mound is thrown away" | strictly in-place max, no traceback, no second array |
| "no junction anywhere still holds a mound it could improve" | the scan has converged: 3 doublings saturate 8 lanes; the inter-block carry is the last unimproved mound |
| the worm's whole continuous path | one fused sweep per row: `n/8` settlings, not `n` glances |
| the height at the far corner | `dp[n][n]`, the returned score |
| **time** | the worm's motion: row index `i` (the only true serial axis) |
| **what stays still** | the tray = the two score rows and the widened `b` |
| **what flows** | the worm = the register, plus one scalar *carry* mound at its tail |

*Breaks:* **"one pair of positions is judged at a time"** — the curled worm settles eight junctions in one motion. Also breaks "cells depend on above/left/diag computed in that order": the worm arrives at a junction from an arbitrary wind and re-tests it later from another face, so the left-dependency is resolved *after* the fact, by curling, not before it, by ordering.

**SEED 3 — "The worm crosses a furrow sideways, spending no sand, to let one cord slip a step ahead of the other."**

| World object | Problem object |
|---|---|
| sideways furrow-crossing | a gap move (`j-1 → j`) |
| "spending no sand" | in the **shifted coordinate** `G[j] = H[j] + 2j`, a gap move costs exactly **0** |
| "the one permitted stumble" | the slip has a *fixed, known* price, so a run of slips is a plain maximum, not a search |
| letting one cord slip ahead | the closed form `dp[i][j] = max_{k≤j} G[k] − 2j` |

*Breaks:* "a slip can only be discovered by having already compared the position before it." If a sideways crossing is free in the right coordinate, a whole run of slips is a **prefix maximum** — obtainable without walking the run.

# CHOSEN SEED

**SEED 2**, the single ink-worm of the eight winds. It is the only one of the three that breaks the preferred assumption *"one pair of positions is judged at a time"* (the curled body settles eight junctions at once), it is the most literal (eight winds → eight lanes; curling across its own trail → `vpermd` self-shift; "no ledger apart from the sand" → in-place, no F-array), and it is the most different from the known way, which sends one hooded memory to kneel at each crossing in order.

SEED 3 is not discarded — it is the *change of coordinate* that makes the worm's free sideways crossing expressible, so the worm's curl computes something exact. And SEED 1 supplies the runtime regime probe: the worm's very first act is a straight crawl down the crossing-line of the two cords, which tells it how far from that line it could ever profitably wander.

# ASSUMPTION BROKEN

1. **"One pair of positions is judged at a time."** Eight junctions are settled per motion of the worm (AVX2, `int32×8`).
2. **"Every cell depends on above/left/diag, computed in that order."** The left dependency is never computed in order: in the coordinate `G[j]=H[j]+2j` a sideways slip is free, so the entire left chain of a row is a prefix maximum, obtained by the worm curling across its own trail at strides 1, 2, 4 and by a one-`max` carry between motions. The serial chain per row drops from `n` dependent maxima to `n/8`.
3. **"The whole grid must be filled."** The worm's first straight crawl yields `L0` = score of the pure diagonal alignment. Since any path deviating `d` from the diagonal scores at most `n − 5d` (proof below), and `L0` is achievable, no optimal path deviates more than `D = ⌊(n − L0)/5⌋`. The worm keeps to that many furrows either side of the crossing-line.

*Proof of the band bound.* An alignment of two length-`n` strings with `p` letter–letter columns has exactly `n − p` gap columns on each side, so its score is `m − x − 4(n−p) ≤ p − 4(n−p) = 5p − 4n`. Reaching offset `d` forces `n − p ≥ d`, hence score `≤ n − 5d`. With `d ≥ D+1`, score `≤ n − 5D − 5 ≤ L0 − 1 <` optimum. So band `|j−i| ≤ D` is exact.

Where the mechanism lands on validated ground: the free-slip coordinate + prefix max is exactly the **scan formulation of linear-gap alignment** (the parallel-scan Smith–Waterman of Khajeh-Saeed et al.), the curl-until-no-mound-improves is Farrar's **lazy-F** correction made deterministic, and the band probe is the standard **score-bounded band**. I let the metaphor arrive at those rather than inventing a new one.

# ARTIFACT

```c
#include <stdlib.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* Small-tray fallback: the worm is not worth waking for a tiny sand-tray.
   Plain two-row Needleman-Wunsch, stack only, no malloc, no SIMD setup. */
static int nw_small(int n, const char *a, const char *b)
{
    int prev[65], cur[65];
    for (int j = 0; j <= n; j++) prev[j] = GAP * j;
    for (int i = 1; i <= n; i++) {
        int ai = (unsigned char)a[i - 1];
        cur[0] = GAP * i;
        for (int j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == (unsigned char)b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int l = cur[j - 1] + GAP;
            int best = d;
            if (u > best) best = u;
            if (l > best) best = l;
            cur[j] = best;
        }
        for (int j = 0; j <= n; j++) prev[j] = cur[j];
    }
    return prev[n];
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < 64) return nw_small(n, a, b);           /* guard: small-tray regime */

    /* ---- the worm's first straight crawl down the crossing-line of the two
            cords: how well do the symbols agree with no slip at all?        */
    int L0 = 0;
    for (int k = 0; k < n; k++)
        L0 += ((unsigned char)a[k] == (unsigned char)b[k]) ? MATCH : MISMATCH;

    /* provable furthest furrow any optimal path can reach (score <= n - 5d) */
    int D = (n - L0) / 5;
    if (D > n) D = n;                               /* degenerates to full tray */

    /* NEG: a mound so low no path through it can ever win.
       real scores >= -4n ; NEG-derived <= NEG + 2n = -6n-64 < -4n.          */
    const int NEG = -(8 * n + 64);

    const size_t pad = (size_t)n + 24;
    int *mem = (int *)malloc(3u * pad * sizeof(int));
    if (!mem) return nw_small(n < 64 ? n : 0, a, b); /* cannot happen for n>=64 */
    int *__restrict prev = mem;
    int *__restrict cur  = mem + pad;
    int *__restrict bi   = mem + 2u * pad;

    for (size_t t = 0; t < 2u * pad; t++) mem[t] = NEG;   /* whole tray flat-low */
    for (int k = 0; k < n; k++) bi[k] = (unsigned char)b[k];
    for (size_t k = (size_t)n; k < pad; k++) bi[k] = -1;  /* never matches */

    int hi0 = (D < n) ? D : n;
    for (int j = 0; j <= hi0; j++) prev[j] = GAP * j;     /* row 0, inside band */

    for (int i = 1; i <= n; i++) {
        int lo = i - D; if (lo < 0) lo = 0;
        int hi = i + D; if (hi > n) hi = n;

        int jstart, carry;                  /* carry lives in G-space: H + 2j */
        if (lo == 0) { cur[0] = GAP * i; jstart = 1; carry = GAP * i; }
        else         { jstart = lo;      carry = NEG; }

        int ai = (unsigned char)a[i - 1];
        int j  = jstart;

#if defined(__AVX2__)
        if (hi - jstart >= 7) {
            const __m256i ninf = _mm256_set1_epi32(-(1 << 30));
            const __m256i two  = _mm256_set1_epi32(2);
            const __m256i one  = _mm256_set1_epi32(1);
            const __m256i av   = _mm256_set1_epi32(ai);
            const __m256i s1   = _mm256_setr_epi32(0,0,1,2,3,4,5,6);
            const __m256i s2   = _mm256_setr_epi32(0,0,0,1,2,3,4,5);
            const __m256i s4   = _mm256_setr_epi32(0,0,0,0,0,1,2,3);
            const __m256i last = _mm256_set1_epi32(7);
            const __m256i step = _mm256_set1_epi32(16);
            __m256i offv = _mm256_setr_epi32(2*j,    2*j+2,  2*j+4,  2*j+6,
                                             2*j+8,  2*j+10, 2*j+12, 2*j+14);
            __m256i cbv  = _mm256_set1_epi32(carry);

            for (; j + 7 <= hi; j += 8) {
                /* eight winds in one motion: eight junctions settled at once */
                __m256i bv  = _mm256_loadu_si256((const __m256i *)(bi   + j - 1));
                __m256i pm1 = _mm256_loadu_si256((const __m256i *)(prev + j - 1));
                __m256i pj  = _mm256_loadu_si256((const __m256i *)(prev + j));
                __m256i eq  = _mm256_cmpeq_epi32(bv, av);
                __m256i sc  = _mm256_sub_epi32(_mm256_and_si256(eq, two), one);
                __m256i d   = _mm256_add_epi32(pm1, sc);      /* slant   */
                __m256i u   = _mm256_sub_epi32(pj,  two);      /* straight*/
                __m256i v   = _mm256_add_epi32(_mm256_max_epi32(d, u), offv);
                __m256i t;
                /* the worm curls across its own trail: strides 1, 2, 4.
                   Each curl re-tests every junction from a farther face;
                   after the third no junction can still be improved.       */
                t = _mm256_permutevar8x32_epi32(v, s1);
                t = _mm256_blend_epi32(t, ninf, 0x01);
                v = _mm256_max_epi32(v, t);
                t = _mm256_permutevar8x32_epi32(v, s2);
                t = _mm256_blend_epi32(t, ninf, 0x03);
                v = _mm256_max_epi32(v, t);
                t = _mm256_permutevar8x32_epi32(v, s4);
                t = _mm256_blend_epi32(t, ninf, 0x0F);
                v = _mm256_max_epi32(v, t);
                /* the one mound left standing behind the worm's tail */
                v   = _mm256_max_epi32(v, cbv);
                cbv = _mm256_permutevar8x32_epi32(v, last);
                _mm256_storeu_si256((__m256i *)(cur + j),
                                    _mm256_sub_epi32(v, offv));
                offv = _mm256_add_epi32(offv, step);
            }
            carry = _mm_cvtsi128_si32(_mm256_castsi256_si128(cbv));
        }
#endif
        /* tail of the band (and the whole row when no AVX2): the same
           free-sideways-slip coordinate, so still one max per junction. */
        for (; j <= hi; j++) {
            int d = prev[j - 1] + (ai == bi[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int h = (d > u) ? d : u;
            int g = h + 2 * j;
            if (g > carry) carry = g;
            cur[j] = carry - 2 * j;
        }

        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[n];
    free(mem);
    return result;
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 8**

*(written before any measurement; no measurement was possible in this session — see MEASUREMENT.)*

Reasoning behind the number, so it can be scored honestly:
- reference: `n²` cells, serial `add→max` chain ≈ 2 cyc/cell, plus streaming `4n²` bytes of table it never re-reads;
- worm: ~24 uops per 8 cells ≈ 0.75 cyc/cell, serial chain only `n/8` maxima per row, two L1-resident rows;
- band factor for random DNA (`L0 ≈ −n/2` → `D ≈ 0.3n`) ≈ 0.6× the cells.
`2 / (0.75 × 0.6) ≈ 4.4` from compute, ×~1.5–2 from the reference's memory traffic → **≈ 7–9**. If the benchmark sequences are similar rather than random, the band collapses and the true speedup should be far larger (identical strings → `D = 0`, `O(n)` work); if they are anti-correlated, `D → n` and the band contributes nothing, leaving ~4–5.

# MEASUREMENT

**No measurement was made.** `alignment_bench`, `alignment_contract` and the other tools are listed but not available in this session, so nothing was compiled, verified against the reference, or timed. I am reporting that plainly rather than quoting a plausible number. What the pipeline should find:

| check | expected |
|---|---|
| exactness vs reference DP | bit-identical for every input, including `n` around the 64 threshold, all-identical strings, shifted strings, and random strings |
| speedup, random DNA | 4–9× |
| speedup, ≥95% identical strings | ≫10× (band width `≈ n/10` or less) |
| speedup, `n < 64` | ≈1× (deliberately falls back) |
| non-AVX2 build | ~2× from the scan coordinate alone (1 max/cell chain instead of 2) |

Hand-verification I *did* do, since I could not run anything: the G-space recurrence and the band sentinels were checked by hand against the reference on `n=1,2` (`AC`/`CA`, `AA`/`AT`, identical strings, `ACGTACGT`/`CGTACGTA` where the `n−5d` bound is tight at score 3). The sentinel invariant — that the position just outside a row's band is still `NEG` in the ping-pong buffer, so no stale real value is ever read as `−∞` — was proved from write ranges (`row i` writes `[i−D, i+D]`; `row i+1` reads at most `i+1+D`; the same buffer's previous use wrote only to `i−2+D`).

# VERDICT

The native's picture translates without any slack, and the translation is not a new algorithm — it is three validated ones that the metaphor happens to name exactly:

- **eight desert winds → eight `int32` lanes.** The worm settling eight junctions in one motion is the broken assumption, and it is the whole speedup.
- **"crosses sideways, spending nothing" → the scan coordinate.** This is the load-bearing line. A gap costing zero is false in the contract's scoring, but true in `G[j] = H[j] + 2j`, and in that coordinate a run of slips is a prefix maximum. That is what makes the eight-lane motion legal at all: without it the left dependency is strictly serial and no width of worm helps. It lands on the published parallel-scan formulation of linear-gap alignment.
- **"curls its own body across ground already crossed... until no junction holds a mound it could improve" → Hillis–Steele self-shift.** Farrar's lazy-F, made branchless: three curls at strides 1, 2, 4 provably saturate eight lanes, so "no mound left to improve" is a compile-time fact rather than a loop with a data-dependent trip count. The metaphor asked for iterate-to-convergence; the register width tells us convergence is at three.
- **"keeps no ledger apart from the sand itself" → no E/F arrays, two rows, `O(n)` memory.** This is probably worth as much as the SIMD against a reference that streams `4n²` bytes.
- **the first straight crawl → runtime regime detection.** The two regimes in `known_way` (full vs. banded) are both handled by *one* code path: `D = ⌊(n − L0)/5⌋` with `D ≥ n` degenerating to full width at the cost of a single `O(n)` probe. The metaphor's own notion of "how far from the crossing-line is it ever worth wandering" turned out to be a provable inequality, `score ≤ n − 5d`, not a heuristic — so the band is exact, not a gamble.

Stated risks and how each is addressed rather than left hanging:
- **overhead if small** → hard `n < 64` fallback to stack-only scalar NW, no malloc, no SIMD prologue.
- **no AVX2** → `#ifdef` leaves the scan-coordinate scalar loop, which is still ~2× the reference's chain; nothing depends on the vector path for correctness.
- **band could be useless** → it costs one `O(n)` pass and the identical code path; it cannot be worse than full width by more than that pass.
- **thread parallelism** → deliberately absent. The native woke *a single* worm, explicitly in contrast to "sending a hooded memory to kneel at each crossing"; a row is `n` cells of one-cycle-deep work, far too little to pay for a barrier per row. Staying single-threaded is both the faithful reading and the right engineering call.

What I would be worried about if I could measure: the inter-block carry chain (`max` + `vpermd` ≈ 4 cycles per 8 cells, i.e. 0.5 cyc/cell) and `vpermd`'s single-port pressure (4 shuffle-port ops per block) may together dominate, pinning the kernel near 0.5–0.75 cyc/cell and putting the real speedup at the low end of my range, ~5× on random DNA, rather than 8. If the measured number comes back below ~3× on random input, my estimate of the reference's cost was simply wrong — most likely the compiler vectorized more of it than I assumed — and I would say so rather than re-tune the prediction.