# MAPPING

**SEED 1 — "coils walked side by side on a sunlit water table, each match dropping a counted fire"**

| World object | Problem object |
|---|---|
| first cord's coils, near edge, **in full sun** | sequence `a`, contiguous byte array, held in SIMD lanes |
| each coil "the exact shape of one of its symbols" | one nucleotide byte |
| second cord's coils, **shadow-line, "waiting their turn like a queue"** | sequence `b` stored **reversed** (`pb[k]=b[n-1-k]`), so "next in the queue" is one contiguous step |
| walking the two rows together, coil against coil | sweeping an **anti-diagonal** `i+j=d`: both cords are read contiguously at once |
| "the sun windows straight through both" | `_mm_cmpeq_epi8` — 16 coil-pairs judged in a single flash, with **no order among them** |
| a small fire dropped onto the water | `+1` in that lane; no fire → `-1` |
| "I count the fires, nothing more, they are the only truth that survives" | scores only; no traceback, no path matrix |

*Breaks:* **"one pair of positions is judged at a time"** (and, in the sunlit line, "computed in that order").

**SEED 2 — "the boundary door breaks exactly once per crossing; one coil crouches past out of step"**

| World object | Problem object |
|---|---|
| the door between the two rows | the offset `i-j` between the cords |
| the door **breaking** | a gap: one cord advances without the other |
| a coil "crouching past its own turn" | the shadow cord skipping one symbol → offset changes by ±1 |
| "I try this breaking at every place a quarrel happens" | the whole family of slip positions is enumerated **up front**, not discovered cell-by-cell |
| "never twice in one crossing" | that crossing family is exactly `|i-j| ≤ 1` |
| a twice-broken row "thrown into the tower, unread; I start over" | such a path is never evaluated at all |

*Breaks:* **"a slip (gap) can only be discovered by having already compared the position before it."** The native postulates the slip *first* and then scores the resulting crossing with a dependency-free sweep — the slip is a hypothesis about geometry, not a consequence of a prior comparison.

**SEED 3 — "each crossing's pile in its own room; the failures sink into the tower; only the highest pile is brought up"**

| World object | Problem object |
|---|---|
| a room per crossing | one accumulated score per offset family |
| the tower under the ground, **unread** | the region of the `(i,j)` grid outside the admitted door-breaks — never touched |
| "walk past all the discarded rows, bring up only the highest pile" | `max` over admitted crossings = the DP maximum, restricted to the admitted region |
| "the queen's threshold, the last coil" | cell `(n,n)` |

*Breaks:* **"the whole grid of every position against every other must be filled in."**

---

# CHOSEN SEED

**SEED 2** (the door that breaks once), with SEED 1 as its walking mechanism and SEED 3 as its storage discipline. It is the only one of the three that breaks the preferred assumption.

Taken literally, SEED 2 is *wrong as stated*: a one-slip family is not global alignment. But the native's own ritual repairs it, and I refuse to quietly swap in the textbook method — I follow what the native actually does. He **first walks the straight crossing and every single-slip crossing and keeps the highest pile.** That highest pile is a *certified floor*: no crossing anywhere can beat it. And the arithmetic of the water table says how far a door may break before the floor forbids it. A crossing whose door breaks out to distance `d` must break it back again, so it burns at least `2d` gap-columns (`-4d`) and loses `d` fire-chances: its pile can never exceed `n - 5d`. So once the native knows his floor `L`, he knows the doors may only break out to

```
d ≤ (n - L) / 5
```

and everything farther is the tower: **unread**. That is precisely SEED 3's tower, sized at runtime by SEED 2's crossings. This is the same object as **Ukkonen-style score-certified banding** (as used in KSW2/edlib) — a validated technique, arrived at rather than imported — composed with the **anti-diagonal wavefront SIMD** of SEED 1 (validated, used in CUDA/SIMD aligners). I let the metaphor land on the known technique instead of inventing a novel one.

Computational mapping, kept literal:
- **memory** = three rolling anti-diagonal buffers (`O(n)`, L1/L2-resident) — the rooms; the `n²` grid is the tower and is never allocated.
- **what flows** = the wavefront, one anti-diagonal per tick.
- **what stays still** = `pa` and reversed `pb`, laid once, read contiguously forever.
- **processor** = the sun: one AVX2 register = 16 coils judged simultaneously.
- **time** = `d = 1 … 2n`, the crossing index.
- **the door's width `W`** = the runtime regime dial.

# ASSUMPTION BROKEN

*"A slip (gap) can only be discovered by having already compared the position before it."*
Here slips are **hypothesised geometrically before any cell is computed**: the crossing scan proposes slip families directly, scores them with pure order-free compares, and the best pile it yields certifies a bound `W` on how far any slip can possibly go. The grid is then only ever consulted inside that door-width.

**Regime detection (required, since the known-way section names both a bounded-score/banded regime and a dense regime):** the crossing scan *is* the detector. Similar cords → high floor → `W≈2` → `O(nW)` work. Divergent cords → floor near `-n` → `W≈0.4n` → wide band. `W` is clamped to `n`, at which point the band stops binding and the kernel degenerates *exactly* into the full anti-diagonal DP. Two further guards, because my own risk statement names them: `n < 64` (SIMD/setup overhead exceeds the win) and `n > 16000` (int16 range `[-2n, n]` would overflow) both fall back to an exact two-row scalar DP.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH (-1)
#define GAP      (-2)

/* ---- fallback: exact NW with two rolling rows (O(n) memory) ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    int *prev, *cur, r;
    prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int v = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int l = cur[j - 1] + GAP;
            if (u > v) v = u;
            if (l > v) v = l;
            cur[j] = v;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---- the native's crossings: the straight one and every single-slip one.
       Each is a dependency-free sweep; the highest pile is a certified floor. */
static int crossing_bound(int n, const char *a, const char *b)
{
    int maxs = n - 1, best;
    if (maxs > 8) maxs = 8;
    if (maxs < 0) maxs = 0;
    best = -4 * n;
    for (int s = 0; s <= maxs; s++) {
        int L = n - s, m1 = 0, m2 = 0, s1, s2;
        for (int k = 0; k < L; k++) {
            m1 += (a[k + s] == b[k]);
            m2 += (a[k] == b[k + s]);
        }
        s1 = 2 * m1 - n - 3 * s;   /* 2s gap columns (-4s) + (2m - (n-s)) */
        s2 = 2 * m2 - n - 3 * s;
        if (s1 > best) best = s1;
        if (s2 > best) best = s2;
    }
    return best;
}

int kernel(int n, const char *a, const char *b)
{
    int16_t *mem, *p2, *p1, *cu, *B0, *B1, *B2;
    char *pa, *pb;
    int W, L, NEG, sz, d, res;

    if (n <= 0) return 0;
    if (n < 64 || n > 16000) return nw_rows(n, a, b);   /* guarded regimes */

    /* --- how far may the door break?  floor L  =>  any path reaching
       |i-j| = d burns >= 2d gap columns and loses d fire-chances, so its
       pile <= n - 5d.  Hence d <= (n-L)/5.  Take one extra for safety. --- */
    L = crossing_bound(n, a, b);
    W = (n - L) / 5 + 1;
    if (W < 2) W = 2;
    if (W > n) W = n;            /* clamp: band stops binding -> full DP */

    NEG = -2 * n - 64;           /* below every reachable score, no overflow */
    sz  = n + 96;

    mem = (int16_t *)malloc((size_t)3 * sz * sizeof(int16_t));
    pa  = (char *)malloc((size_t)n + 64);
    pb  = (char *)malloc((size_t)n + 64);
    if (!mem || !pa || !pb) { free(mem); free(pa); free(pb); return nw_rows(n, a, b); }

    memcpy(pa, a, (size_t)n);
    memset(pa + n, 0x01, 64);
    for (d = 0; d < n; d++) pb[d] = b[n - 1 - d];   /* shadow row, reversed queue */
    memset(pb + n, 0x02, 64);

    for (d = 0; d < 3 * sz; d++) mem[d] = (int16_t)NEG;
    B0 = mem + 32; B1 = mem + sz + 32; B2 = mem + 2 * sz + 32;

    B0[0] = 0;                   /* cell (0,0), diagonal d = 0 */
    p2 = B2;                     /* diagonal d-2 (empty at start) */
    p1 = B0;                     /* diagonal d-1 = 0            */
    cu = B1;

#if defined(__AVX2__)
    {
    const __m256i vtwo  = _mm256_set1_epi16(2);
    const __m256i vmone = _mm256_set1_epi16(-1);
    const __m256i vgap  = _mm256_set1_epi16(GAP);
#endif
    for (d = 1; d <= 2 * n; d++) {
        int lo = d - n, hi = d - 1, t1, hb, base, i;
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        t1 = d - W;
        if (t1 > 0) { int lb = (t1 + 1) >> 1; if (lb > lo) lo = lb; }
        hb = (d + W) >> 1; if (hb < hi) hi = hb;

        base = n - d;            /* b[j-1] == pb[base + i], contiguous ascending */
        i = lo;
#if defined(__AVX2__)
        for (; i + 15 <= hi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(pa + (i - 1)));
            __m128i cb = _mm_loadu_si128((const __m128i *)(pb + (base + i)));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i sub = _mm256_add_epi16(_mm256_and_si256(eq, vtwo), vmone);
            __m256i dg = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(p2 + (i - 1))), sub);
            __m256i g1 = _mm256_loadu_si256((const __m256i *)(p1 + (i - 1)));
            __m256i g2 = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i gp = _mm256_add_epi16(_mm256_max_epi16(g1, g2), vgap);
            _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi16(dg, gp));
        }
#endif
        for (; i <= hi; i++) {
            int sv = (pa[i - 1] == pb[base + i]) ? MATCH : MISMATCH;
            int v = (int)p2[i - 1] + sv;
            int u = (int)p1[i - 1] + GAP;
            int l = (int)p1[i] + GAP;
            if (u > v) v = u;
            if (l > v) v = l;
            cu[i] = (int16_t)v;
        }
        cu[lo - 1] = (int16_t)NEG;      /* the tower: unread on both flanks */
        cu[hi + 1] = (int16_t)NEG;
        if (d <= n && d <= W) {         /* the two grid edges, when in band */
            cu[0] = (int16_t)(GAP * d);
            cu[d] = (int16_t)(GAP * d);
        }
        { int16_t *t = p2; p2 = p1; p1 = cu; cu = t; }
    }
#if defined(__AVX2__)
    }
#endif

    res = (int)p1[n];               /* the queen's threshold, cell (n,n) */
    free(mem); free(pa); free(pb);
    return res;
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 9.0**

Reasoning, stated before any measurement: the reference pays ~3 cycles/cell (the `left` dependency serialises it) over `n²` cells plus `4n²` bytes of memory traffic. The wavefront removes the dependency (16 cells/vector, ~0.25–0.4 cycles/cell) and collapses memory to `O(n)`; the certified door-width removes ~40% of cells on random DNA (`L≈-n/2 ⇒ W≈0.3n`) and far more on similar DNA. `3 / (0.6 × 0.35) ≈ 14` in theory; I discount to **9** for per-diagonal setup, short diagonals at both ends, unaligned loads, and the ~5% cost of the crossing scan.

# MEASUREMENT

**Not measured — and I will not pretend otherwise.** `alignment_bench` and `alignment_contract` were not available in this session (the harness stated no tools are callable), so the four-improvement measure-and-refine loop could not be run even once. The number above is a prediction with no measurement behind it, and the pipeline's own timing supersedes it. What I *can* state is verified by hand rather than by clock:

- The recurrence was traced cell-by-cell on `n=2, a="AC", b="AC"` through all four anti-diagonals against the reference table; every cell (including `dp[1][2]=-1`, `dp[2][1]=-1`, `dp[2][2]=2`) matched.
- Every read on diagonal `d` was shown to lie in `[lo(d)-1, hi(d)]` of the source buffer, which is exactly the range written plus the two sentinels — `lo` is non-decreasing and `hi` grows by at most 1 per diagonal, so no stale value is ever consumed. The vector loop is bounded by `i+15 <= hi`, so it never stores outside the band either.
- The band is safe because `W = ⌊(n-L)/5⌋ + 1` is *strictly* greater than the maximum admissible deviation, so every predecessor of every cell on the optimal path is itself inside the band; band cells can only be under-valued, never over-valued, so the answer at `(n,n)` is exact.

# VERDICT

The native's door is real and it is not the textbook method: it proposes slips as geometry *before* comparing anything, which is exactly the assumption the standard DP cannot break. Followed literally, the ritual produces three things that all survive translation — a sunlit order-free comparison (anti-diagonal SIMD), a certified door-width (Ukkonen-style score-banding), and a tower of grid that is never read (`O(n)` memory instead of `O(n²)`). I deliberately let it land on those two validated techniques rather than inventing something new.

Where it can lose, plainly: **(a)** if the sequences are maximally divergent the floor collapses toward `-n`, `W→0.4n`, and the band buys only ~20% — but never less than nothing, since `W` is clamped to `n` and the kernel then *is* the full wavefront DP; **(b)** below `n=64` the setup and crossing scan would dominate — guarded, falls back to exact two-row DP; **(c)** above `n=16000` int16 would overflow — guarded, same fallback. Each risk I named is guarded rather than hoped away.

What I am least sure of, and would test first with the bench: whether the crossing scan's ~5% cost is repaid at the small end of the size sweep (say `n≈64–128`), and whether an 8-bit lane path is safe once `W` is small enough that the within-diagonal range is under 127 — that would be the obvious fifth improvement, and it is the door narrowing far enough that the sun can carry twice as many coils at once.