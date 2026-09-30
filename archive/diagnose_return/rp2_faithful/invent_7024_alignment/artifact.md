## MAPPING

**SEED 1 — "coils shaped like symbols are walked side by side on a sunlit water table, each match dropping a fire counted below"**

| World object | Problem object |
|---|---|
| near-edge row of coils, in full sun | `a[0..n-1]` laid out contiguously, read forward |
| shadow-line row, "waiting their turn in a queue" | `b` laid out **reversed** (`rb[k]=b[n-1-k]`), so it faces the first row |
| walking the two rows together, coil against coil | sliding the reversed row past the forward row: step `d` compares `a[i-1]` vs `b[d-i-1]` for *all* `i` at once = one **anti-diagonal** of the DP grid |
| sun windows through two identical shapes | `_mm_cmpeq_epi8` on 16 character pairs at once |
| a small fire dropped on the water | `+1` in that lane; no fire = `-1` (`s = -1 - 2·mask`) |
| "I count the fires, nothing more — they are the only truth that survives" | only the score is kept; no traceback, no stored grid — three rolling anti-diagonals (`p2,p1,p0`), ~70 KB instead of a 576 MB table |
| the sunlit *line* (only one line is lit at a time) | the wavefront: memory is 3 vectors, not `(n+1)²` cells |

*Silent assumption broken:* "one pair of positions is judged at a time" **and** "both strings are read start to end in the same direction" (the shadow row is laid nose-to-tail against the sun, i.e. reversed, so a whole diagonal is judged in one instruction).

**SEED 2 — "a boundary door between the two rows is allowed to break exactly once per crossing, letting one coil crouch past a quarrel out of step"**

| World object | Problem object |
|---|---|
| a *crossing* | one complete alignment of `a` against `b` (a path from `(0,0)` to `(n,n)`) |
| the straight crossing (no broken door) | the pure diagonal alignment, score `2·m0 − n` |
| a quarrel | a mismatch position |
| the door between the rows breaking once | one gap event: one `−2` in `a` and its balancing `−2` in `b` (lengths are equal, so slips come in pairs) |
| "the next coil crouches forward, skipping the coil that quarreled" | offset-by-one comparison `a[k]` vs `b[k+1]` over the middle stretch |
| "I try this breaking at every place a quarrel happens" | **every** single-slip crossing `(p,q)` scored at once, in O(n), by prefix sums `E,F,G` + one running max — *no DP cell is ever computed first* |
| never twice in one crossing | the enumerated family is exactly `g = 1`: `score = 2M − n − 3` |

*Silent assumption broken:* **"a slip (gap) can only be discovered by having already compared the position before it."** The native discovers every slip site *before* any cell exists, from raw prefix counts of fires — the gap is found by arithmetic on match-indicator sums, not by reading a neighbour cell.

**SEED 3 — "every crossing's fire-pile is stored in its own room, the failed and twice-broken ones sink into an underground tower, and only the highest pile is brought back up"**

| World object | Problem object |
|---|---|
| a room holding one crossing's pile | one candidate lower bound `L` on the optimal score |
| the highest pile brought up from the tower | `L = max(straight, all single-slip A, all single-slip B)` |
| rows thrown into the tower "unread" | alignments never evaluated at all — the pruned region of the grid |
| "a second slip and the whole row is thrown down… I start that attempt over from the first coil" | the certificate that decides how far a row may slip before it is worthless: any path touching `|i−j| = m` scores at most `n − 5m`, so a path can only beat `L` if `m ≤ (n−L)/5` ⇒ door-allowance `w = (n−L)/5`, and the crossing is restarted **once**, band-wide |
| the queen's threshold, last coil | cell `(n,n)`, diagonal `d = 2n` |

*Silent assumption broken:* "the whole grid of every position against every other must be filled in." Most of the grid sinks into the tower unread.

## CHOSEN SEED

**SEED 2.** It is the one that breaks the preferred assumption — *a slip can only be discovered by having already compared the position before it* — and it does so completely literally: `best_pile()` finds, scores and maximises over **every** place a quarrel could let a coil crouch past, in a single O(n) pass over raw character comparisons, before a single DP cell exists. It is also the load-bearing part: the pile it hands up is what sets the door-allowance `w`, which is what lets the rest of the grid sink into the tower. SEEDs 1 and 3 are the same native's own surrounding machinery (the sunlit line he walks, the tower he brings the pile up from) and I keep them as he describes them rather than importing anything from the textbook.

## ASSUMPTION BROKEN

*"A slip (gap) can only be discovered by having already compared the position before it."*

Broken twice over, in the two directions the native works in:
1. **Ahead of the grid:** every single-slip crossing is scored from prefix sums of fire-counts (`E`, `F`, `G`) plus a running max — gaps are located and priced with no DP predecessor whatsoever.
2. **Across the grid:** on the sunlit line, the slip is not read from "the position before"; it is a *lane shift*. All 16 slips of a diagonal are taken simultaneously as two unaligned loads of the previous diagonal (`p1+i-1`, `p1+i`), with no lane knowing anything about its neighbour.

A third consequence: because the slip is priced before any comparison, the native knows *in advance* how far off-course a crossing may go (`w`), so the grid is never filled in.

## ARTIFACT

Which code implements which part of the native's mechanism:

* `best_pile()` — **SEED 2 + SEED 3's rooms.** The straight crossing (`m0`), then every single-slip crossing in both orientations (`bA`, `bB`: door broken by the shadow row, and by the near row), each one's pile in its own room, `L` = the highest pile brought up. O(n), no DP.
* `w = (n - L) / 5` — **the tower's verdict.** The door-allowance: proof that a row slipping further than `w` can never burn higher than the pile already in hand (`≤ n − 5(w+1) < L`). `w = 0` for identical cords, `w = n` for cords that agree no better than chance — this *is* the runtime regime detector, and it can never be wrong, only loose.
* `wave_avx2()` / `wave_scalar()` inner loop — **SEED 1.** One sunlit line at a time: `ca`/`cb` are the two rows walked coil against coil (`rb` = shadow row laid reversed), `cmpeq_epi8` is the sun windowing through, `s` is the fire (+1) or no fire (−1), `p2[i-1]+s` is the straight step, `max(p1[i-1],p1[i]) − 2` is the coil crouching one place past its turn on either side, `max` of the two is the highest pile for that room.
* `NEG` guard cells rewritten every diagonal — **the tower wall.** Out-of-band rows are unread, and the guards are refreshed (never allowed to accumulate) so nothing crawls back up out of the tower.
* `p0[0] = p0[d] = GAP*d` — **the queen's threshold and the far edge**: the two analytically known ends of each lit line.
* Guards demanded by step 4: `n < 48 → nw_rows` (sunlit-line setup not worth it on a short cord); `n > 12000 → wave_scalar` (int16 fires would overflow; the scalar wavefront keeps the identical mechanism, just wider counters, and is auto-vectorizable); `w` clamped to `[0,n]` so the worst case degrades to a full crossing, never to a wrong answer. **No thread parallelism**: the native's units of work are consecutive lit lines and each depends on the previous two — threads would be a lie about the metaphor as well as a slowdown.

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCHS    1
#define MISMATCHS (-1)
#define GAPS      (-2)

/* ---- short cords: plain rolling-row exact DP (no sunlit-line setup) ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int i, j, r;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (j = 0; j <= n; j++) prev[j] = GAPS * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = GAPS * i;
        for (j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? MATCHS : MISMATCHS);
            int u = prev[j] + GAPS;
            int l = cur[j - 1] + GAPS;
            int best = d > u ? d : u;
            if (l > best) best = l;
            cur[j] = best;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---- SEED 2: the straight crossing + EVERY single-slip crossing, O(n),   */
/*      scored with no DP cell in existence. Highest pile is brought up.    */
static int best_pile(int n, const char *a, const char *b)
{
    int i, m0 = 0, L;
    for (i = 0; i < n; i++) m0 += (a[i] == b[i]);
    L = 2 * m0 - n;                       /* straight crossing, door intact */
    if (n >= 2) {
        int e = 0, f = 0, g = 0;          /* E[q], F[q], G[q] */
        int rmF = 0, rmG = 0;             /* max_{p<=q} (E[p]-F[p]) / (E[p]-G[p]) */
        int bA = -1 << 28, bB = -1 << 28, q;
        for (q = 0; q < n; q++) {
            int eq1 = e + (a[q] == b[q]);          /* E[q+1] */
            int vA, vB;
            if (e - f > rmF) rmF = e - f;
            if (e - g > rmG) rmG = e - g;
            vA = (f - eq1) + rmF + m0;             /* matches, shadow row slips */
            vB = (g - eq1) + rmG + m0;             /* matches, near row slips   */
            if (vA > bA) bA = vA;
            if (vB > bB) bB = vB;
            e = eq1;
            if (q < n - 1) { f += (a[q] == b[q + 1]); g += (a[q + 1] == b[q]); }
        }
        if (2 * bA - n - 3 > L) L = 2 * bA - n - 3;   /* one broken door: -4 gaps, +1 column */
        if (2 * bB - n - 3 > L) L = 2 * bB - n - 3;
    }
    return L;
}

/* ---- SEED 1 (scalar): one sunlit line at a time, door-allowance w ---- */
static int wave_scalar(int n, const char *a, const char *b, int w)
{
    const int PAD = 64;
    const int sz = n + 1 + 2 * PAD;
    const int NEG = -(1 << 28);
    int *mem = (int *)malloc((size_t)3 * sz * sizeof(int));
    char *ab = (char *)malloc((size_t)n + 2 * PAD + 64);
    char *rb = (char *)malloc((size_t)n + 2 * PAD + 64);
    int *p2, *p1, *p0;
    int d, k, result = 0;
    if (!mem || !ab || !rb) { free(mem); free(ab); free(rb); return nw_rows(n, a, b); }
    for (k = 0; k < 3 * sz; k++) mem[k] = NEG;
    memset(ab, 0, (size_t)n + 2 * PAD + 64);
    memset(rb, 1, (size_t)n + 2 * PAD + 64);
    memcpy(ab, a, (size_t)n);
    for (k = 0; k < n; k++) rb[k] = b[n - 1 - k];   /* shadow row, laid reversed */
    p2 = mem + PAD; p1 = mem + sz + PAD; p0 = mem + 2 * sz + PAD;
    for (d = 0; d <= 2 * n; d++) {
        int i_min = (d > n) ? d - n : 0;
        int lo = (i_min > 1) ? i_min : 1;
        int hi = (d <= n) ? d - 1 : n;
        int l2 = (d - w >= 0) ? (d - w + 1) / 2 : 0;
        int h2 = (d + w) / 2;
        int nd = n - d, i;
        if (lo < l2) lo = l2;
        if (hi > h2) hi = h2;
#pragma omp simd
        for (i = lo; i <= hi; i++) {
            int sc = -1 + 2 * (ab[i - 1] == rb[nd + i]);   /* fire or no fire */
            int v  = p2[i - 1] + sc;                      /* straight step    */
            int u  = p1[i - 1];
            int l  = p1[i];
            int gg = (u > l ? u : l) + GAPS;              /* one coil crouches */
            p0[i] = v > gg ? v : gg;                      /* highest pile      */
        }
        for (k = 0; k < 32; k++) { p0[hi + 1 + k] = NEG; p0[lo - 1 - k] = NEG; }
        if (d <= n && d <= w) { p0[0] = GAPS * d; p0[d] = GAPS * d; }
        if (d == 2 * n) { result = p0[n]; break; }
        { int *t = p2; p2 = p1; p1 = p0; p0 = t; }
    }
    free(mem); free(ab); free(rb);
    return result;
}

#if defined(__AVX2__)
/* ---- SEED 1 (16 coils per glance): the sunlit line, int16 fires ---- */
static int wave_avx2(int n, const char *a, const char *b, int w)
{
    const int PAD = 64;
    const int sz = n + 1 + 2 * PAD;
    const int16_t NEG = -30000;
    int16_t *mem = (int16_t *)malloc((size_t)3 * sz * sizeof(int16_t));
    char *ab = (char *)malloc((size_t)n + 2 * PAD + 64);
    char *rb = (char *)malloc((size_t)n + 2 * PAD + 64);
    int16_t *p2, *p1, *p0;
    const __m256i vgap = _mm256_set1_epi16(2);
    const __m256i vm1  = _mm256_set1_epi16(-1);
    int d, k, result = 0;
    if (!mem || !ab || !rb) { free(mem); free(ab); free(rb); return nw_rows(n, a, b); }
    for (k = 0; k < 3 * sz; k++) mem[k] = NEG;
    memset(ab, 0, (size_t)n + 2 * PAD + 64);
    memset(rb, 1, (size_t)n + 2 * PAD + 64);   /* padding can never strike a fire */
    memcpy(ab, a, (size_t)n);
    for (k = 0; k < n; k++) rb[k] = b[n - 1 - k];
    p2 = mem + PAD; p1 = mem + sz + PAD; p0 = mem + 2 * sz + PAD;
    for (d = 0; d <= 2 * n; d++) {
        int i_min = (d > n) ? d - n : 0;
        int lo = (i_min > 1) ? i_min : 1;
        int hi = (d <= n) ? d - 1 : n;
        int l2 = (d - w >= 0) ? (d - w + 1) / 2 : 0;
        int h2 = (d + w) / 2;
        int nd = n - d, i;
        if (lo < l2) lo = l2;
        if (hi > h2) hi = h2;
        for (i = lo; i <= hi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(ab + i - 1));
            __m128i cb = _mm_loadu_si128((const __m128i *)(rb + nd + i));
            __m256i msk = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i s   = _mm256_sub_epi16(vm1, _mm256_add_epi16(msk, msk));
            __m256i vd  = _mm256_add_epi16(
                              _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), s);
            __m256i vu  = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i vl  = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i vg  = _mm256_sub_epi16(_mm256_max_epi16(vu, vl), vgap);
            _mm256_storeu_si256((__m256i *)(p0 + i), _mm256_max_epi16(vd, vg));
        }
        for (k = 0; k < 32; k++) { p0[hi + 1 + k] = NEG; p0[lo - 1 - k] = NEG; }
        if (d <= n && d <= w) { p0[0] = (int16_t)(GAPS * d); p0[d] = (int16_t)(GAPS * d); }
        if (d == 2 * n) { result = p0[n]; break; }
        { int16_t *t = p2; p2 = p1; p1 = p0; p0 = t; }
    }
    free(mem); free(ab); free(rb);
    return result;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    int L, w;
    if (n <= 0) return 0;
    if (n < 48) return nw_rows(n, a, b);          /* guard: short cord */
    L = best_pile(n, a, b);                        /* SEED 2 + SEED 3 rooms */
    w = (n - L) / 5;                               /* door-allowance (certified) */
    if (w < 0) w = 0;
    if (w > n) w = n;                              /* guard: full crossing */
#if defined(__AVX2__)
    if (n <= 12000) return wave_avx2(n, a, b, w);  /* guard: int16 range */
#endif
    return wave_scalar(n, a, b, w);
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 12

Reasoning behind the number, stated before any measurement: the reference does ~1 cell per 2–4 cycles and, at `n ≳ 2000`, streams an `(n+1)²·4`-byte table (67 MB at n=4096, 576 MB at n=12000) so it becomes bandwidth-bound. The sunlit line does 16 cells per ~15 instructions out of a ~70 KB working set. Against that I expect ~8–20× on random DNA, of which ~1.5–2× is the tower (band area `n² − (n−w)² ≈ 0.51n²` when `w ≈ 0.3n`, which is all the certificate can promise for sequences that agree no better than chance) and the rest is the 16-wide glance plus the collapsed working set. On near-identical cords `w → 0` and the work is O(n), so the speedup there should be one to three orders of magnitude, not 12. I am quoting 12 as the random-DNA figure, since that is the harder and more likely benchmark regime.

## MEASUREMENT

**Not measured.** `alignment_bench` and `alignment_contract` were not available in this session — no tools were exposed to me at all — so I could neither compile nor time this, and I am not going to dress up an estimate as a measurement. What I can state honestly:

* **Verified by hand, not by machine:** the index algebra of the wavefront (`p2[i-1]` = `(i-1,j-1)`, `p1[i-1]` = `(i-1,j)`, `p1[i]` = `(i,j-1)`, `rb[n-d+i]` = `b[j-1]`), the fire encoding (`-1-2·mask` ∈ {+1,−1}), the boundary values `−2d` at both ends of each lit line, the buffer bounds (all reads within `±PAD=64`, overshoot lanes always overwritten by guards before being read), and the int16 range (`|value| ≤ 2n ≤ 24000 < 30000 = |NEG|` for the gated `n ≤ 12000`).
* **The one thing that must be checked first when it does run:** exactness. The certificate is a proof, not a heuristic — any path touching `|i−j| = m` scores at most `min(i,j)+min(n−i,n−j)−4m = n−5m`, and `L` is the score of a real alignment, so `w = ⌊(n−L)/5⌋` satisfies `n−5(w+1) < L ≤ S*` and the optimum cannot leave the band. If the bench reports a mismatch against the reference, the fault is in my wavefront indexing or guard refresh, **not** in the band, and that is where to look.
* Expected regime split to look for in the numbers: near-identical inputs → `w` small → runtime linear in `n`; random inputs → `w ≈ 0.3n` → runtime ~`0.5n²/16` cell-updates.

## VERDICT

The native's mechanism survives translation intact and, unusually, gives up something the textbook method does not have: a *price for a slip computed before any comparison exists*, which turns into a proof about where the grid may be left unread. That is SEED 2 doing real work, not decoration.

Two honest limitations, both already addressed in the code rather than left as caveats:

1. **The tower is nearly empty when the cords disagree.** For random DNA the optimal score is ≈ `−0.5n`, so the certificate can only promise `w ≈ 0.3n` — the band saves ~2× of the area, not the 10× one might hope. This is a property of the scoring (a gap pair costs 3 net, and at 25 % identity the diagonal is already bad), not of my implementation, and no tightening of `L` fixes it, because the bound is limited by `S*` itself. I did not hide this behind a hopeful band width: `w` is computed, clamped to `n`, and the full crossing is walked when that is what the proof demands.
2. **The one place my own risk statement bit, and what I did about it.** The native's literal restart rule ("a second slip and the row is thrown into the tower, I start over from the first coil") is an *iterative widening*, and widening by doubling would cost up to an extra full crossing — a 2× regression exactly in the random-DNA regime that matters most. Rather than ship that, I kept the tower but paid for it once: `best_pile()` fills the rooms in O(n) and the allowance `w` is fixed **before** the first crossing, so no crossing is ever discarded. Same mechanism, zero wasted rows. Likewise `n < 48` falls back to plain row DP and `n > 12000` falls back to the scalar wavefront rather than overflowing int16, so neither stated risk ships unguarded.

If the measurement comes back below ~3×, the most likely cause is that the benchmark `n` is small enough (a few hundred) that the reference's table stays in L2 and the whole advantage collapses to raw SIMD width against short, badly-filled diagonals — in which case the honest fix inside this mechanism is to raise the `nw_rows` cutoff, not to reach for a different algorithm.