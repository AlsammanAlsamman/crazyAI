# MAPPING

**SEED 1 — the low sun and the doubled shadow**

| world object | problem object |
|---|---|
| two twisted grass-ropes, knots dyed in 4 berry-colors | the two length-`n` strings over {A,C,G,T} |
| rope hung along the **east wall** | `a`, indexed by `i`, read forward |
| rope stretched **from the doorway across the floor at a slant** | `b`, laid along the anti-diagonal — i.e. read *backwards* relative to `a` (buffer `pb[k] = b[n-1-k]`) |
| the empty room, no tiles ever laid | the (n+1)² DP grid, never materialized |
| a crossing | cell (i,j) |
| the low sun through the one window, at the hour it always comes | **one** `cmpeq_epi8` issued once per anti-diagonal |
| one doubled shadow instead of two | the lane mask: 16 crossings judged by a single instruction |
| treasure / lit tile | `a[i-1]==b[j-1]`, substitution `+1` |
| the cat, answering no one | the branchless `max` — no test, no branch, it just sits where the compare fired |

*Breaks:* "one pair of positions is judged at a time" **and** "both strings are read start to end in the same direction". Laying the second rope at a slant is precisely what makes both ropes *contiguous* under one vector load, because on `i+j = d` the `a`-index rises while the `b`-index falls.

**SEED 2 — the cost of the bare floor is read off the ropes, never walked**

| world object | problem object |
|---|---|
| bare floor between two treasures | the cells that cannot lie on any optimal path |
| "how many knots forward on the wall-rope, how many on the floor-rope" | pure index arithmetic on (i,j) |
| counting doorframes instead of pacing the distance | closed-form accounting: every alignment of two equal-length strings satisfies **score = 2M − n − 3g**, with `M` matches and `g` indel-pairs |
| "the corridor's cost was never a secret — only its length was" | the recurrence is known; the only unknown is **how wide a corridor can hold an optimal path** |
| "and the ropes already told me that" | the ropes' own doubled shadows on the main diagonal, `M0 = #{i : a[i]==b[i]}`, pin that width: since `S* ≥ 2M0 − n` and any path with `g` slips scores `≤ n − 5g`, every optimal path obeys `g ≤ ⌊2(n−M0)/5⌋ =: W`, and `\|i−j\| ≤ g ≤ W` everywhere along it |

*Breaks:* **"the whole grid of every position against every other must be filled in."** The floor outside `|i−j| ≤ W` is provably empty and is never touched.

**SEED 3 — keep only the cheapest tally, drop the rest**

| world object | problem object |
|---|---|
| running tally arriving at a treasure | a dp value arriving at a cell |
| "slip spent, or not yet spent" | the two-state affine bookkeeping — which **collapses** here, because the gap is linear (−2 per knot), so one tally per crossing suffices |
| letting the worse tally fall like a bad thread-end | `max()`; and, structurally, that only **three** anti-diagonals stay alive — every older one is dropped |

*Breaks:* the grid assumption in the *memory* sense only (O(n) instead of O(n²)); it does not reduce the *work*.

# CHOSEN SEED

**SEED 2**, the one that reads the corridor's length off the ropes. It is the only seed that breaks the privileged assumption ("the whole grid must be filled in") in the sense of *work*, not just storage, and it is the furthest from Needleman–Wunsch: the textbook method has no notion that the ropes themselves announce how narrow the room is.

SEED 1 is not discarded — it is the *mechanism* SEED 2 needs (the doubled-shadow count `M0` is exactly what sets the corridor width, and the same sun-strike drives the inner loop). SEED 3 supplies the three-diagonal rolling window.

# ASSUMPTION BROKEN

Primary: **"the whole grid of every position against every other must be filled in."** Only `|i−j| ≤ W` is filled, with `W` proved sufficient from `M0` before a single cell is computed.

Secondary: **"both strings are read start to end in the same direction"** (`b` is reversed once, so the anti-diagonal wavefront reads both ropes as contiguous forward windows), and **"one pair of positions is judged at a time"** (16 dyes compared per instruction).

Not broken: the dependency structure itself. Every cell still equals the reference's `max(diag+s, up−2, left−2)`; I refuse to approximate it. The native's "allow the ropes to slip past each other **once**" is a *heuristic restriction* and would be wrong in general — I keep the exact recurrence and use the native's slip-counting only where it yields a **proof** (the bound on `g`).

**Computational mapping.** Memory = three `int16` rows, the only thing kept; the floor is never tiled. What flows = the anti-diagonal wavefront. What stays still = the two padded rope buffers. Processor = one SIMD lane per crossing. Time = `d = i+j`, the hour of the sun; exactly `2n+1` ticks.

**Regime recognition (in-world).** The cat counts the doubled shadows on the main diagonal *first*. Many treasures on the straight line ⇒ narrow corridor ⇒ tiny band. Few ⇒ wide corridor, and when `W ≥ n` the very same code degenerates to the full grid, so the "wide" regime needs no second kernel. Two further regimes are guarded explicitly: a tiny room (`n < 64`) gets the plain scalar DP with no setup at all, and a room too large for 16-bit tallies (`n > 15000`) or a machine with no sun (no SSE2) falls back to a two-row scalar NW.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) \
    && (defined(__SSE2__) || defined(__AVX2__) || defined(_M_X64))
#  include <immintrin.h>
#  define KSIMD 1
#else
#  define KSIMD 0
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* ---------- fallback: exact two-row Needleman-Wunsch, O(n) memory ---------- */
static int nw_rows(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    int *mem = (int *)malloc((size_t)2 * ((size_t)n + 1) * sizeof(int));
    if (!mem) return 0;
    int *prev = mem, *cur = mem + (n + 1);
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        const char ai = a[i - 1];
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int dg = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int up = prev[j] + GAP;
            int lf = cur[j - 1] + GAP;
            int bst = dg;
            if (up > bst) bst = up;
            if (lf > bst) bst = lf;
            cur[j] = bst;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(mem);
    return r;
}

#if KSIMD
/* ---------- the sun-strike: 16 (or 8) crossings judged at once ---------- */
#if defined(__AVX2__)
#  define VW 16
   typedef __m256i kvec;
#  define KLOAD(p)     _mm256_loadu_si256((const __m256i *)(const void *)(p))
#  define KSTORE(p,x)  _mm256_storeu_si256((__m256i *)(void *)(p), (x))
#  define KADD(x,y)    _mm256_add_epi16((x),(y))
#  define KSUB(x,y)    _mm256_sub_epi16((x),(y))
#  define KMAX(x,y)    _mm256_max_epi16((x),(y))
#  define KSET1(v)     _mm256_set1_epi16((short)(v))
static inline kvec kscore(const char *pa, const char *pb)
{
    __m128i ca = _mm_loadu_si128((const __m128i *)(const void *)pa);
    __m128i cb = _mm_loadu_si128((const __m128i *)(const void *)pb);
    __m128i m8 = _mm_cmpeq_epi8(ca, cb);          /* one doubled shadow per byte */
    __m256i m  = _mm256_cvtepi8_epi16(m8);        /* -1 = same dye, 0 = different */
    return _mm256_add_epi16(_mm256_add_epi16(m, m), _mm256_set1_epi16(1)); /* +1 / -1 */
}
#else
#  define VW 8
   typedef __m128i kvec;
#  define KLOAD(p)     _mm_loadu_si128((const __m128i *)(const void *)(p))
#  define KSTORE(p,x)  _mm_storeu_si128((__m128i *)(void *)(p), (x))
#  define KADD(x,y)    _mm_add_epi16((x),(y))
#  define KSUB(x,y)    _mm_sub_epi16((x),(y))
#  define KMAX(x,y)    _mm_max_epi16((x),(y))
#  define KSET1(v)     _mm_set1_epi16((short)(v))
static inline kvec kscore(const char *pa, const char *pb)
{
    __m128i ca = _mm_loadl_epi64((const __m128i *)(const void *)pa);
    __m128i cb = _mm_loadl_epi64((const __m128i *)(const void *)pb);
    __m128i m8 = _mm_cmpeq_epi8(ca, cb);
    __m128i m  = _mm_unpacklo_epi8(m8, m8);       /* 0xFFFF = same dye */
    return _mm_add_epi16(_mm_add_epi16(m, m), _mm_set1_epi16(1));
}
#endif

/* ---------- banded anti-diagonal wavefront; exact, never tiles the floor ----------
   Correctness of the band: for equal-length a,b any alignment has
       #deletions = #insertions = g,  #aligned pairs = n-g,
       score = 2M - n - 3g   and   M <= n-g   =>   score <= n - 5g.
   The straight-diagonal alignment scores L = 2*M0 - n, so the optimum S* >= L,
   hence every optimal alignment has g <= (2n - 2*M0)/5 = W, and along its path
   |i-j| = |#del so far - #ins so far| <= g <= W.  A DP restricted to |i-j| <= W
   therefore contains an optimal path and returns S* exactly.                     */
static int nw_band(int n, const char *a, const char *b, int W)
{
    const short NEG = -32000;             /* < -2n for n <= 15000; -2n is the true floor */
    const int   PAD = 64;
    const size_t row  = (size_t)n + 3 + 2 * (size_t)PAD;
    const size_t clen = (size_t)n + 2 * (size_t)PAD;

    unsigned char *blk =
        (unsigned char *)malloc(3 * row * sizeof(short) + 2 * clen + 64);
    if (!blk) return nw_rows(n, a, b);

    short *A0 = (short *)(void *)blk;
    short *A1 = A0 + row;
    short *A2 = A1 + row;
    char  *pa = (char *)(void *)(A2 + row);
    char  *pb = pa + clen;

    memset(A0, 0, 3 * row * sizeof(short));
    memset(pa, 0x01, clen);
    memset(pb, 0x02, clen);              /* pads can never match each other */
    memcpy(pa, a, (size_t)n);
    for (int k = 0; k < n; k++) pb[k] = b[n - 1 - k];   /* the slanted rope */

    short *p2 = A0, *p1 = A1, *cu = A2;
    const kvec vg = KSET1(2);

    for (int d = 0; d <= 2 * n; d++) {               /* d = the hour of the sun */
        int t0 = d - W;
        int lo = (t0 <= 0) ? 0 : ((t0 + 1) >> 1);    /* ceil((d-W)/2) */
        int hi = (d + W) >> 1;                       /* floor((d+W)/2) */
        if (lo < d - n) lo = d - n;
        if (lo < 0)     lo = 0;
        if (hi > d)     hi = d;
        if (hi > n)     hi = n;

        int ilo = (lo < 1) ? 1 : lo;
        int ihi = (hi < d - 1) ? hi : d - 1;

        if (ihi >= ilo) {
            const char  *qa = pa + (ilo - 1);              /* wall rope, forward  */
            const char  *qb = pb + (n - d + ilo);          /* floor rope, forward */
            const short *sd = p2 + (ilo - 1);              /* dp[i-1][j-1] */
            const short *su = p1 + (ilo - 1);              /* dp[i-1][j]   */
            const short *sl = p1 + ilo;                    /* dp[i][j-1]   */
            short *dst = cu + ilo;
            int len = ihi - ilo + 1;
            for (int t = 0; t < len; t += VW) {
                kvec s  = kscore(qa + t, qb + t);
                kvec vd = KADD(KLOAD(sd + t), s);
                kvec vu = KSUB(KLOAD(su + t), vg);
                kvec vl = KSUB(KLOAD(sl + t), vg);
                KSTORE(dst + t, KMAX(KMAX(vd, vu), vl));   /* drop the bad thread-ends */
            }
        }
        if (lo == 0) cu[0] = (short)(-2 * d);              /* dp[0][d] */
        if (hi == d) cu[d] = (short)(-2 * d);              /* dp[d][0] */
        if (lo >= 1) { cu[lo - 1] = NEG; if (lo >= 2) cu[lo - 2] = NEG; }
        cu[hi + 1] = NEG;
        cu[hi + 2] = NEG;

        short *t = p2; p2 = p1; p1 = cu; cu = t;           /* keep only three rows */
    }
    int r = (int)p1[n];
    free(blk);
    return r;
}
#endif /* KSIMD */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    /* regime 1: a tiny room -- no setup, no vectors, no allocation */
    if (n < 64) {
        int bufA[66], bufB[66];
        int *prev = bufA, *cur = bufB;
        for (int j = 0; j <= n; j++) prev[j] = j * GAP;
        for (int i = 1; i <= n; i++) {
            const char ai = a[i - 1];
            cur[0] = i * GAP;
            for (int j = 1; j <= n; j++) {
                int dg = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
                int up = prev[j] + GAP;
                int lf = cur[j - 1] + GAP;
                int bst = dg;
                if (up > bst) bst = up;
                if (lf > bst) bst = lf;
                cur[j] = bst;
            }
            int *t = prev; prev = cur; cur = t;
        }
        return prev[n];
    }

#if KSIMD
    /* regime 2: the ropes announce the corridor width, then the sun does the work.
       16-bit tallies are valid while 2n < 32000.                                  */
    if (n <= 15000) {
        int m0 = 0;
        for (int i = 0; i < n; i++) m0 += (a[i] == b[i]);   /* doubled shadows */
        long long W = (2LL * (long long)(n - m0)) / 5 + 4;  /* proof + safety margin */
        if (W > n) W = n;                                   /* W == n => full grid */
        if (W < 8) W = 8;
        if (W > n) W = n;
        return nw_band(n, a, b, (int)W);
    }
#endif

    /* regime 3: too large for 16-bit tallies, or no SIMD at all */
    return nw_rows(n, a, b);
}
```

# PREDICTION

PREDICTION: speedup_vs_dp = 12

Reasoning behind the number, stated before any measurement: ~8× from 16-wide `int16` lanes on the anti-diagonal (the reference's row-wise loop carries a dependency through `dp[i][j-1]` and cannot vectorize at all), times ~1.7× from the band on unrelated DNA (`M0 ≈ n/4 ⇒ W ≈ 0.3n ⇒ ~0.5n²` cells instead of `n²`), plus a modest extra factor from writing 6 KB of rolling rows instead of streaming a 4n² -byte table. On *similar* sequence pairs (`M0 ≈ 0.9n ⇒ W ≈ 0.04n`) the band alone is ~12× and the total should land far above 12 — I expect the measured number to swing hard with the benchmark's sequence identity, and I am predicting for the unrelated-DNA case, which is the pessimistic one.

# MEASUREMENT

**Not performed.** No tools were available in this session — `alignment_bench` and `alignment_contract` could not be called, so I have no measured speedup and no independent correctness check to report. I will not dress up an estimate as a measurement. The prediction above stands as written, before any data, and the pipeline's numbers should be taken over it wherever they disagree.

What I *did* verify, by hand rather than by tool: the full trace of `n=2, a="AC", b="AC"` through `nw_band` (d = 0…4), which reproduces `dp[1][1]=1`, `dp[1][2]=−1`, `dp[2][1]=−1`, `dp[2][2]=2`; and the index-range argument that on every diagonal `d` the reads `p1[ilo−1 … ihi]` and `p2[ilo−1 … ihi−1]` land either on cells that were genuinely written on diagonals `d−1`/`d−2` or on the `NEG` sentinels written one and two positions outside each band edge — never on unwritten memory. That is an argument, not evidence.

# VERDICT

The native's idea survives translation without being replaced by the textbook method, and it survives in a strong form: *the corridor's cost was never a secret — only its length was, and the ropes already told me that* is literally true here. `M0`, a single O(n) pass of doubled shadows, yields a **provable** half-width `W = ⌊2(n−M0)/5⌋`, and everything outside it is skipped without ever being evaluated. That is the grid-filling assumption broken outright, not softened.

Two things I refused to take literally, and I want that on the record rather than buried:

1. **"Allowing the ropes to slip past each other once."** Taken at face value this restricts alignments to a single indel block, which is simply not the same function as the contract's. I kept the exact recurrence and used the native's slip-counting only where it produces a theorem (`score ≤ n − 5g`), not a shortcut.
2. **"The treasures alone don't excuse me from the corridor's cost."** Correct, and it is the reason a match-chaining / sparse-DP kernel would have been wrong. I did not build one.

Every condition under which this could lose is guarded rather than merely noted, as required:
- **small-`n` overhead** → `n < 64` takes a stack-only scalar DP with no allocation and no vector setup;
- **16-bit tally overflow** → `n > 15000` falls back to the exact two-row scalar NW (values are bounded by `−2n`, and `NEG = −32000` sits below that only while `n ≤ 15000`);
- **no SIMD available** → `#if KSIMD` routes to the same scalar fallback;
- **band too wide to help** (unrelated or adversarial input, `M0 = 0`) → `W` clamps to `n` and the *identical* code path degenerates to the full grid, so the worst case is the plain SIMD wavefront, never worse than it.

No thread parallelism. The metaphor's unit of work is one anti-diagonal, and there are `2n+1` of them in strict sequence; threading would buy `2n+1` barriers against a few hundred cells of work each. At the sizes this contract plausibly runs, that is a loss, so I left it out rather than ship a guarded mechanism I expect to be dead code.

The honest residual risk is the band index arithmetic — the ceil/floor edges and the sentinel placement are where a bug would hide, and a band bug produces a *wrong score*, not a slow one. The `+4` margin on `W` absorbs an off-by-one in the bound but not an off-by-one in the indexing. If the pipeline reports a mismatch against the reference rather than a slowdown, that is where it is, and the correct repair is to raise the `W < 8` floor to `W = n` (full grid) and re-measure the pure wavefront in isolation.