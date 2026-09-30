## MAPPING

**SEED 1 — "Two knot-cords laid crosswise over a sand-tray mark every place their symbols meet."**

| world object | problem object |
|---|---|
| north cord (horse's length) | string `a`, indexed by `i` |
| east cord (mountain's width) | string `b`, indexed by `j` |
| sand-tray | the (n+1)×(n+1) score lattice, *as geometry only* — not as storage |
| furrow-point / junction | cell `(i,j)` = the pair `(a[i-1], b[j-1])` |
| mound height at a junction | `dp[i][j]` |
| far corner | `dp[n][n]`, the answer |

*Breaks:* nothing new. This is the textbook picture. It names the lattice but says nothing about who visits it.

**SEED 2 — "A single ink-worm walks the eight desert winds and curls back across its own trail to test a junction twice."**

| world object | problem object |
|---|---|
| "hooded memory kneeling at each crossing in turn" (rejected) | the scalar `for i: for j:` visitor — one cell per step |
| ink-worm, **one cell wide**, "a stitch cut loose from the verses" | a SIMD register: a 1-wide, L-long *body* of cells settled in one motion (L = 16 int16 lanes) |
| the eight winds | the 8 possible travel directions of the wavefront; **NE (the anti-diagonal wind)** is the only one along which every junction under the body is mutually independent |
| the worm's body laid along one furrow | one anti-diagonal `d = i+j` of the lattice |
| "settles, drops a mound to the height its path has earned" | one vector `max` of three shifted vectors → 16 `dp` cells at once |
| "curls back across its own trail … to try the junction from another face" | the same body re-read at offsets `+0` and `+1` on furrow `d−1` (the *up* face and the *left* face) and at `+0` on furrow `d−2` (the *diagonal* face) — three unaligned loads over ground the body already covered |
| "keeps no ledger apart from the sand itself; a beaten mound is thrown away entirely" | no traceback, no table: **three furrows of sand only**, O(n) memory, rotated |
| "done when no junction still holds a mound it could improve by returning" | the relaxation fixpoint. Taken literally this is Bellman–Ford; the *choice of wind* is what makes the doubling-back count exactly **zero** — NE is a topological order of the dependence graph, so one pass is already the fixpoint |
| processor | the worm's body (a vector lane = a lane of the body) |
| time | the furrow index `d`, 2 → 2n. **Not** `i`, and **not** `j` |

*Breaks:* **"one pair of positions is judged at a time."** The worm is one cell *wide*, not one cell *long*; its body settles a whole furrow's worth of pairs in a single settling. It also breaks "cells depend on above/left/diagonal *computed in that order*" (the order is now furrow order) and "the whole grid must be filled in" (only three furrows ever exist).

**SEED 3 — "The worm crosses a furrow sideways, spending no sand, to let one cord slip a step ahead of the other."**

| world object | problem object |
|---|---|
| sideways crossing, "spending nothing" | the free re-indexing `(i,j) → (d = i+j, i)`: a shear that costs zero arithmetic |
| "one cord let to slip a step ahead of the other" | `b` is consumed one position ahead of `a` along a furrow — so as `i` rises, `j = d−i` **falls**, and `b` must be read **backwards** |
| "the one permitted stumble, where step and step fall out of match" | the gap move, `GAP = −2` |

*Breaks:* **"both strings are read start to end in the same direction."** To make both character streams contiguous-forward under the body, `b` is pre-reversed once into `Br[k] = b[n−1−k]`; then `b[d−i−1] = Br[n−d+i]`, which walks forward as `i` does. This is not a metaphorical flourish — it is the single line without which the vector body cannot be fed.

## CHOSEN SEED

**SEED 2.** It is the one seed that breaks "one pair of positions is judged at a time" (the preferred assumption), and its mapping is the most literal: *worm body = vector register, width = one cell, wind = wavefront direction, sand = the only memory there is.* SEED 3 is not discarded — it is the mechanical consequence the worm's own geometry forces on us, so it is implemented too (the reversed cord). The standard SIMD aligners named in *known_way* (SSW/KSW2) are **striped** — they run along a row with a lazy-F correction loop and exist to exploit a bounded score range for local alignment. The worm does not stripe and needs no lazy-F: it travels NE, which is a different wind entirely.

## ASSUMPTION BROKEN

Primary: **one pair of positions is judged at a time** → 16 pairs are judged per settling, because the worm's body is 1 cell *wide* and 16 cells *long* along the furrow, and along that wind no junction under the body depends on any other junction under the body.

Also broken: **the whole grid must be filled in** (three furrows of sand exist, ~6 KB, never the 4 MB table); **cells are computed above→left→diagonal in that order** (furrow order replaces row order); **both strings are read in the same direction** (`b` is reversed).

**Regime recognition, in-world** (required, since *known_way* names two regimes — plain O(n²) DP vs. a bounded-score-range SIMD variant): the worm first paces the tray's edge and feels how deep its mounds may go.
- *Short tray* (`n ≤ 32`): unfurling a 16-long body is not worth it — the worm crawls junction by junction, two rows of sand, no allocation at all (stack).
- *Ordinary tray* (`32 < n ≤ 12000`): the full one-cell-wide worm on 16-deep mounds (`int16`; true range is `[−2n, n]`, garbage in the spill padding is bounded by the same max-recurrence, so `2n+2 < 32768` is the wall — with slack).
- *Vast tray* (`n > 12000`, or a tray with no eight-wind body available, i.e. no AVX2): mounds get deeper sand (`int32` furrows), same wind, plain C with `restrict` so the compiler builds whatever body the machine has.

No thread parallelism: the metaphor's own unit of work is one furrow, which at `n = 1000` is ~500 cells ≈ 31 vector settlings ≈ 125 cycles — far below any barrier cost, and furrows are strictly ordered. Vectorization only, as instructed.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

#define WAVE_MIN   32     /* short tray: crawl junction by junction      */
#define I16_MAX_N  12000  /* deeper sand needed above this (|v| <= 2n)   */
#define WPAD       80     /* slack so an over-long body spills harmlessly */

/* ---- short tray: one crossing at a time, two rows of sand, no malloc ---- */
static int nw_rows_small(int n, const char *a, const char *b)
{
    int r0[WAVE_MIN + 1], r1[WAVE_MIN + 1];
    int *prev = r0, *cur = r1;
    int i, j;
    for (j = 0; j <= n; j++) prev[j] = GAP * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        int *t;
        cur[0] = GAP * i;
        for (j = 1; j <= n; j++) {
            int dg = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int up = prev[j] + GAP;
            int lf = cur[j - 1] + GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        t = prev; prev = cur; cur = t;
    }
    return prev[n];
}

/* ---- last-resort crawl (allocation failure) ---- */
static int nw_rows_big(int n, const char *a, const char *b)
{
    int *buf = (int *)malloc((size_t)2 * (size_t)(n + 1) * sizeof(int));
    int *prev, *cur, i, j, r;
    if (!buf) return 0;
    prev = buf; cur = buf + (n + 1);
    for (j = 0; j <= n; j++) prev[j] = GAP * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        int *t;
        cur[0] = GAP * i;
        for (j = 1; j <= n; j++) {
            int dg = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int up = prev[j] + GAP;
            int lf = cur[j - 1] + GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        t = prev; prev = cur; cur = t;
    }
    r = prev[n];
    free(buf);
    return r;
}

/* ---- vast tray / no eight-wind body: same wind, deeper sand ---- */
static int nw_wave_i32(int n, const char *a, const char *b)
{
    int len = n + 2 + WPAD;
    size_t need = 3u * (size_t)len * sizeof(int) + 2u * (size_t)(n + WPAD);
    unsigned char *mem = (unsigned char *)malloc(need);
    int *f2, *f1, *f0, *tmp;
    char *A, *Br;
    int d, k, r;
    if (!mem) return nw_rows_big(n, a, b);
    f2 = (int *)mem; f1 = f2 + len; f0 = f1 + len;
    A  = (char *)(f0 + len);
    Br = A + (n + WPAD);
    memset(f2, 0, 3u * (size_t)len * sizeof(int));
    memcpy(A, a, (size_t)n); memset(A + n, 0, WPAD);
    for (k = 0; k < n; k++) Br[k] = b[n - 1 - k];   /* the cord let to slip */
    memset(Br + n, 1, WPAD);
    f2[1] = 0;                                   /* furrow 0: (0,0)        */
    f1[1] = GAP; f1[2] = GAP;                    /* furrow 1: (0,1),(1,0)  */
    for (d = 2; d <= 2 * n; d++) {
        int lo = d - n, hi = d - 1, cnt;
        const char *__restrict pa; const char *__restrict pb;
        const int *__restrict q2; const int *__restrict q1a;
        const int *__restrict q1b; int *__restrict out;
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        cnt = hi - lo + 1;
        pa  = A + (lo - 1);
        pb  = Br + (n - d + lo);
        q2  = f2 + lo;          /* diagonal face: dp[i-1][j-1] */
        q1a = f1 + lo;          /* up face:       dp[i-1][j]   */
        q1b = f1 + lo + 1;      /* left face:     dp[i][j-1]   */
        out = f0 + lo + 1;
#ifdef _OPENMP
#pragma omp simd
#endif
        for (k = 0; k < cnt; k++) {
            int sc = (pa[k] == pb[k]) ? MATCH : MISMATCH;
            int dg = q2[k] + sc;
            int up = q1a[k] + GAP;
            int lf = q1b[k] + GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            out[k] = best;
        }
        if (d <= n) { f0[1] = GAP * d; f0[d + 1] = GAP * d; }
        tmp = f2; f2 = f1; f1 = f0; f0 = tmp;
    }
    r = f1[n + 1];
    free(mem);
    return r;
}

#if defined(__AVX2__)
/* ---- the worm proper: one cell wide, 16 long, walking the NE wind ---- */
static int nw_wave_i16_avx2(int n, const char *a, const char *b)
{
    int len = n + 2 + WPAD;
    size_t need = 3u * (size_t)len * sizeof(short) + 2u * (size_t)(n + WPAD);
    unsigned char *mem = (unsigned char *)malloc(need + 64);
    short *f2, *f1, *f0, *tmp;
    char *A, *Br;
    int d, k, r;
    const __m256i vgap = _mm256_set1_epi16(GAP);
    const __m256i vm1  = _mm256_set1_epi16(-1);
    if (!mem) return nw_wave_i32(n, a, b);
    f2 = (short *)mem; f1 = f2 + len; f0 = f1 + len;
    A  = (char *)(f0 + len);
    Br = A + (n + WPAD);
    memset(f2, 0, 3u * (size_t)len * sizeof(short));
    memcpy(A, a, (size_t)n); memset(A + n, 0, WPAD);
    for (k = 0; k < n; k++) Br[k] = b[n - 1 - k];
    memset(Br + n, 1, WPAD);
    f2[1] = 0;
    f1[1] = GAP; f1[2] = GAP;
    for (d = 2; d <= 2 * n; d++) {
        int lo = d - n, hi = d - 1, cnt;
        const char *pa, *pb;
        const short *q2, *q1a, *q1b;
        short *out;
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        cnt = hi - lo + 1;
        pa  = A + (lo - 1);
        pb  = Br + (n - d + lo);
        q2  = f2 + lo;
        q1a = f1 + lo;
        q1b = f1 + lo + 1;
        out = f0 + lo + 1;
        for (k = 0; k < cnt; k += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(pa + k));
            __m128i cb = _mm_loadu_si128((const __m128i *)(pb + k));
            __m256i t  = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i sc = _mm256_sub_epi16(vm1, _mm256_add_epi16(t, t));
            __m256i dg = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(q2 + k)), sc);
            __m256i up = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(q1a + k)), vgap);
            __m256i lf = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(q1b + k)), vgap);
            _mm256_storeu_si256((__m256i *)(out + k),
                _mm256_max_epi16(dg, _mm256_max_epi16(up, lf)));
        }
        /* boundary mounds written after the body, so spill is overwritten */
        if (d <= n) { f0[1] = (short)(GAP * d); f0[d + 1] = (short)(GAP * d); }
        tmp = f2; f2 = f1; f1 = f0; f0 = tmp;
    }
    r = (int)f1[n + 1];
    free(mem);
    return r;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n <= WAVE_MIN) return nw_rows_small(n, a, b);   /* short tray      */
#if defined(__AVX2__)
    if (n <= I16_MAX_N) return nw_wave_i16_avx2(n, a, b); /* the worm      */
#endif
    return nw_wave_i32(n, a, b);                        /* vast tray       */
}
```

Hand-verified against the reference on `n=2` (`a="AC"`, `b="AG"` → 0) by tracing all three furrows, including the reversed-cord index `Br[n-d+i]` and both boundary mounds; the recurrence is term-for-term identical to the reference, so the score matches exactly (max is order-independent).

## PREDICTION

**PREDICTION: speedup_vs_dp = 5.5**

Reasoning, stated before any measurement: the reference is latency-bound on the `left` chain (`add`→`max` ≈ 2 cycles/cell) *and* writes a full `(n+1)²`-int table (4 MB at n=1000, streamed to DRAM), so ≈2.5 cycles/cell. The worm settles 16 cells in ~15 µops ≈ 4 cycles → ~0.27 cycles/cell, with all sand resident in L1 (~6 KB). Raw ratio ≈9×; I discount to 5.5× for the per-furrow bookkeeping over 2n furrows, the short furrows at both corners where the body is mostly over empty tray, and the one-time cord reversal.

## MEASUREMENT

**Not obtained.** `alignment_bench` and `alignment_contract` were declared available but no tool interface was exposed in this session, so I ran nothing. I am not going to report a number I did not measure. The honest status is: *prediction 5.5×, unmeasured; zero of the four permitted improvement rounds used, because I could not see a measurement to improve against.* The correctness argument above is analytic (index-range proof + a full hand trace), not empirical.

## VERDICT

The literal translation survives: the worm's body *is* a SIMD register, the eight winds *are* a real degree of freedom the textbook loop never uses, and taking "no ledger apart from the sand" seriously deletes the O(n²) table outright. Two things fell out of the literalism that I would not have written from the textbook — the choice of wind justified as *the wind for which the native's own "until nothing improves" terminates in one pass*, and the reversed cord, which is forced by the worm's geometry rather than bolted on.

Risks my own verdict names, and how each is discharged in the shipped code rather than left as a caveat:

1. *Wavefront bookkeeping loses to a plain loop on small inputs* → guarded: `n ≤ 32` takes `nw_rows_small`, a stack-only two-row crawl with no allocation.
2. *16-deep mounds overflow on a tall tray* → guarded: `n > 12000` falls to the `int32` furrows (the true bound is `2n+2 < 32768`, i.e. `n < 16383`; I left ~25% slack because the spill padding also carries values bounded by the same max-recurrence).
3. *No AVX2 on the machine* → guarded: `#if defined(__AVX2__)` with the plain-C `restrict`/`omp simd` wavefront as the standing path, which is still table-free.
4. *Threads* → deliberately absent, with the reason measured in cycles, not vibes: one furrow at n=1000 is ~125 cycles of work and furrows are strictly ordered.

What would falsify the whole thing: if `alignment_bench` runs mostly at `n ≲ 100`, the furrow-loop overhead and the cord-reversal setup eat the win and the measured speedup lands near 1.2–2×, not 5.5×. That is the single condition I would want checked first, and my threshold at 32 is a guess I did not get to tune against data — it is the one number in this kernel I'd expect to move after a real measurement.