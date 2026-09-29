# APPROACH

**Mapping the disguised solution onto Needleman–Wunsch.**

| Parade image | Real mechanism |
|---|---|
| "Skip drawing any scoreboard at all" | Never materialize the `(n+1)²` DP table. The reference's table is pure memory traffic (67 MB at n=4096); we keep only three small rows. |
| "Walk down both lines together, front to back" | Walk **anti-diagonals** `u = i+j` from 0 to 2n. This is the one traversal order in which the three NW dependencies — `(i-1,j-1)` at `u-2`, `(i-1,j)` and `(i,j-1)` both at `u-1` — are *all already finished*, so the whole anti-diagonal is data-parallel. The contract's "each box depends on above / left / above-left, filled one at a time" is respected exactly; only the enumeration order changes. |
| "a single running tally in your head" | The score along the main diagonal, `S_diag = Σ ±1`, computed in one O(n) pass. It is a *valid achievable alignment score*, hence a lower bound on the optimum. |
| "a slim moving window of the last few steps' tallies… glance back only that short distance" | A **band** `|i−j| ≤ W`. This is the load-bearing part and it must be *provably* sufficient, not heuristic. For equal-length strings an alignment with `k` insertions and `k` deletions has `n−k` substitution columns, so its score is `≤ (n−k) − 4k = n − 5k`, and its path never leaves `|i−j| ≤ k`. So any alignment outside band `W` scores `≤ n − 5(W+1)`. Taking `W = ⌊(n − LB)/5⌋` for any lower bound `LB` makes every excluded alignment strictly worse than `LB` — the banded answer *is* the exact NW answer. |
| "let that older memory drop away as you keep walking forward" | Three rotating anti-diagonal buffers (`u-2`, `u-1`, `u`), indexed by absolute `i` so the neighbours are just unaligned loads at `i-1`, `i-1`, `i`. Out-of-band neighbours are two explicitly re-written `-INF` sentinels per step — proven below to be the only out-of-band cells ever read. |

**Two-shot band tightening.** Run once with a tiny band (`W ≤ 32`, cost `O(n)`) to get a much better lower bound than `S_diag`, then once with `W = ⌊(n−S)/5⌋`. Because `S(W)` is monotone in `W`, the second run's certificate `⌊(n−S)/5⌋ ≤ W` is guaranteed to hold — **at most two banded passes, ever**.

**SIMD.** `int16` with AVX2 = 16 cells per instruction group (`n ≤ 8000`; in-band values are `≥ −3n = −24000`, sentinel `−30000`, saturating adds so garbage lanes can never wrap). `int32` path (8 lanes) above that, scalar banded fallback without AVX2. Scores are compared by loading `a[i-1…]` forward and a **reversed copy** of `b`, which makes both operands of the anti-diagonal match test contiguous forward loads. Both strings are still consumed start-to-end; the reversal is a memory layout, not a change of alignment direction.

Identical sequences collapse to `W=1` → **O(n)**. Worst case (all-mismatch) is `W=0.4n` → 0.8n² cells, still under the reference, at 16 lanes.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define NWPB 64   /* element padding on both sides of every buffer */

/* ---------------- scalar banded anti-diagonal NW ---------------- */
static int nw_band_scalar(int n, const unsigned char *A, const unsigned char *Brev,
                          int *q0, int *q1, int *q2, int W)
{
    const int NEGI = -(1 << 28);
    int m = n + 1 + 2 * NWPB, t, u;
    for (t = 0; t < m; t++) { q0[t] = NEGI; q1[t] = NEGI; q2[t] = NEGI; }
    int *dm2 = q0 + NWPB, *dm1 = q1 + NWPB, *cur = q2 + NWPB;
    for (u = 0; u <= 2 * n; u++) {
        int tt = u - W;
        int ilo = (tt <= 0) ? 0 : ((tt + 1) >> 1);
        if (u - n > ilo) ilo = u - n;
        int ihi = (u + W) >> 1;
        if (ihi > n) ihi = n;
        if (ihi > u) ihi = u;
        int base = n - u, i;
        for (i = ilo; i <= ihi; i++) {
            int s = (A[i - 1] == Brev[base + i]) ? 1 : -1;
            int v = dm2[i - 1] + s;
            int p = dm1[i - 1], q = dm1[i];
            int w = ((p > q) ? p : q) - 2;
            cur[i] = (v > w) ? v : w;
        }
        if (ilo == 0) cur[0] = -2 * u;
        if (ihi == u) cur[u] = -2 * u;
        cur[ilo - 1] = NEGI;
        cur[ihi + 1] = NEGI;
        { int *tp = dm2; dm2 = dm1; dm1 = cur; cur = tp; }
    }
    return dm1[n];
}

#if defined(__AVX2__)
/* ---------------- AVX2, 16 lanes of int16 (n <= 8000) ---------------- */
static int nw_band_i16(int n, const unsigned char *A, const unsigned char *Brev,
                       short *q0, short *q1, short *q2, int W)
{
    const short NEGS = -30000;
    int m = n + 1 + 2 * NWPB, t, u;
    for (t = 0; t < m; t++) { q0[t] = NEGS; q1[t] = NEGS; q2[t] = NEGS; }
    short *dm2 = q0 + NWPB, *dm1 = q1 + NWPB, *cur = q2 + NWPB;
    const __m256i vTWO = _mm256_set1_epi16(2);
    const __m256i vM1  = _mm256_set1_epi16(-1);
    const __m256i vP1  = _mm256_set1_epi16(1);
    for (u = 0; u <= 2 * n; u++) {
        int tt = u - W;
        int ilo = (tt <= 0) ? 0 : ((tt + 1) >> 1);
        if (u - n > ilo) ilo = u - n;
        int ihi = (u + W) >> 1;
        if (ihi > n) ihi = n;
        if (ihi > u) ihi = u;
        int base = n - u, i;
        for (i = ilo; i <= ihi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(A + (i - 1)));
            __m128i cb = _mm_loadu_si128((const __m128i *)(Brev + (base + i)));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i sc = _mm256_blendv_epi8(vM1, vP1, eq);
            __m256i dg = _mm256_loadu_si256((const __m256i *)(dm2 + (i - 1)));
            __m256i up = _mm256_loadu_si256((const __m256i *)(dm1 + (i - 1)));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(dm1 + i));
            __m256i vv = _mm256_max_epi16(
                             _mm256_adds_epi16(dg, sc),
                             _mm256_subs_epi16(_mm256_max_epi16(up, lf), vTWO));
            _mm256_storeu_si256((__m256i *)(cur + i), vv);
        }
        if (ilo == 0) cur[0] = (short)(-2 * u);
        if (ihi == u) cur[u] = (short)(-2 * u);
        cur[ilo - 1] = NEGS;
        cur[ihi + 1] = NEGS;
        { short *tp = dm2; dm2 = dm1; dm1 = cur; cur = tp; }
    }
    return (int)dm1[n];
}

/* ---------------- AVX2, 8 lanes of int32 (large n) ---------------- */
static int nw_band_i32(int n, const unsigned char *A, const unsigned char *Brev,
                       int *q0, int *q1, int *q2, int W)
{
    const int NEGI = -(1 << 28);
    int m = n + 1 + 2 * NWPB, t, u;
    for (t = 0; t < m; t++) { q0[t] = NEGI; q1[t] = NEGI; q2[t] = NEGI; }
    int *dm2 = q0 + NWPB, *dm1 = q1 + NWPB, *cur = q2 + NWPB;
    const __m256i vTWO = _mm256_set1_epi32(2);
    const __m256i vM1  = _mm256_set1_epi32(-1);
    const __m256i vP1  = _mm256_set1_epi32(1);
    for (u = 0; u <= 2 * n; u++) {
        int tt = u - W;
        int ilo = (tt <= 0) ? 0 : ((tt + 1) >> 1);
        if (u - n > ilo) ilo = u - n;
        int ihi = (u + W) >> 1;
        if (ihi > n) ihi = n;
        if (ihi > u) ihi = u;
        int base = n - u, i;
        for (i = ilo; i <= ihi; i += 8) {
            __m128i ca = _mm_loadl_epi64((const __m128i *)(A + (i - 1)));
            __m128i cb = _mm_loadl_epi64((const __m128i *)(Brev + (base + i)));
            __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
            __m256i sc = _mm256_blendv_epi8(vM1, vP1, eq);
            __m256i dg = _mm256_loadu_si256((const __m256i *)(dm2 + (i - 1)));
            __m256i up = _mm256_loadu_si256((const __m256i *)(dm1 + (i - 1)));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(dm1 + i));
            __m256i vv = _mm256_max_epi32(
                             _mm256_add_epi32(dg, sc),
                             _mm256_sub_epi32(_mm256_max_epi32(up, lf), vTWO));
            _mm256_storeu_si256((__m256i *)(cur + i), vv);
        }
        if (ilo == 0) cur[0] = -2 * u;
        if (ihi == u) cur[u] = -2 * u;
        cur[ilo - 1] = NEGI;
        cur[ihi + 1] = NEGI;
        { int *tp = dm2; dm2 = dm1; dm1 = cur; cur = tp; }
    }
    return dm1[n];
}
#endif /* __AVX2__ */

static int nw_band(int n, const unsigned char *A, const unsigned char *Brev,
                   unsigned char *rows, size_t rowb, int W)
{
#if defined(__AVX2__)
    if (n <= 8000)
        return nw_band_i16(n, A, Brev, (short *)rows,
                           (short *)(rows + rowb), (short *)(rows + 2 * rowb), W);
    return nw_band_i32(n, A, Brev, (int *)rows,
                       (int *)(rows + rowb), (int *)(rows + 2 * rowb), W);
#else
    return nw_band_scalar(n, A, Brev, (int *)rows,
                          (int *)(rows + rowb), (int *)(rows + 2 * rowb), W);
#endif
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    size_t nb   = (size_t)n + 2 * NWPB;
    size_t ne   = (size_t)n + 1 + 2 * NWPB;
    size_t seqb = (nb + 63) & ~(size_t)63;
    size_t rowb = ((ne * sizeof(int)) + 63) & ~(size_t)63;

    unsigned char *mem = (unsigned char *)malloc(2 * seqb + 3 * rowb + 64);
    if (!mem) return 0;
    unsigned char *abuf = mem;
    unsigned char *bbuf = mem + seqb;
    unsigned char *rows = mem + 2 * seqb;

    memset(abuf, 0xF1, seqb);
    memset(bbuf, 0xF2, seqb);
    unsigned char *A    = abuf + NWPB;
    unsigned char *Brev = bbuf + NWPB;

    int i, sdiag = 0;
    for (i = 0; i < n; i++) A[i] = (unsigned char)a[i];
    for (i = 0; i < n; i++) Brev[i] = (unsigned char)b[n - 1 - i];
    for (i = 0; i < n; i++) sdiag += (a[i] == b[i]) ? 1 : -1;

    /* provably sufficient band half-width from a lower bound LB: W = floor((n-LB)/5) */
    int W = (n - sdiag) / 5;
    if (W > 32) W = 32;          /* cheap probe first: sharpens LB for O(n) cost */
    if (W < 1)  W = 1;
    if (W > n)  W = n;

    int S;
    for (;;) {
        S = nw_band(n, A, Brev, rows, rowb, W);
        int need = (n - S) / 5;
        if (need <= W || W >= n) break;   /* certificate: nothing outside band can win */
        W = need;
        if (W > n) W = n;
    }
    free(mem);
    return S;
}
```

# PREDICTION

PREDICTION: 15

# MEASUREMENT

**Not performed — reported plainly rather than fabricated.** This session has no tool access (`alignment_bench` / `alignment_contract` are listed but unavailable), so the four permitted measure-and-improve rounds could not be run. Nothing below is a timing; these are the only claims I can actually stand behind:

- **Correctness argument (complete, not empirical).** (1) The banded value is the max over paths with every vertex in `|i−j| ≤ W`, hence an achievable alignment score, hence `≤ S*`. (2) Any alignment with `k > W` indel pairs scores `≤ n − 5(W+1)`; the loop exits only when `⌊(n−S)/5⌋ ≤ W`, i.e. `S ≥ n − 5W − 4 > n − 5(W+1)`, so no excluded alignment beats `S`. Therefore `S = S*` exactly. (3) The only out-of-band cells ever read are indices `ilo−1` and `ihi+1` of the two previous anti-diagonals — because `ilo` is non-decreasing and `ihi` grows by at most 1 per step — and both are re-stamped `-INF` every step, so stale/garbage lanes beyond `ihi+1` are provably unreachable. (4) All out-of-range accesses are into explicit 64-element pads on padded private copies of `a` and reversed `b`; the caller's buffers are read only in-range.
- **Work count (exact, not timed).** Two banded passes, `≈ 2n(W+1)` cells each, with `W ≤ 0.4n` in the all-mismatch worst case and `W = 1` for identical inputs. Worst case ≈ 0.8 n² cells at 16 cells/vector-op vs the reference's n² scalar cells through an `(n+1)²`-int table.
- **Untested risk I want on the record:** the `int16` lane path is the one I would most want a bench to confirm; the `n ≤ 8000` guard gives ≥ 5000 counts of headroom between the worst in-band value (`−3n`) and the sentinel (`−30000`), and every arithmetic op there is saturating, but that is reasoning, not a measurement.

# VERDICT

The mechanism translates cleanly and, importantly, *provably* — the "slim moving window" is not a heuristic band here but one with an exactness certificate, which is what the "exactly matching a reference DP aligner" requirement demands. I expect a solid win (predicted 15×, plausibly 50–100× on similar sequences, and O(n) on identical ones), but **I did not measure it, so the prediction stands unverified.** If the bench feeds it maximally dissimilar equal-length sequences the band degrades to ~0.8n² and the win narrows to roughly the SIMD factor alone; that is the case I would check first if timings become available.