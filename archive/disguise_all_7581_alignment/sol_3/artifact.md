## APPROACH

The disguised solution maps onto Needleman–Wunsch piece by piece:

| Lantern story | Real implementation |
|---|---|
| "a single reference card listing, for every pair of colors, match or not" | The two sequences are pre-encoded into integer lane arrays once; the match/mismatch value is then read for 16 pairs at once, branchlessly: `eq = cmpeq(va,vb); s = (eq & 2) - 1` → `+1`/`-1`. No per-cell compare-and-branch. |
| "grabs a whole handful of left lanterns against a whole handful of right lanterns at once" | **Anti-diagonal (wavefront) vectorization.** All cells on one anti-diagonal `d = i+j` are mutually independent — `H(i,d-i)` reads only diagonals `d-1` and `d-2`. This kills the "up, then left, then diagonal, in that order, one pair at a time" assumption. To make *both* handfuls stream forward in memory, **b is stored reversed** (`br[t]=b[n-1-t]`), so cell `i` on diagonal `d` reads `A[i-1]` and `BR[n-d+i]` — two contiguous ascending loads. That directly breaks the "both strings read start to end in the same direction" assumption. |
| "fills a whole stretch of squares in a single sweep" | One 256-bit store writes 16 DP cells. Per 16 cells: 5 loads, `cmpeq/and/sub/add/max/sub/max`, 1 store — ~0.8 uops/cell versus a ~2–3-cycle serial `dp[i][j-1]` dependency chain per cell in the reference. |
| "they only slow down at the rare spot where a skip might actually pay off" | A **provably sufficient band**, not a heuristic one. With this exact scoring, any alignment with `k` indel pairs scores `2m - n - 3k ≤ n - 5k`, and a path reaching displacement `δ = |i-j|` needs `k ≥ δ`. So with any known lower bound `L` on the optimal score, no optimal path leaves `|i-j| ≤ ⌊(n-L)/5⌋`. The main-diagonal score `2m₀-n` gives `L` for free; a cheap narrow first pass (`W/8`) gives a much better `L` when the sequences are related by a shift, and the second pass with the tightened width is then guaranteed exact. Cells outside the band are fenced with `-inf` sentinels written at exactly the two positions each diagonal can be read at. |
| "the numbers written down are exactly the same" | Every cell that is filled gets the identical NW value; the band only omits cells that provably cannot lie on an optimal path. Verified by hand on n=1 and n=2 (including the `AC`/`CA` gap case). |

int16 lanes (16-wide) are used while `n ≤ 9000` (in-band values are bounded below by `-3n`, safely above the `-30000` sentinel); an int32 8-wide instantiation of the same macro handles larger `n`; a banded scalar path covers non-AVX2 builds and allocation failure.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------------- banded scalar fallback (no AVX2 / malloc failure) -------- */
static int nw_band_scalar(int n, const unsigned char *a, const unsigned char *b, int W)
{
    const int NEG = -1073741824;
    int *mem = (int *)malloc((size_t)2 * (size_t)(n + 2) * sizeof(int));
    int *prev, *cur, i, j, r, jb;
    if (!mem) return 0;
    prev = mem; cur = mem + (n + 2);
    jb = (W < n) ? W : n;
    for (j = 0; j <= jb; j++) prev[j] = -2 * j;
    if (jb + 1 <= n + 1) prev[jb + 1] = NEG;
    for (i = 1; i <= n; i++) {
        int jlo = i - W, jhi = i + W;
        unsigned char ai = a[i - 1];
        if (jlo < 1) jlo = 1;
        if (jhi > n) jhi = n;
        if (i <= W) cur[0] = -2 * i; else cur[jlo - 1] = NEG;
        for (j = jlo; j <= jhi; j++) {
            int v = prev[j - 1] + ((ai == b[j - 1]) ? 1 : -1);
            int t = prev[j] - 2; if (t > v) v = t;
            t = cur[j - 1] - 2;  if (t > v) v = t;
            cur[j] = v;
        }
        if (jhi + 1 <= n + 1) cur[jhi + 1] = NEG;
        { int *tp = prev; prev = cur; cur = tp; }
    }
    r = prev[n];
    free(mem);
    return r;
}

#if defined(__AVX2__)
/* Banded anti-diagonal wavefront.  cur[i] = H(i, d-i)
   H(i,j) = max( H(i-1,j-1)+s , H(i-1,j)-2 , H(i,j-1)-2 )
          = max( d0[i-1]+s   , d1[i-1]-2  , d1[i]-2   )                      */
#define NW_GEN(FN, T, LANES, SET1, CMPEQ, MAXOP, ADDOP, SUBOP, NEGV)                       \
static int FN(int n, const T *A, const T *BR, int W, T *buf, int stride)                   \
{                                                                                          \
    T *d0 = buf + 1, *d1 = buf + stride + 1, *d2 = buf + 2 * stride + 1;                   \
    const __m256i vone = SET1(1), vtwo = SET1(2);                                          \
    const int dmax = 2 * n;                                                                \
    int d;                                                                                 \
    for (d = 0; d <= dmax; d++) {                                                          \
        int ilo, ihi, lo, hi, i, u, t;                                                     \
        ilo = d - n; if (ilo < 0) ilo = 0;                                                 \
        t = d - W; if (t > 0) { t = (t + 1) >> 1; if (t > ilo) ilo = t; }                  \
        ihi = (d < n) ? d : n;                                                             \
        t = (d + W) >> 1; if (t < ihi) ihi = t;                                            \
        lo = (ilo > 1) ? ilo : 1;                                                          \
        hi = (ihi < d - 1) ? ihi : d - 1;                                                  \
        u  = n - d;                                                                        \
        for (i = lo; i <= hi; i += LANES) {                                                \
            __m256i eq = CMPEQ(_mm256_loadu_si256((const __m256i *)(A + (i - 1))),         \
                               _mm256_loadu_si256((const __m256i *)(BR + (u + i))));       \
            __m256i sc = SUBOP(_mm256_and_si256(eq, vtwo), vone);                          \
            __m256i dg = ADDOP(_mm256_loadu_si256((const __m256i *)(d0 + (i - 1))), sc);   \
            __m256i gp = SUBOP(MAXOP(_mm256_loadu_si256((const __m256i *)(d1 + (i - 1))),  \
                                     _mm256_loadu_si256((const __m256i *)(d1 + i))),       \
                               vtwo);                                                      \
            _mm256_storeu_si256((__m256i *)(d2 + i), MAXOP(dg, gp));                       \
        }                                                                                  \
        if (ilo == 0) d2[0] = (T)(-2 * d);                                                 \
        if (ihi == d) d2[d] = (T)(-2 * d);                                                 \
        d2[ilo - 1] = (T)(NEGV);                                                           \
        d2[ihi + 1] = (T)(NEGV);                                                           \
        { T *tp = d0; d0 = d1; d1 = d2; d2 = tp; }                                         \
    }                                                                                      \
    return (int)d1[n];                                                                     \
}

NW_GEN(nw_band_i16, short, 16, _mm256_set1_epi16, _mm256_cmpeq_epi16, _mm256_max_epi16,
       _mm256_add_epi16, _mm256_sub_epi16, -30000)
NW_GEN(nw_band_i32, int, 8, _mm256_set1_epi32, _mm256_cmpeq_epi32, _mm256_max_epi32,
       _mm256_add_epi32, _mm256_sub_epi32, -1073741824)
#endif

int kernel(int n, const char *a, const char *b)
{
    const unsigned char *ua = (const unsigned char *)a;
    const unsigned char *ub = (const unsigned char *)b;
    int i, m0 = 0, Wreq, Wa, Wb, S;

    if (n <= 0) return 0;

    /* lower bound L = score of the no-gap path; no optimal path can leave
       |i-j| <= floor((n-L)/5), because a path with displacement d needs >= d
       indel pairs and then scores at most n-5d.                              */
    for (i = 0; i < n; i++) m0 += (ua[i] == ub[i]);
    Wreq = (2 * (n - m0)) / 5;            /* = (n - (2*m0 - n)) / 5 */
    if (Wreq < 1) Wreq = 1;
    if (Wreq > n) Wreq = n;

    Wa = (Wreq > 32) ? (Wreq >> 3) : Wreq;   /* cheap probe pass */
    if (Wa < 16) Wa = 16;
    if (Wa > Wreq) Wa = Wreq;

#if defined(__AVX2__)
    if (n <= 9000) {
        int stride = n + 40, pad = n + 48;
        short *mem = (short *)calloc((size_t)(2 * pad + 3 * stride), sizeof(short));
        if (mem) {
            short *A = mem, *BR = mem + pad, *buf = mem + 2 * pad;
            for (i = 0; i < n; i++) A[i]  = (short)ua[i];
            for (i = 0; i < n; i++) BR[i] = (short)ub[n - 1 - i];  /* b reversed */
            S = nw_band_i16(n, A, BR, Wa, buf, stride);
            if (Wa < Wreq) {
                Wb = (n - S) / 5;          /* S is a valid lower bound on optimum */
                if (Wb < 1) Wb = 1;
                if (Wb > n) Wb = n;
                if (Wb > Wa) S = nw_band_i16(n, A, BR, Wb, buf, stride);
            }
            free(mem);
            return S;
        }
    } else {
        int stride = n + 24, pad = n + 32;
        int *mem = (int *)calloc((size_t)(2 * pad + 3 * stride), sizeof(int));
        if (mem) {
            int *A = mem, *BR = mem + pad, *buf = mem + 2 * pad;
            for (i = 0; i < n; i++) A[i]  = (int)ua[i];
            for (i = 0; i < n; i++) BR[i] = (int)ub[n - 1 - i];
            S = nw_band_i32(n, A, BR, Wa, buf, stride);
            if (Wa < Wreq) {
                Wb = (n - S) / 5;
                if (Wb < 1) Wb = 1;
                if (Wb > n) Wb = n;
                if (Wb > Wa) S = nw_band_i32(n, A, BR, Wb, buf, stride);
            }
            free(mem);
            return S;
        }
    }
#endif
    return nw_band_scalar(n, ua, ub, Wreq);
}
```

## PREDICTION: 15

(speedup factor over the reference `kernel` on random-DNA inputs of a few hundred to a few thousand bases)

Reasoning behind the number: ~16× from int16 lane width minus wavefront overhead → ~8–10× effective, times ~1.7–1.9× from the provable band at 25 % baseline identity, minus the ~12 % cost of the probe pass, plus whatever the reference loses to its `(n+1)²`-int allocation and page faults. On highly similar sequences the band collapses and the factor should be far larger (100×+); on tiny `n` it should fall to ~4–6×.

## MEASUREMENT

**Not measured.** No tools were available in this session — `alignment_bench` and `alignment_contract` were listed but the session is toolless, so I could not compile, run the contract check, or time anything. I am reporting that plainly rather than inventing numbers, and the prediction above therefore stands unvalidated. The four allotted improvement rounds were likewise not spent; what is above is the first and only version.

What I *did* verify, by hand rather than by machine:
- n=1 `A`/`A` → traced through all three anti-diagonals → returns 1. ✓
- n=2 `AC`/`CA` → full reference table computed by hand gives −2; the banded wavefront trace (including the `-inf` sentinel standing in for the excluded cells (0,2) and (2,0)) also gives −2. ✓
- Buffer-bound arithmetic: max store index `3n+96` against `3n+120` available; A/BR read tails `n+14` against `n+48` padding. ✓
- Sentinel safety: in-band values are ≥ `-3n` ≥ `-27000` at the `n ≤ 9000` cutoff, strictly above `NEG+1 = -29999`, and every in-band interior cell has an in-band diagonal predecessor, so no sentinel can ever win a `max`. ✓

## VERDICT

The mechanism translates cleanly and, I believe, correctly: precomputed match card → branchless SIMD match vector; handfuls of lanterns → anti-diagonal wavefront with `b` stored reversed so both operands stream forward; single sweep → one 256-bit store per 16 DP cells; "only slow down where a skip could pay off" → a band that is *provably* sufficient rather than heuristic, tightened by one cheap probe pass.

Honest caveats: the whole speed claim is unmeasured. Two things I deliberately left on the table and would attack first with a bench in hand — (1) no AVX-512 instantiation (the mask-register `cmpeq` shape does not fit the macro, worth ~1.5–2× on a server), and (2) no threading: anti-diagonal-level OpenMP is a loss (2n barriers swamp the ~ms of work), and block-wavefront tiling is the right answer but adds boundary-exchange complexity I was not willing to ship untested. If measurement shows the probe pass is pure overhead on the actual benchmark distribution, dropping it is a one-line change.