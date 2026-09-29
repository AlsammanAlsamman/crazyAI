## APPROACH

**Mapping the disguised solution onto the real problem.** The camp rulebook *is* Needleman–Wunsch, and I keep every clause of it:

| Disguised element | Real implementation |
|---|---|
| Mira's / Theo's trails of 4 colors | `a`, `b`, length `n`, alphabet {A,C,G,T} |
| Match +1, mismatch −1, hop −2 | `MATCH=+1, MISMATCH=-1, GAP=-2` |
| One giant scoreboard, every square vs. every square | The full `(n+1)×(n+1)` DP grid — **every cell is still computed**, no band, no heuristic, no early exit |
| Never write a box until above, left, and above-left are written | Exactly the three-neighbour recurrence `H[i][j]=max(H[i-1][j-1]+s, H[i-1][j]-2, H[i][j-1]-2)` |
| Both trails walked start→finish, never backwards | Forward `i` and `j` only; `b` is *stored* reversed purely as an address-arithmetic device, the comparison performed is still `a[i-1] vs b[j-1]` |
| Bottom-right box is the official score | Return `H[n][n]` |
| "A single judge working alone, box by box, waiting on three neighbours" | ← **this is the only thing I change**: not *what* is computed, but the *order* and the *number of judges working at once* |

**The one change: anti-diagonal wavefront.** The rulebook forbids writing a box before its three neighbours exist — it does **not** require row-major order. All boxes on one anti-diagonal `d = i+j` have their three neighbours on diagonals `d-1` and `d-2`, so they are mutually independent and may be filled **simultaneously**. That is many judges filling one diagonal stripe at once, each still obeying the three-neighbour rule literally. So:

- Keep only **three anti-diagonal buffers** instead of the whole sheet of graph paper (O(n) memory instead of O(n²)) — the reference's real cost at large `n` is streaming a 64 MB table through DRAM.
- Fill each diagonal with **SIMD**: 32 cells per instruction (AVX-512BW, `int16`) or 16 (AVX2).
- Killer detail: along a diagonal, `i` grows while `j` shrinks, so `b` would be read backwards — unvectorizable. Fix: pre-store `br[k]=b[n-1-k]` once, then `b[d-i-1] == br[n-d+i]`, which now **advances forward in lockstep with `a`**. Two contiguous byte loads, one `vpcmpeqb`, one `vpmovsxbw`, and the match/mismatch vector is `p2 - 1 - 2·eq`.
- `int16` is safe because every table value lies in `[-2n, n]`; guarded by `n ≤ 16000`, with an exact `int` two-row fallback otherwise (and for non-AVX2 targets).

Boundary cells (`i=0` or `j=0`) are written explicitly per diagonal after the vector store; I verified that every over-written padding lane lies strictly outside the range any later diagonal reads, so garbage never leaks into a live cell.

I deliberately did **not** add OpenMP: a `parallel for` per diagonal needs 2n barriers (~10 ms for n=4000) against ~0.5 ms of actual work — it would be a large *slowdown*. Real parallelism here needs a tiled skewed wavefront, which I judged too bug-prone to ship untested.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

#define NW_MATCH     1
#define NW_MISMATCH -1
#define NW_GAP       2   /* magnitude; penalty is -NW_GAP */

/* Exact fallback: same recurrence, two rows, int precision. */
static int nw_rows(int n, const char *a, const char *b)
{
    int *prev, *cur, r;
    prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = -NW_GAP * j;
    for (int i = 1; i <= n; i++) {
        const char ai = a[i - 1];
        cur[0] = -NW_GAP * i;
        for (int j = 1; j <= n; j++) {
            int dg = prev[j - 1] + ((ai == b[j - 1]) ? NW_MATCH : NW_MISMATCH);
            int up = prev[j]     - NW_GAP;
            int lf = cur[j - 1]  - NW_GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

#if defined(__AVX2__)
    /* int16 is exact while the whole table fits in [-2n, n]. */
    if (n <= 16000) {
        const int stride = n + 1 + 64;            /* slack for SIMD overrun */
        int16_t *base = (int16_t *)malloc((size_t)3 * (size_t)stride * sizeof(int16_t));
        char    *A    = (char *)malloc((size_t)n + 160);
        char    *BR   = (char *)malloc((size_t)n + 160);

        if (base && A && BR) {
            memset(base, 0,    (size_t)3 * (size_t)stride * sizeof(int16_t));
            memset(A,    0x00, (size_t)n + 160);
            memset(BR,   0x7F, (size_t)n + 160);  /* pads can never compare equal */
            memcpy(A, a, (size_t)n);
            for (int k = 0; k < n; k++) BR[k] = b[n - 1 - k];

            int16_t *p2 = base;                   /* diagonal d-2 */
            int16_t *p1 = base + stride;          /* diagonal d-1 */
            int16_t *cu = base + 2 * stride;      /* diagonal d   */

            p2[0] = 0;                            /* d = 0 */
            p1[0] = (int16_t)(-NW_GAP);           /* d = 1 */
            p1[1] = (int16_t)(-NW_GAP);

            for (int d = 2; d <= 2 * n; d++) {
                int ilo = d - n; if (ilo < 1) ilo = 1;
                int ihi = d - 1; if (ihi > n) ihi = n;
                const int boff = n - d;           /* br index = i + boff */

#if defined(__AVX512BW__)
                {
                    const __m512i vneg1 = _mm512_set1_epi16(-1);
                    const __m512i vgap  = _mm512_set1_epi16(NW_GAP);
                    for (int i = ilo; i <= ihi; i += 32) {
                        __m256i ca = _mm256_loadu_si256((const __m256i *)(A  + (i - 1)));
                        __m256i cb = _mm256_loadu_si256((const __m256i *)(BR + (i + boff)));
                        __m512i eq = _mm512_cvtepi8_epi16(_mm256_cmpeq_epi8(ca, cb));
                        __m512i pd = _mm512_loadu_si512((const void *)(p2 + (i - 1)));
                        __m512i pu = _mm512_loadu_si512((const void *)(p1 + (i - 1)));
                        __m512i pl = _mm512_loadu_si512((const void *)(p1 + i));
                        __m512i dg = _mm512_sub_epi16(_mm512_add_epi16(pd, vneg1),
                                                      _mm512_add_epi16(eq, eq));
                        __m512i ul = _mm512_sub_epi16(_mm512_max_epi16(pu, pl), vgap);
                        _mm512_storeu_si512((void *)(cu + i), _mm512_max_epi16(dg, ul));
                    }
                }
#else
                {
                    const __m256i vneg1 = _mm256_set1_epi16(-1);
                    const __m256i vgap  = _mm256_set1_epi16(NW_GAP);
                    for (int i = ilo; i <= ihi; i += 16) {
                        __m128i ca = _mm_loadu_si128((const __m128i *)(A  + (i - 1)));
                        __m128i cb = _mm_loadu_si128((const __m128i *)(BR + (i + boff)));
                        __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                        __m256i pd = _mm256_loadu_si256((const __m256i *)(p2 + (i - 1)));
                        __m256i pu = _mm256_loadu_si256((const __m256i *)(p1 + (i - 1)));
                        __m256i pl = _mm256_loadu_si256((const __m256i *)(p1 + i));
                        __m256i dg = _mm256_sub_epi16(_mm256_add_epi16(pd, vneg1),
                                                      _mm256_add_epi16(eq, eq));
                        __m256i ul = _mm256_sub_epi16(_mm256_max_epi16(pu, pl), vgap);
                        _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi16(dg, ul));
                    }
                }
#endif
                if (d <= n) {                     /* first row / first column cells */
                    int16_t bv = (int16_t)(-NW_GAP * d);
                    cu[0] = bv;
                    cu[d] = bv;
                }
                { int16_t *t = p2; p2 = p1; p1 = cu; cu = t; }
            }

            int res = (int)p1[n];                 /* bottom-right box */
            free(base); free(A); free(BR);
            return res;
        }
        free(base); free(A); free(BR);
    }
#endif
    return nw_rows(n, a, b);
}
```

## PREDICTION

PREDICTION: 25

(Speedup factor versus the reference `kernel` above, on a benchmark dominated by `n` in the low thousands. Reasoning: ~16× from AVX2 lanes or ~32× from AVX-512BW lanes on the arithmetic, multiplied by a further gain from dropping the O(n²) table to three O(n) buffers that stay resident in L1/L2, partially offset by short ramp-up/ramp-down diagonals and per-diagonal loop overhead.)

## MEASUREMENT

**Not measured.** No tools were available in this session — `alignment_bench` and `alignment_contract` could not be invoked, so the four allowed measure-and-improve rounds did not happen. I am reporting that plainly rather than inventing numbers: the prediction above stands untested, and the figure of 25 is an *a priori* estimate from instruction counts and memory-traffic reasoning, not an observation.

What I did verify, by hand rather than by running:
- Index bounds for `A`, `BR`, `p1`, `p2`, `cu` at both ends of every diagonal, including the `d ≤ n` and `d > n` regimes.
- That SIMD lanes written past `ihi` are never read as a live cell on any later diagonal (the live read range of diagonal `d` is exactly `[max(0,d-n-1), min(n,d-1)]`, which the previous diagonals cover exactly).
- Hand-traced `n = 1` and `n = 2` against the recurrence.
- `int16` headroom: table range `[-2n, n]`, worst intermediate `-2n-2 = -32002 > -32768` at the `n ≤ 16000` guard.

## VERDICT

Faithful to the mechanism, unverified on hardware. The kernel computes the identical `(n+1)²` grid with the identical three-neighbour recurrence and returns `H[n][n]`; the only departure from the disguised solution is traversal order (anti-diagonal instead of row-major) and the number of cells resolved per step — which is precisely the inefficiency the story complains about ("a single judge working through the whole giant scoreboard alone"), not a different algorithm.

Two honest caveats I would resolve first if I could run anything:
1. The AVX-512 branch is compile-time selected and I could not confirm it compiles on the target; if `_mm512_cvtepi8_epi16` / `_mm512_max_epi16` are unavailable the build fails outright rather than degrading. If the harness rejects it, delete the `#if defined(__AVX512BW__)` block and keep only the AVX2 path — that is the change I would make first.
2. No multithreading. "Fastest I can" here means fastest *single-threaded*; a tiled skewed wavefront is the remaining 4–8× on a multicore box, and I declined to ship it blind.