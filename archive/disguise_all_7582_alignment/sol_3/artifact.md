## APPROACH

**Mapping the disguised solution onto the real problem**

| Camp story | Real implementation |
|---|---|
| Giant scoreboard, Mira down the side / Theo across the top | The `(n+1)×(n+1)` Needleman–Wunsch DP matrix `dp[i][j]` |
| A box needs above, left, above-left already written | `dp[i][j] = max(dp[i-1][j-1]+s, dp[i-1][j]-2, dp[i][j-1]-2)` |
| "Boxes on the same lower-left→upper-right slanting line never need each other" | Cells with constant `i+j = d` (anti-diagonal) are mutually independent; they only read anti-diagonals `d-1` and `d-2` |
| **A whole row of judges standing on one diagonal, all filling their box in the same breath** | **SIMD lanes**: one vector register = 16 (AVX2, `int16`) or 32 (AVX-512BW) cells of the same anti-diagonal computed in one instruction stream. The "team of judges" is the vector width. |
| Shift the team one step over to the next diagonal | Rotate three rolling anti-diagonal buffers `d2 → d1 → d0` |
| "Most of that scoreboard holds nothing interesting" | Never materialize the `n²` table at all — only 3 buffers of `n+O(1)` `int16` values, which live in L1/L2 instead of streaming 4·n² bytes to DRAM. This is where most of the real win comes from. |
| Same neighbors, same relative order, identical bottom-right answer | Bit-exact same recurrence, same tie behavior (`max` is associative/commutative here), so the result equals the reference for every input. Both strings are still read **start→end**. |

**The one implementation trick that makes the diagonal sweep vectorizable.** Indexing each anti-diagonal buffer by `i` (not by `j`), cell `i` on diagonal `d` needs `d2[i-1]`, `d1[i-1]`, `d1[i]` — pure unit-stride shifts, so all three operands are contiguous unaligned loads. The character comparison would normally need `b[d-i-1]`, a *reversed* walk over `b`. So I precompute `br[k] = b[n-1-k]` once; then `b[d-i-1] = br[(n-d)+i]`, which is also unit-stride in `i`. Both operands become plain 16/32-byte loads: `cmpeq_epi8` → `cvtepi8_epi16` → `(mask & 2) - 1` gives `+1/-1` with no branches.

**No masked tails.** Buffers and the two byte arrays are over-allocated, so the last vector of each diagonal is allowed to run past the live range and write scratch. I proved this is safe: garbage lanes satisfy `v_d ≤ max(v_{d-2}+1, v_{d-1}-2)` and `v_d ≥ v_{d-2}-1`, so over `2n` diagonals garbage stays in `[-3n, 2n]`; with `int16` that is safe for `n ≤ 10000` (which is far past the point where the reference's `(n+1)²·4` byte table is even allocatable). Larger `n` falls back to a scalar `int32` diagonal sweep. Live cells never read a garbage index (all reads have index ≤ `ihi`), and the two boundary cells of each diagonal are rewritten *after* the vector loop.

**Why not OpenMP threads.** Thread-level judges on each diagonal need a barrier per diagonal: `2n` barriers × ~0.5 µs ≈ 1 ms at n=1000, versus ~0.1 ms of total SIMD work. The barrier dominates, so the "team" is realized as SIMD lanes (and, for AVX-512 machines, 32 judges at once). Keeping it single-threaded also keeps it deterministic.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__) || defined(__AVX512BW__)
#include <immintrin.h>
#endif

#define NW_I16_MAX 10000  /* int16 safe bound incl. scratch-lane growth (|v| <= 3n) */

/* Scalar anti-diagonal sweep, 32-bit: fallback for no-AVX2 builds and huge n. */
static int nw_diag_scalar(int n, const char *a, const char *b)
{
    size_t dl = (size_t)n + 2;
    int *buf = (int *)calloc(3 * dl, sizeof(int));
    if (!buf) return 0;
    int *d2 = buf, *d1 = buf + dl, *d0 = buf + 2 * dl;
    d2[0] = 0;                 /* diagonal 0: dp[0][0] */
    d1[0] = -2; d1[1] = -2;    /* diagonal 1: dp[0][1], dp[1][0] */
    for (int d = 2; d <= 2 * n; d++) {
        int ilo = (d - n > 1) ? d - n : 1;
        int ihi = (d - 1 < n) ? d - 1 : n;
        for (int i = ilo; i <= ihi; i++) {
            int v = d2[i - 1] + ((a[i - 1] == b[d - i - 1]) ? 1 : -1);
            int u = d1[i - 1] - 2, l = d1[i] - 2;
            if (u > v) v = u;
            if (l > v) v = l;
            d0[i] = v;
        }
        if (d <= n) { d0[0] = -2 * d; d0[d] = -2 * d; }
        int *t = d2; d2 = d1; d1 = d0; d0 = t;
    }
    int r = d1[n];
    free(buf);
    return r;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if !(defined(__AVX2__) || defined(__AVX512BW__))
    return nw_diag_scalar(n, a, b);
#else
    if (n > NW_I16_MAX) return nw_diag_scalar(n, a, b);

    const size_t PAD  = 128;
    const size_t blen = (size_t)n + 2 * PAD;              /* padded byte arrays  */
    const size_t dl   = (size_t)n + 2 + 128;              /* padded i16 diagonals */

    unsigned char *raw = (unsigned char *)malloc(2 * blen + 3 * dl * sizeof(int16_t) + 64);
    if (!raw) return nw_diag_scalar(n, a, b);

    unsigned char *ap  = raw;          /* ap[k]  = a[k]                     */
    unsigned char *brp = raw + blen;   /* brp[k] = b[n-1-k]  (reversed b)   */
    memset(raw, 0, 2 * blen);
    memcpy(ap, a, (size_t)n);
    for (int k = 0; k < n; k++) brp[k] = (unsigned char)b[n - 1 - k];

    int16_t *base16 = (int16_t *)(((uintptr_t)(raw + 2 * blen) + 63u) & ~(uintptr_t)63u);
    memset(base16, 0, 3 * dl * sizeof(int16_t));
    int16_t *d2 = base16, *d1 = base16 + dl, *d0 = base16 + 2 * dl;

    d2[0] = 0;                                  /* diagonal 0 */
    d1[0] = -2; d1[1] = -2;                     /* diagonal 1 */

#if defined(__AVX512BW__)
    const __m512i w1 = _mm512_set1_epi16(1);
    const __m512i w2 = _mm512_set1_epi16(2);    /* doubles as the gap penalty */
#else
    const __m256i v1 = _mm256_set1_epi16(1);
    const __m256i v2 = _mm256_set1_epi16(2);
#endif

    const int nn = n, dmax = 2 * n;
    for (int d = 2; d <= dmax; d++) {
        int ilo  = (d - nn > 1) ? d - nn : 1;
        int ihi  = (d - 1 < nn) ? d - 1 : nn;
        int bas  = nn - d;                      /* brp index = bas + i, always >= 0 */
        int i    = ilo;

#if defined(__AVX512BW__)
        for (; i <= ihi; i += 32) {
            __m256i ca = _mm256_loadu_si256((const __m256i *)(ap + (i - 1)));
            __m256i cb = _mm256_loadu_si256((const __m256i *)(brp + (bas + i)));
            __m512i m  = _mm512_cvtepi8_epi16(_mm256_cmpeq_epi8(ca, cb));
            __m512i s  = _mm512_sub_epi16(_mm512_and_si512(m, w2), w1);   /* +1 / -1 */
            __m512i dg = _mm512_add_epi16(_mm512_loadu_si512((const void *)(d2 + i - 1)), s);
            __m512i up = _mm512_loadu_si512((const void *)(d1 + i - 1));
            __m512i lf = _mm512_loadu_si512((const void *)(d1 + i));
            __m512i r  = _mm512_max_epi16(dg,
                             _mm512_sub_epi16(_mm512_max_epi16(up, lf), w2));
            _mm512_storeu_si512((void *)(d0 + i), r);
        }
#else
        for (; i <= ihi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(ap + (i - 1)));
            __m128i cb = _mm_loadu_si128((const __m128i *)(brp + (bas + i)));
            __m256i m  = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i s  = _mm256_sub_epi16(_mm256_and_si256(m, v2), v1);   /* +1 / -1 */
            __m256i dg = _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(d2 + i - 1)), s);
            __m256i up = _mm256_loadu_si256((const __m256i *)(d1 + i - 1));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(d1 + i));
            __m256i r  = _mm256_max_epi16(dg,
                             _mm256_sub_epi16(_mm256_max_epi16(up, lf), v2));
            _mm256_storeu_si256((__m256i *)(d0 + i), r);
        }
#endif
        /* boundary cells written last: the tail vector may have clobbered index d */
        if (d <= nn) { d0[0] = (int16_t)(-2 * d); d0[d] = (int16_t)(-2 * d); }

        int16_t *t = d2; d2 = d1; d1 = d0; d0 = t;
    }

    int res = (int)d1[nn];   /* after the final rotation d1 == diagonal 2n */
    free(raw);
    return res;
#endif
}
```

## PREDICTION

PREDICTION: 12

(12× faster than the reference `kernel` at n≈1000–4000: ~5 cells/cycle with AVX2 `int16` versus ~3–4 cycles/cell for the scalar reference, plus the elimination of the `4n²`-byte table write traffic. Expected ~20× on an AVX-512BW machine.)

## MEASUREMENT

**Not measured — I must report this plainly.** No tools were available in this session (`alignment_bench` / `alignment_contract` could not be invoked), so I ran zero timing or differential-correctness runs, and I performed zero of the four permitted improvement iterations. The number above is a static cost model, not an observation:

- per 16 cells (AVX2): 2 byte loads, `cmpeq`, `cvtepi8_epi16`, `and`, `sub`, 3 × 256-bit loads, `add`, 2 × `max`, `sub`, 1 store ≈ 13 µops ≈ 3–4 cycles ⇒ ≈ 4–5 cells/cycle;
- reference: serial `dp[i][j-1]` dependency blocks auto-vectorization, ~3–4 cycles/cell plus `4n²` bytes of store traffic (4 MB at n=1000, i.e. out of L2);
- overhead terms modeled: `⌈len/16⌉` rounding per diagonal ⇒ `n²/16 + n` vector iterations, plus `O(n)` setup.

Residual risks I could not retire by measurement, stated honestly: (1) short diagonals near the two corners waste part of a vector, so small-`n` speedup will be lower than 12×; (2) the AVX-512 branch is compile-time-selected and was never executed here; (3) correctness rests on my hand proof of the scratch-lane bound `|v| ≤ 3n` and of "no live cell ever reads a scratch index", not on a differential test against the reference.

## VERDICT

The mechanism is translated faithfully and, I believe, correctly: anti-diagonal wavefront, three rolling buffers, the "team of judges" realized as SIMD lanes across one anti-diagonal, identical recurrence and identical final cell, both sequences read forward, no algorithmic substitution (no banding, no bit-parallel edit-distance trick, no reversal). The two non-obvious implementation choices that make the wavefront actually fast rather than merely parallel-in-principle are indexing diagonals by `i` and pre-reversing `b`, which turn every operand into a unit-stride load, and dropping the `n²` table entirely so the working set is three L1-resident vectors.

I predict ~12×, but I did not verify it, and I did not verify correctness against the reference either — so this artifact should be treated as *unvalidated* until `alignment_contract` and `alignment_bench` are run against it. If the measured speedup lands below ~6×, the first thing I would change is the corner waste (fusing consecutive short diagonals into one vector) and unrolling the diagonal loop 2× to hide the store-to-load latency on `d1`; I would not change the approach.