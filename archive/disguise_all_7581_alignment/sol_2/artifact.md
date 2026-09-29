## APPROACH

**The mapping, piece by piece.**

| Disguised element | Real implementation |
|---|---|
| Scoreboard square (i,j) | DP cell `dp[i][j]` of the Needleman–Wunsch matrix |
| Lone head judge, one square at a time | The reference loop: `dp[i][j]` depends on `dp[i][j-1]`, a serial add→max chain of ≥2 cycles per cell, unvectorizable |
| "A slanting stripe never needs answers from any square on that same stripe" | Anti-diagonal `d = i+j`. Cell (i,j) reads (i−1,j−1) on diagonal *d−2* and (i−1,j),(i,j−1) on diagonal *d−1*. **Zero intra-diagonal dependence.** |
| "Hand every judge a different square on the same stripe; all write at the same moment" | One SIMD lane per judge: 32 cells per instruction (AVX-512BW) or 16 (AVX2), all stored in one `vmovdqu`. The idle townsfolk are the unused vector lanes in the reference. |
| "Once the stripe is filled, the team steps to the next stripe" | Rotate three buffers `p2 ← p1 ← p0`; each holds one anti-diagonal, indexed by absolute row `i`. |
| "The finished scoreboard comes out identical" | Bit-identical score: same max, same order-independent recurrence. |

**Why indexing by row `i` is the whole trick.** Let `D_d[i] = dp[i][d-i]`. Then

```
D_d[i] = max( D_{d-2}[i-1] + s ,  max(D_{d-1}[i-1], D_{d-1}[i]) - 2 )
```

All three neighbours are at offsets 0/−1 in *contiguous* arrays — no gather, no cross-lane shuffle, no shift-in of a scalar. The `max(u,l)-2` factoring saves one subtract per vector.

**Making the letters contiguous too.** Lane `i` compares `a[i-1]` against `b[d-i-1]` — `b` runs *backwards*. So I precompute `rb[k] = b[n-1-k]`; then the needed byte is `rb[i + (n-d)]`, a forward unit-stride load. One `vpcmpeqb` on 32 bytes feeds a mask-blend to ±1, then widens to int16. (This is the "both strings read start to end in the same direction" assumption being broken.)

**int16.** Scores lie in [−2n, n], so int16 is exact for n ≤ 16383 — that doubles the judging team from 16 to 32 per instruction. An int32 anti-diagonal path covers larger n.

**Tails by padding, not by branching.** Buffers and sequence copies are padded 128 elements; the last vector of each stripe overruns harmlessly. I verified by hand that every *valid* read of `p1`/`p2` lands strictly inside the previous stripes' valid ranges, so overrun garbage can never propagate; boundary cells `D_d[0] = D_d[d] = -2d` are written *after* the vector loop, since the overrun can clobber index `d`.

**No OS threads, deliberately.** I costed it: an `omp for` barrier per stripe is ~1 µs ≈ 3500 cycles, while a stripe of length L costs ~0.15·L cycles here. Threads only pay off past L ≈ 2·10⁵, i.e. n ≫ 100 000 — and the *reference* already allocates (n+1)²·4 bytes (1 GB at n=16 000), so no benchmark reaches that. The team of judges is the SIMD lanes; adding a second team would spend more time forming up than judging.

Hand-verified against the reference on n=2, a="AC", b="AG" (score 0), cell by cell, including both boundary writes and the d>n regime.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512BW__) && defined(__AVX512VL__)
#  include <immintrin.h>
#  define NW_W 32
#  define NW_SIMD16 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define NW_W 16
#  define NW_SIMD16 1
#endif

#define NW_PAD 128
#define NW_I16_MAX 16000

/* last-resort fallback: O(n) memory row DP (used only if allocation fails) */
static int nw_rowdp(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int i, j, r = 0;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (j = 0; j <= n; j++) prev[j] = -2 * j;
    for (i = 1; i <= n; i++) {
        cur[0] = -2 * i;
        for (j = 1; j <= n; j++) {
            int dg = prev[j - 1] + (a[i - 1] == b[j - 1] ? 1 : -1);
            int u = prev[j] - 2, l = cur[j - 1] - 2;
            int m = dg; if (u > m) m = u; if (l > m) m = l;
            cur[j] = m;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* anti-diagonal wavefront, int32 (large n / no AVX2); no overrun, plain C */
static int nw_i32(int n, const unsigned char *__restrict pa,
                  const unsigned char *__restrict prb)
{
    size_t sz = (size_t)n + 1 + NW_PAD;
    int *mem = (int *)calloc(3 * sz, sizeof(int));
    int *p2, *p1, *p0;
    int d, res = 0, dmax = 2 * n;
    if (!mem) return 0;
    p2 = mem; p1 = mem + sz; p0 = mem + 2 * sz;
    p2[0] = 0; p1[0] = -2; p1[1] = -2;
    for (d = 2; d <= dmax; d++) {
        int lo = d - n, hi = d - 1, off = n - d, i;
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        for (i = lo; i <= hi; i++) {
            int u = p1[i - 1], l = p1[i];
            int g = (u > l ? u : l) - 2;
            int dg = p2[i - 1] + (pa[i - 1] == prb[i + off] ? 1 : -1);
            p0[i] = g > dg ? g : dg;
        }
        if (d <= n) { p0[0] = -2 * d; p0[d] = -2 * d; }
        if (d == dmax) res = p0[n];
        { int *t = p2; p2 = p1; p1 = p0; p0 = t; }
    }
    free(mem);
    return res;
}

#ifdef NW_SIMD16
/* anti-diagonal wavefront, int16 lanes: the whole stripe judged at once */
static int nw_i16(int n, const unsigned char *__restrict pa,
                  const unsigned char *__restrict prb)
{
    size_t sz = (size_t)n + 1 + NW_PAD;
    short *mem = (short *)calloc(3 * sz, sizeof(short));
    short *p2, *p1, *p0;
    int d, res = 0, dmax = 2 * n;
    if (!mem) return nw_i32(n, pa, prb);
    p2 = mem; p1 = mem + sz; p0 = mem + 2 * sz;
    p2[0] = 0; p1[0] = -2; p1[1] = -2;
#if NW_W == 32
    {
    const __m512i vtwo = _mm512_set1_epi16(2);
    const __m512i vp1v = _mm512_set1_epi16(1);
    const __m512i vm1v = _mm512_set1_epi16(-1);
#else
    {
    const __m256i vtwo = _mm256_set1_epi16(2);
    const __m128i bp1  = _mm_set1_epi8(1);
    const __m128i bm1  = _mm_set1_epi8(-1);
#endif
    for (d = 2; d <= dmax; d++) {
        int lo = d - n, hi = d - 1, off = n - d, i;
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        for (i = lo; i <= hi; i += NW_W) {
#if NW_W == 32
            __m256i av = _mm256_loadu_si256((const __m256i *)(pa + (i - 1)));
            __m256i bv = _mm256_loadu_si256((const __m256i *)(prb + (i + off)));
            __mmask32 keq = _mm256_cmpeq_epi8_mask(av, bv);
            __m512i sv = _mm512_mask_blend_epi16(keq, vm1v, vp1v);
            __m512i x2 = _mm512_loadu_si512((const void *)(p2 + (i - 1)));
            __m512i xu = _mm512_loadu_si512((const void *)(p1 + (i - 1)));
            __m512i xl = _mm512_loadu_si512((const void *)(p1 + i));
            __m512i g  = _mm512_sub_epi16(_mm512_max_epi16(xu, xl), vtwo);
            __m512i dg = _mm512_add_epi16(x2, sv);
            _mm512_storeu_si512((void *)(p0 + i), _mm512_max_epi16(g, dg));
#else
            __m128i av = _mm_loadu_si128((const __m128i *)(pa + (i - 1)));
            __m128i bv = _mm_loadu_si128((const __m128i *)(prb + (i + off)));
            __m128i eq = _mm_cmpeq_epi8(av, bv);
            __m256i sv = _mm256_cvtepi8_epi16(_mm_blendv_epi8(bm1, bp1, eq));
            __m256i x2 = _mm256_loadu_si256((const __m256i *)(p2 + (i - 1)));
            __m256i xu = _mm256_loadu_si256((const __m256i *)(p1 + (i - 1)));
            __m256i xl = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i g  = _mm256_sub_epi16(_mm256_max_epi16(xu, xl), vtwo);
            __m256i dg = _mm256_add_epi16(x2, sv);
            _mm256_storeu_si256((__m256i *)(p0 + i), _mm256_max_epi16(g, dg));
#endif
        }
        /* boundaries written AFTER the loop: the tail overrun may touch index d */
        if (d <= n) { short v = (short)(-2 * d); p0[0] = v; p0[d] = v; }
        if (d == dmax) res = (int)p0[n];
        { short *t = p2; p2 = p1; p1 = p0; p0 = t; }
    }
    }
    free(mem);
    return res;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    size_t cs;
    unsigned char *pa, *prb;
    int k, r;
    if (n <= 0) return 0;
    cs = (size_t)n + NW_PAD;
    pa  = (unsigned char *)calloc(cs, 1);
    prb = (unsigned char *)calloc(cs, 1);
    if (!pa || !prb) { free(pa); free(prb); return nw_rowdp(n, a, b); }
    memcpy(pa, a, (size_t)n);
    for (k = 0; k < n; k++) prb[k] = (unsigned char)b[n - 1 - k];
#ifdef NW_SIMD16
    r = (n <= NW_I16_MAX) ? nw_i16(n, pa, prb) : nw_i32(n, pa, prb);
#else
    r = nw_i32(n, pa, prb);
#endif
    free(pa); free(prb);
    return r;
}
```

## PREDICTION: 12

(Speedup factor over the reference kernel. Budget behind it: reference ≈ 2.5 cycles/cell — a 2-cycle add→max loop-carried chain on `dp[i][j-1]` plus (n+1)² ints of first-touch page faults; mine ≈ 13 µops per 16-cell vector ≈ 3 cycles ≈ 0.19 cycles/cell on AVX2, ~0.10 on AVX-512BW, with a 3·(n+1)·2-byte working set that stays in L1/L2. 2.5/0.19 ≈ 13; I shade to 12 for stripe-tail waste and the ~64n wasted overrun cells.)

## MEASUREMENT

**Not performed — I could not measure.** The task listed `alignment_bench` and `alignment_contract`, but this session was explicitly delivered with no tools available, so I never executed a single compile or timing run. I am reporting that plainly rather than inventing numbers: **there are zero measured data points behind the artifact above, and the "at most four improvement rounds" budget went entirely unused.**

What I did instead, and what it is worth:

- **Hand-traced correctness** on n=2 (`a="AC"`, `b="AG"`), all six interior/boundary cells, ending at the correct score 0. Also traced n=1 (score ±1) and the `d = n+1`, `d = n+2` transitions where the stripe stops touching the matrix border.
- **Proved the padding is safe by index arithmetic**, not by hoping: max touched indices are `p0[hi+31] ≤ n+31` (arrays `n+1+128`), `pa[hi+30] ≤ n+30` and `prb[hi+30] ≤ n+30` (arrays `n+128`); and every *valid* read of `p1`/`p2` is shown to fall inside the previous two stripes' valid ranges, so tail garbage cannot propagate into a real cell.
- The cycle budget above is an analytic port-pressure estimate, not a measurement.

Unverified risks I am not going to paper over: the AVX-512 branch (`_mm256_cmpeq_epi8_mask` / `_mm512_mask_blend_epi16`) has never been compiled; if `-march=native` selects it and I have a blend-polarity error there, the AVX-512 machine returns wrong answers while an AVX2 machine is fine. The `NW_I16_MAX = 16000` int16/int32 crossover is likewise untested, though the reference's own (n+1)²·4-byte allocation (1 GB at n=16 000) makes it unlikely to be exercised.

## VERDICT

The mechanism translates cleanly and, I believe, faithfully: the "stripe with no internal dependencies" is exactly the anti-diagonal `i+j = d`, and the "team of judges writing at the same moment" is exactly a vector register, with the row-indexed three-buffer layout chosen specifically so the three neighbours land at offsets 0/−1 in contiguous memory — no shuffles, no gathers, and no dependency *within* a stripe, which is what lets the out-of-order engine keep many vectors in flight.

But the honest verdict on the deliverable is: **this is a predicted 12×, not a demonstrated one.** The artifact is complete and contract-exact (`int kernel(int, const char*, const char*)`, NW +1/−1/−2, identical score), and I stand behind the correctness reasoning, which is arithmetic rather than optimism. I do not get to claim the speed number until something compiles and times it.

One design decision I want on the record because it cuts *against* the disguised story's imagery: I refused to add OpenMP threads. The story says the crowd of idle townsfolk should be put to work, and a per-stripe `omp for` is the literal reading — but the barrier costs ~3500 cycles while a stripe costs ~0.15·L, so threads lose until n ≫ 100 000, a size the reference itself cannot reach without a multi-gigabyte allocation. The correct team of judges here is 32 SIMD lanes, not 8 cores. If the benchmark turns out to use enormous n, the right fix is a *tiled* wavefront (block anti-diagonals over threads, this same kernel inside each tile) — still the same mechanism, one level up — and that is the first thing I would build with a working bench.