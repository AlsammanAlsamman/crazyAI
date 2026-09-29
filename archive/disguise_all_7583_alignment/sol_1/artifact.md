# APPROACH

**How the disguised solution maps onto the real problem**

| Parade rulebook | Real DP |
|---|---|
| Line A / Line B children, front→back | strings `a`, `b`, indices 1..n, both read start→end |
| flag colors red/blue/green/yellow | bases A/C/G/T |
| pair earns a point / loses a point | match `+1` / mismatch `-1` |
| a "slip" (child steps out, everyone shifts) | a gap, `-2` |
| the giant scoreboard, every A-child × every B-child | the full `(n+1)×(n+1)` NW matrix — **every** cell, including the far-corner ones |
| each box needs the box above, to the left, and above-left | `D[i][j] = max(D[i-1][j-1]+s, D[i-1][j]-2, D[i][j-1]-2)` |
| read the answer from the last box, bottom corner | return `D[n][n]` |

I keep the mechanism **exactly**: same recurrence, same full grid, every cell of the (n+1)² table is evaluated, same direction, no band, no early exit, no algorithmic substitution. What changes is only the *order in which the judge walks the boxes* — which the rulebook's own dependency statement permits: a box only needs up, left, and above-left. Row-major is one legal order; **anti-diagonal (wavefront) order** is another. Along `i+j = d`, all cells are mutually independent, so a whole anti-diagonal is computed in parallel by SIMD while the three dependencies are read from the two previous anti-diagonals.

Implementation details that buy the speed:

1. **Anti-diagonal wavefront, 3 rolling buffers of length `n+1`** instead of an `(n+1)²` table. Reference allocates and streams `4(n+1)²` bytes to DRAM (64 MB at n=4000); mine has an O(n) working set that lives in L1/L2. `D[i-1][j-1]` is `prev2[i-1]`, `D[i-1][j]` is `prev1[i-1]`, `D[i][j-1]` is `prev1[i]` — all contiguous, unit-stride.
2. **Reversed copy of `b`.** Along a diagonal `i` rises while `j` falls, which would make `b` accesses descend; with `B[k]=b[n-1-k]`, the needed byte is `B[n-d+i]` — so both character streams are contiguous ascending loads, no gather, no shuffle.
3. **16-bit lanes.** `|D| ≤ 2n`, so for `n ≤ 16000` everything fits in `int16` → 16 cells/vector (AVX2) or 32 cells/vector (AVX-512BW), ~10 instructions per vector. A 32-bit path covers larger `n`.
4. `match/mismatch` vector built branchlessly: byte compare → sign-extend to `{0,-1}` → `s = (-1) - 2t ∈ {-1,+1}`.
5. Buffers/character copies padded by 64 so the vector loop may harmlessly overshoot `hi`; I verified the overshoot lanes are never read back as valid cells (the read window of diagonal `e` at step `e+1` is exactly `[max(0,e-n), min(n,e)]`, precisely the written window).
6. Boundary cells `D[i][0]=-2i`, `D[0][j]=-2j` are written after the interior sweep of each diagonal so overshoot cannot clobber them.

No threading: a per-diagonal OpenMP barrier costs ~1 µs × 2n barriers, which dwarfs the ~n²/32 vector ops; a blocked pipelined wavefront would be the only viable threading and isn't worth the correctness risk here.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <immintrin.h>
#endif

/* ---- 16-bit anti-diagonal wavefront: valid while 2n+2 < 32767 ---- */
static int nw_i16(int n, const char *a, const char *b)
{
    const int PAD = 64;
    size_t blen = (size_t)n + 1 + (size_t)PAD;
    int16_t *mem = (int16_t *)malloc(3 * blen * sizeof(int16_t));
    char *cbuf = (char *)malloc(2 * ((size_t)n + (size_t)PAD));
    if (!mem || !cbuf) { free(mem); free(cbuf); return 0; }

    int16_t *p2 = mem, *p1 = mem + blen, *cu = mem + 2 * blen;
    char *A = cbuf, *B = cbuf + (size_t)n + PAD;

    memcpy(A, a, (size_t)n);
    memset(A + n, 0x00, PAD);
    for (int i = 0; i < n; i++) B[i] = b[n - 1 - i];   /* reversed b */
    memset(B + n, 0x7F, PAD);

    p2[0] = 0;                 /* diagonal 0: D[0][0] */
    p1[0] = -2; p1[1] = -2;    /* diagonal 1: D[0][1], D[1][0] */

    const int dmax = 2 * n;
    for (int d = 2; d <= dmax; d++) {
        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        int off = n - d;
        int i = lo;

#if defined(__AVX512BW__) && defined(__AVX512F__)
        {
            const __m512i m1 = _mm512_set1_epi16(-1);
            const __m512i g2 = _mm512_set1_epi16(2);
            for (; i <= hi; i += 32) {
                __m256i ca = _mm256_loadu_si256((const __m256i *)(A + i - 1));
                __m256i cbv = _mm256_loadu_si256((const __m256i *)(B + off + i));
                __m256i eq = _mm256_cmpeq_epi8(ca, cbv);
                __m512i t = _mm512_cvtepi8_epi16(eq);              /* 0 or -1 */
                __m512i s = _mm512_sub_epi16(m1, _mm512_add_epi16(t, t));
                __m512i dg = _mm512_loadu_si512((const void *)(p2 + i - 1));
                __m512i uu = _mm512_loadu_si512((const void *)(p1 + i - 1));
                __m512i ll = _mm512_loadu_si512((const void *)(p1 + i));
                __m512i best = _mm512_max_epi16(
                        _mm512_add_epi16(dg, s),
                        _mm512_sub_epi16(_mm512_max_epi16(uu, ll), g2));
                _mm512_storeu_si512((void *)(cu + i), best);
            }
        }
#elif defined(__AVX2__)
        {
            const __m256i m1 = _mm256_set1_epi16(-1);
            const __m256i g2 = _mm256_set1_epi16(2);
            for (; i <= hi; i += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(A + i - 1));
                __m128i cbv = _mm_loadu_si128((const __m128i *)(B + off + i));
                __m128i eq = _mm_cmpeq_epi8(ca, cbv);
                __m256i t = _mm256_cvtepi8_epi16(eq);              /* 0 or -1 */
                __m256i s = _mm256_sub_epi16(m1, _mm256_add_epi16(t, t));
                __m256i dg = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i uu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i ll = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m256i best = _mm256_max_epi16(
                        _mm256_add_epi16(dg, s),
                        _mm256_sub_epi16(_mm256_max_epi16(uu, ll), g2));
                _mm256_storeu_si256((__m256i *)(cu + i), best);
            }
        }
#elif defined(__SSE4_1__)
        {
            const __m128i m1 = _mm_set1_epi16(-1);
            const __m128i g2 = _mm_set1_epi16(2);
            for (; i <= hi; i += 8) {
                __m128i ca = _mm_loadl_epi64((const __m128i *)(A + i - 1));
                __m128i cbv = _mm_loadl_epi64((const __m128i *)(B + off + i));
                __m128i eq = _mm_cmpeq_epi8(ca, cbv);
                __m128i t = _mm_cvtepi8_epi16(eq);
                __m128i s = _mm_sub_epi16(m1, _mm_add_epi16(t, t));
                __m128i dg = _mm_loadu_si128((const __m128i *)(p2 + i - 1));
                __m128i uu = _mm_loadu_si128((const __m128i *)(p1 + i - 1));
                __m128i ll = _mm_loadu_si128((const __m128i *)(p1 + i));
                __m128i best = _mm_max_epi16(
                        _mm_add_epi16(dg, s),
                        _mm_sub_epi16(_mm_max_epi16(uu, ll), g2));
                _mm_storeu_si128((__m128i *)(cu + i), best);
            }
        }
#endif
        for (; i <= hi; i++) {                         /* scalar tail */
            int sc = (A[i - 1] == B[off + i]) ? 1 : -1;
            int v = (int)p2[i - 1] + sc;
            int u = (int)p1[i - 1];
            int l = (int)p1[i];
            int m = (u > l ? u : l) - 2;
            cu[i] = (int16_t)(v > m ? v : m);
        }
        if (d <= n) {                                  /* grid boundaries */
            int16_t bv = (int16_t)(-2 * d);
            cu[0] = bv;
            cu[d] = bv;
        }
        { int16_t *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }
    }

    int res = (int)p1[n];
    free(mem); free(cbuf);
    return res;
}

/* ---- 32-bit anti-diagonal wavefront: large n ---- */
static int nw_i32(int n, const char *a, const char *b)
{
    const int PAD = 64;
    size_t blen = (size_t)n + 1 + (size_t)PAD;
    int32_t *mem = (int32_t *)malloc(3 * blen * sizeof(int32_t));
    char *cbuf = (char *)malloc(2 * ((size_t)n + (size_t)PAD));
    if (!mem || !cbuf) { free(mem); free(cbuf); return 0; }

    int32_t *p2 = mem, *p1 = mem + blen, *cu = mem + 2 * blen;
    char *A = cbuf, *B = cbuf + (size_t)n + PAD;

    memcpy(A, a, (size_t)n);
    memset(A + n, 0x00, PAD);
    for (int i = 0; i < n; i++) B[i] = b[n - 1 - i];
    memset(B + n, 0x7F, PAD);

    p2[0] = 0;
    p1[0] = -2; p1[1] = -2;

    const int dmax = 2 * n;
    for (int d = 2; d <= dmax; d++) {
        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        int off = n - d;
        int i = lo;

#if defined(__AVX2__)
        {
            const __m256i m1 = _mm256_set1_epi32(-1);
            const __m256i g2 = _mm256_set1_epi32(2);
            for (; i <= hi; i += 8) {
                __m128i ca = _mm_loadl_epi64((const __m128i *)(A + i - 1));
                __m128i cbv = _mm_loadl_epi64((const __m128i *)(B + off + i));
                __m128i eq = _mm_cmpeq_epi8(ca, cbv);
                __m256i t = _mm256_cvtepi8_epi32(eq);
                __m256i s = _mm256_sub_epi32(m1, _mm256_add_epi32(t, t));
                __m256i dg = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i uu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i ll = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m256i best = _mm256_max_epi32(
                        _mm256_add_epi32(dg, s),
                        _mm256_sub_epi32(_mm256_max_epi32(uu, ll), g2));
                _mm256_storeu_si256((__m256i *)(cu + i), best);
            }
        }
#endif
        for (; i <= hi; i++) {
            int sc = (A[i - 1] == B[off + i]) ? 1 : -1;
            int v = p2[i - 1] + sc;
            int u = p1[i - 1];
            int l = p1[i];
            int m = (u > l ? u : l) - 2;
            cu[i] = (v > m ? v : m);
        }
        if (d <= n) { cu[0] = -2 * d; cu[d] = -2 * d; }
        { int32_t *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }
    }

    int res = p1[n];
    free(mem); free(cbuf);
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n <= 16000) return nw_i16(n, a, b);   /* |D| <= 2n fits int16 */
    return nw_i32(n, a, b);
}
```

# PREDICTION

PREDICTION: 15

(Stated before any measurement: ~15× speedup over the reference aligner. Reasoning: the reference is limited by a 2–3 cycle serial `add→max` dependency chain per cell along each row plus streaming `4(n+1)²` bytes of table to memory; the wavefront does 16–32 cells per ~10 instructions with an O(n) cache-resident working set, i.e. ~0.2–0.4 cycles/cell. Range I'd accept as consistent: 8×–25×, with the low end if only SSE4.1 is available or if the grid is small enough that per-diagonal loop overhead dominates.)

# MEASUREMENT

Not obtained in this session — no tools were available to me here (`alignment_bench` / `alignment_contract` could not be invoked), so I have **no measured numbers** and will not invent any. What I did verify by hand instead:

- **Recurrence identity.** `cu[i] = max(p2[i-1]+s, max(p1[i-1], p1[i]) - 2)` with `p2`=diag `d-2`, `p1`=diag `d-1` expands to exactly `max(D[i-1][j-1]+s, D[i-1][j]-2, D[i][j-1]-2)` for `j = d-i`.
- **Character indexing.** `B[n-d+i] = b[d-i-1] = b[j-1]`; index stays in `[0, n-1]` over the whole interior range.
- **Overshoot safety.** Vector writes reach at most `hi+VLEN-1 ≤ n+31`; buffers and both char copies are padded by 64. The read window of diagonal `e` on the next step is exactly `[max(0,e-n), min(n,e)]`, which is precisely the range written as valid cells — garbage lanes are provably never consumed (they may wrap in `int16`, which is defined for the intrinsics and inconsequential).
- **Small cases.** `n=1`: `max(s, -4) = s` → `+1` / `-1`, matching the reference. `n=0` → 0. Boundary rows/columns `-2i`, `-2j` seeded on diagonals 0 and 1 and maintained for every `d ≤ n`.
- **Range.** `|D[i][j]| ≤ 2n`, and the worst intermediate is `-2n-2`, so the `int16` path is exact for `n ≤ 16000`; beyond that the `int32` path takes over.

# VERDICT

**Mechanism: preserved, not replaced.** Every cell of the full (n+1)² grid is still computed with the identical three-neighbour max recurrence, both sequences read front-to-back, answer read from the bottom-right corner. Only the traversal order changed from row-major to anti-diagonal — a reordering the rulebook's own dependency rule explicitly licenses — which is what unlocks SIMD, plus the table is collapsed to three rolling anti-diagonals since nothing older than `d-2` is ever read.

**Confidence: high on correctness, unmeasured on speed.** I predicted 15×; I could not run the benchmark here, so that prediction stands untested — treat it as a claim, not a result. The honest failure mode to watch for: if the harness reports a speedup well under ~5×, the most likely causes in order are (a) the build lacks AVX2 so only the SSE4.1/scalar path ran, (b) `n` is small enough (≲128) that per-diagonal loop setup dominates the ~n/2-cell diagonals, or (c) at large `n` the three buffers spill out of L2 and the kernel becomes bandwidth-bound — the fix for (c) within this same mechanism is anti-diagonal blocking (tile the wavefront into cache-sized parallelograms), not a change of algorithm.