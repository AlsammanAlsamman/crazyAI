# APPROACH

**Mapping the disguised solution onto the real problem.** The bead strings are the two DNA sequences; "same colour = +1 / different = −1 / pretend-a-bead-is-missing = −2" is exactly `MATCH=+1, MISMATCH=-1, GAP=-2`. The "big square grid" is the full (n+1)×(n+1) Needleman–Wunsch table, the corner start is the `-2j` / `-2i` boundary initialisation, and "built only from above, left, and above-left" is the three-way recurrence. The crucial constraint — *a gap can only be discovered by having already finished the square right before it* — is precisely the statement that no banding, no pruning, no closed form: every cell must be produced by its three already-computed predecessors. I keep that literally: **every one of the n² cells is computed, from exactly those three neighbours, and never before all three exist.**

What I change is only *which* cells are worked on simultaneously, which the dependency rule itself permits. In row-major order the `left` neighbour is a strict serial chain — one cell per step, forever. The fix that does *not* alter the mechanism: sweep the grid by **anti-diagonals**. On diagonal `d = i+j`, the three predecessors of every cell live on diagonals `d-1` and `d-2`, so all cells on a diagonal are mutually independent while still each strictly following their own three predecessors. The order of "discovery" is preserved cell-for-cell; only the order between *unrelated* cells changes, so the result is bit-identical to the reference.

Concretely:
- Three rolling `int16` diagonal buffers (`p2`,`p1`,`cu`), indexed by `i`. `cu[i] = max(p2[i-1] + s, max(p1[i-1], p1[i]) - 2)`, where `p1[i-1]` is the *up* neighbour and `p1[i]` is the *left* neighbour. No prefix-scan, no loop-carried scalar chain at all.
- `b` is stored **reversed** (`br[k] = b[n-1-k]`), which makes the character pair for consecutive `i` on a diagonal two *contiguous* byte loads (`a[i-1..]` and `br[n-d+i..]`); this is the only trick needed to keep the score lookup vectorisable.
- Scores come from `vpcmpeqb` + `vpmovsxbw`: mask is `-1` on match, so `diag + score = diag - 2*mask - 1` in two integer ops.
- Boundary cells (`i=0` and `j=0`, i.e. `-2d`) are written explicitly after each diagonal's vector sweep — that is the same `-2j` initialisation the grid corner rule gives.
- 32 cells per iteration (2× AVX2 `epi16`), over-covering past the diagonal end into padding. I proved the over-run positions (`> min(n,d)`) are never read back as valid, so the garbage lanes are inert.

Cost model: 5 loads + 1 store + 7 ALU ops per 16 cells ⇒ **load-port bound at ~2.5 cycles / 16 cells ≈ 0.16 cycles/cell**, versus the reference's serial `left` dependency (`add`+`max` ≈ 2–3 cycles/cell) plus an O(n²) 4-byte table it must stream to memory. `int16` is exact because |H| ≤ 2n; the fast path is gated to `n ≤ 16000` (and `n ≥ 64`), with an O(n)-memory scalar DP — itself faster than the reference — as the fallback.

Single-threaded deliberately: OpenMP over a diagonal would need 2n barriers, which costs more than the whole computation. A tiled wavefront is the next step, not this one.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* O(n)-memory exact Needleman-Wunsch; used for tiny n, huge n, or no AVX2. */
static int nw_scalar(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int r = 0;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = -2 * j;
    for (int i = 1; i <= n; i++) {
        const char ai = a[i - 1];
        cur[0] = -2 * i;
        for (int j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? 1 : -1);
            int u = prev[j] - 2;
            int l = cur[j - 1] - 2;
            int m = d > u ? d : u;
            if (l > m) m = l;
            cur[j] = m;
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
    if (n < 64 || n > 16000) return nw_scalar(n, a, b);
    {
        size_t bs = (((size_t)n + 96) + 31) & ~(size_t)31;   /* bytes, mult of 32  */
        size_t ss = (((size_t)n + 96) + 15) & ~(size_t)15;   /* shorts, mult of 16 */
        unsigned char *raw = (unsigned char *)malloc(2 * bs + 6 * ss + 64);
        unsigned char *base;
        char *pa, *pbr;
        short *d2, *d1, *d0, *p2, *p1, *cu;
        int d, res;
        const __m256i vone = _mm256_set1_epi16(1);
        const __m256i vtwo = _mm256_set1_epi16(2);

        if (!raw) return nw_scalar(n, a, b);
        base = (unsigned char *)(((uintptr_t)raw + 31) & ~(uintptr_t)31);
        pa  = (char *)base;
        pbr = (char *)(base + bs);
        d2  = (short *)(base + 2 * bs);
        d1  = (short *)(base + 2 * bs + 2 * ss);
        d0  = (short *)(base + 2 * bs + 4 * ss);

        memcpy(pa, a, (size_t)n);
        memset(pa + n, 0, bs - (size_t)n);
        for (int k = 0; k < n; k++) pbr[k] = b[n - 1 - k];
        memset(pbr + n, 1, bs - (size_t)n);
        memset(d2, 0, 2 * ss);
        memset(d1, 0, 2 * ss);
        memset(d0, 0, 2 * ss);

        p2 = d2; p1 = d1; cu = d0;
        p2[0] = 0;                       /* diagonal 0: H[0][0]            */
        p1[0] = -2; p1[1] = -2;          /* diagonal 1: H[0][1], H[1][0]   */

        for (d = 2; d <= 2 * n; d++) {
            int ilo = d - n; if (ilo < 1) ilo = 1;
            int ihi = d - 1; if (ihi > n) ihi = n;
            {
                const char  *ap = pa  - 1;          /* index by i -> a[i-1]      */
                const char  *bp = pbr + (n - d);    /* index by i -> b[d-i-1]    */
                const short *q2 = p2  - 1;          /* index by i -> H[i-1][j-1] */
                const short *qu = p1  - 1;          /* index by i -> H[i-1][j]   */
                const short *ql = p1;               /* index by i -> H[i][j-1]   */
                int i;
                for (i = ilo; i <= ihi; i += 32) {
                    __m256i m0 = _mm256_cvtepi8_epi16(
                        _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(ap + i)),
                                       _mm_loadu_si128((const __m128i *)(bp + i))));
                    __m256i m1 = _mm256_cvtepi8_epi16(
                        _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(ap + i + 16)),
                                       _mm_loadu_si128((const __m128i *)(bp + i + 16))));
                    __m256i g0 = _mm256_loadu_si256((const __m256i *)(q2 + i));
                    __m256i g1 = _mm256_loadu_si256((const __m256i *)(q2 + i + 16));
                    __m256i u0 = _mm256_loadu_si256((const __m256i *)(qu + i));
                    __m256i u1 = _mm256_loadu_si256((const __m256i *)(qu + i + 16));
                    __m256i l0 = _mm256_loadu_si256((const __m256i *)(ql + i));
                    __m256i l1 = _mm256_loadu_si256((const __m256i *)(ql + i + 16));
                    /* diag + (match? +1 : -1)  ==  diag - 2*mask - 1 */
                    __m256i s0 = _mm256_sub_epi16(
                                     _mm256_sub_epi16(g0, _mm256_add_epi16(m0, m0)), vone);
                    __m256i s1 = _mm256_sub_epi16(
                                     _mm256_sub_epi16(g1, _mm256_add_epi16(m1, m1)), vone);
                    __m256i h0 = _mm256_sub_epi16(_mm256_max_epi16(u0, l0), vtwo);
                    __m256i h1 = _mm256_sub_epi16(_mm256_max_epi16(u1, l1), vtwo);
                    _mm256_storeu_si256((__m256i *)(cu + i),      _mm256_max_epi16(s0, h0));
                    _mm256_storeu_si256((__m256i *)(cu + i + 16), _mm256_max_epi16(s1, h1));
                }
            }
            if (d <= n) {                    /* grid edges: H[0][d] = H[d][0] = -2d */
                cu[0] = (short)(-2 * d);
                cu[d] = (short)(-2 * d);
            }
            { short *t = p2; p2 = p1; p1 = cu; cu = t; }
        }

        res = p1[n];                         /* H[n][n] on diagonal 2n */
        free(raw);
        return res;
    }
#else
    return nw_scalar(n, a, b);
#endif
}
```

# PREDICTION

PREDICTION: 14

(Meaning: ~14× faster than the reference O(n²) row-major DP at benchmark-typical n, from ~0.16 cycles/cell against the reference's ~2–3 cycles/cell serial `left` chain plus its O(n²) table traffic.)

# MEASUREMENT

**Not performed — and I am reporting that plainly rather than inventing numbers.** This session was launched with no tools available: `alignment_bench` and `alignment_contract` could not be invoked, and there is no compiler run behind this answer. The prediction above was therefore never checked, and I did not get the four permitted improvement rounds; the artifact is the result of static reasoning only (uop/port accounting per 16 cells, plus a hand proof of index bounds and of the claim that the SIMD over-run lanes are never read back).

What I *did* verify by hand, since I could not verify it by execution:
- index algebra of the anti-diagonal mapping (`p1[i-1]`=up, `p1[i]`=left, `p2[i-1]`=diag, `pbr[n-d+i]`=`b[d-i-1]`), checked symbolically and re-checked on `d=2`;
- the interior range `[max(1,d-n), min(n,d-1)]` is non-empty for all `d∈[2,2n]`, and every load it issues lands inside the written region of the two previous diagonals;
- the highest index any later diagonal reads from diagonal `d` is `min(n,d)`, which is either computed in-range or explicitly overwritten by the `-2d` boundary store — so the 32-wide over-run cannot contaminate a real cell;
- `int16` exactness: |H| ≤ 2n, intermediates ≥ −2n−2 and ≤ n+1, safe for the gated `n ≤ 16000`.

Residual risk I cannot discharge without a run: an off-by-one in the boundary writes would show up as a wrong score on the first benchmark case, and the `n<64` / `n>16000` gates route to the scalar path, so a bug there would be silent-but-slow rather than wrong.

# VERDICT

The mechanism is preserved, not replaced: the full grid is still filled, every cell still derived from its above / left / above-left predecessors, gaps still only "discovered" from the already-finished adjacent cell. The single change is traversal order — anti-diagonals instead of rows — which is a legal linearisation of the identical dependency DAG and therefore bit-identical, while removing the one-cell-per-step serial chain that makes the row-major version slow. Everything else is implementation quality: reversed `b` to make the score lookup contiguous, `int16` lanes, 32 cells per iteration, O(n) memory instead of the reference's O(n²) table.

Claimed: ~14×. Demonstrated: nothing — no measurement was possible here, so treat the number as an unvalidated prediction, not a result. Run `alignment_contract` then `alignment_bench` against this artifact before believing it.

If it validates, the clear next move is a tiled wavefront under OpenMP (tiles of ~256×256, parallel over tile anti-diagonals, boundary rows/columns exchanged through shared arrays), gated to `n ≥ 2048` so the ~31 barriers stay cheap relative to the work — plausibly another 4–6× on a typical core count. A bare `#pragma omp parallel for` over a single diagonal is *not* that move and would be a regression: 2n barriers cost more than the entire computation.