## APPROACH

**Mapping the disguised solution onto Needleman–Wunsch.**

| Bead-string story | Real kernel |
|---|---|
| Square grid, Mia's beads across / Leo's down | The `(n+1)×(n+1)` NW DP table, `a` on `i`, `b` on `j` |
| +1 same, −1 different, −2 for a "missing bead" | `MATCH=+1`, `MISMATCH=−1`, `GAP=−2` |
| A square is built only from above / left / above-left | `dp[i][j] = max(dp[i-1][j-1]+s, dp[i-1][j]-2, dp[i][j-1]-2)` — untouched, so the score is bit-identical to the reference |
| "A slanted line of squares never needs each other" | Anti-diagonal `d = i+j`: cells on one anti-diagonal are mutually independent; up/left live on `d−1`, diagonal lives on `d−2` |
| "A row of helpers, one per square, all at the same moment" | **SIMD lanes are the helpers.** One vector instruction fills 16 (AVX2) or 32 (AVX-512BW) cells of the anti-diagonal simultaneously |
| "The next slanted line steps in using only what the previous line finished" | Three rotating `O(n)` buffers `pp (d−2)`, `p (d−1)`, `cur (d)` |
| "You must walk the grid, never spot a missing bead directly" | No banding, no score-range shortcut, no heuristic — every cell is still computed |

Indexing algebra that makes it fast: on anti-diagonal `d`, writing `cur[i] = dp[i][d-i]`, the recurrence becomes

```
cur[i] = max( pp[i-1] + s(a[i-1], b[d-i-1]),  max(p[i-1], p[i]) - 2 )
```

All three DP operands are **contiguous, unit-stride** loads (offsets 0 and −1). The only awkward term is the character comparison, since `b`'s index *decreases* as `i` increases. Fixing that: precompute `br[k] = b[n-1-k]`, then `b[d-i-1] = br[(n-d)+i]` — also contiguous and increasing. So the whole wavefront is pure streaming vector work: two byte loads → `pcmpeqb` → widen to lanes → blend ±1 → 2 adds, 2 maxes, 1 store per vector of cells.

Two integer widths: `int16` lanes when `n ≤ 16000` (score range is `[-2n, n]`, so `-32002` is the worst intermediate — safely inside `int16`), `int32` otherwise. Diagonal tails are handled by letting the final vector store *spill* past the diagonal's valid end into padding; I proved that spilled indices `> min(n,d)` are never read by diagonals `d+1` or `d+2`, and the two boundary cells (`i=0`, `i=d`) are written *after* the spill, so the spill is harmless and there is no scalar tail loop at all.

Memory is `O(n)` (three diagonals) instead of the reference's `O(n²)` table — for large `n` the reference is DRAM-bound writing hundreds of MB while this stays in L1/L2. That is a second, independent source of the speedup.

OpenMP across the diagonal was considered and rejected on purpose: it would need `2n` barriers, and one diagonal at `n=20000` is only ~0.7 µs of work — barrier-dominated. The "helpers" are lanes, which is the cheap realization of the same idea.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------- scalar two-row fallback (tiny n / no AVX2 / alloc failure) ---------- */
static int nw_scalar(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = -2 * j;
    for (int i = 1; i <= n; i++) {
        cur[0] = -2 * i;
        const char ai = a[i - 1];
        for (int j = 1; j <= n; j++) {
            int diag = prev[j - 1] + (ai == b[j - 1] ? 1 : -1);
            int up   = prev[j] - 2;
            int left = cur[j - 1] - 2;
            int best = diag > up ? diag : up;
            if (left > best) best = left;
            cur[j] = best;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(prev); free(cur);
    return r;
}

#if defined(__AVX2__)

#if defined(__AVX512BW__) && defined(__AVX512VL__)
#define NW_AVX512 1
#define NW_W16 32
#define NW_W32 16
#else
#define NW_AVX512 0
#define NW_W16 16
#define NW_W32 8
#endif

/* A   : padded copy of a
   BR  : padded reversed copy of b, BR[k] = b[n-1-k]
   mem : 3*stride cells of scratch, stride >= n + 129            */
static int nw_wave16(int n, const char *A, const char *BR,
                     int16_t *mem, int stride)
{
    int16_t *p0 = mem, *p1 = mem + stride, *p2 = mem + 2 * stride;
#if NW_AVX512
    const __m512i vp1  = _mm512_set1_epi16(1);
    const __m512i vm1  = _mm512_set1_epi16(-1);
    const __m512i vgap = _mm512_set1_epi16(-2);
#else
    const __m256i vp1  = _mm256_set1_epi16(1);
    const __m256i vm1  = _mm256_set1_epi16(-1);
    const __m256i vgap = _mm256_set1_epi16(-2);
#endif
    const int lim = 2 * n;
    for (int d = 0; d <= lim; d++) {
        int ilo = d - n; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > n) ihi = n;
        int cnt = ihi - ilo + 1;

        const char *arow = A + (ilo - 1);
        const char *brow = BR + (n - d) + ilo;
        const int16_t *pp = p0 + (ilo - 1);
        const int16_t *pl = p1 + (ilo - 1);
        const int16_t *pu = p1 + ilo;
        int16_t *cu = p2 + ilo;

        for (int k = 0; k < cnt; k += NW_W16) {
#if NW_AVX512
            __mmask32 m = _mm256_cmpeq_epi8_mask(
                _mm256_loadu_si256((const __m256i *)(arow + k)),
                _mm256_loadu_si256((const __m256i *)(brow + k)));
            __m512i sc = _mm512_mask_blend_epi16(m, vm1, vp1);
            __m512i dv = _mm512_add_epi16(
                _mm512_loadu_si512((const void *)(pp + k)), sc);
            __m512i gv = _mm512_add_epi16(
                _mm512_max_epi16(_mm512_loadu_si512((const void *)(pl + k)),
                                 _mm512_loadu_si512((const void *)(pu + k))),
                vgap);
            _mm512_storeu_si512((void *)(cu + k), _mm512_max_epi16(dv, gv));
#else
            __m128i ma = _mm_cmpeq_epi8(
                _mm_loadu_si128((const __m128i *)(arow + k)),
                _mm_loadu_si128((const __m128i *)(brow + k)));
            __m256i m16 = _mm256_cvtepi8_epi16(ma);
            __m256i sc  = _mm256_blendv_epi8(vm1, vp1, m16);
            __m256i dv  = _mm256_add_epi16(
                _mm256_loadu_si256((const __m256i *)(pp + k)), sc);
            __m256i gv  = _mm256_add_epi16(
                _mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(pl + k)),
                                 _mm256_loadu_si256((const __m256i *)(pu + k))),
                vgap);
            _mm256_storeu_si256((__m256i *)(cu + k), _mm256_max_epi16(dv, gv));
#endif
        }
        if (d <= n) {                      /* boundary cells, written last  */
            p2[0] = (int16_t)(-2 * d);     /* dp[0][d]                      */
            p2[d] = (int16_t)(-2 * d);     /* dp[d][0]  (overwrites spill)  */
        }
        int16_t *t = p0; p0 = p1; p1 = p2; p2 = t;
    }
    return (int)p1[n];
}

static int nw_wave32(int n, const char *A, const char *BR,
                     int32_t *mem, int stride)
{
    int32_t *p0 = mem, *p1 = mem + stride, *p2 = mem + 2 * stride;
#if NW_AVX512
    const __m512i vp1  = _mm512_set1_epi32(1);
    const __m512i vm1  = _mm512_set1_epi32(-1);
    const __m512i vgap = _mm512_set1_epi32(-2);
#else
    const __m256i vp1  = _mm256_set1_epi32(1);
    const __m256i vm1  = _mm256_set1_epi32(-1);
    const __m256i vgap = _mm256_set1_epi32(-2);
#endif
    const int lim = 2 * n;
    for (int d = 0; d <= lim; d++) {
        int ilo = d - n; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > n) ihi = n;
        int cnt = ihi - ilo + 1;

        const char *arow = A + (ilo - 1);
        const char *brow = BR + (n - d) + ilo;
        const int32_t *pp = p0 + (ilo - 1);
        const int32_t *pl = p1 + (ilo - 1);
        const int32_t *pu = p1 + ilo;
        int32_t *cu = p2 + ilo;

        for (int k = 0; k < cnt; k += NW_W32) {
#if NW_AVX512
            __mmask16 m = _mm_cmpeq_epi8_mask(
                _mm_loadu_si128((const __m128i *)(arow + k)),
                _mm_loadu_si128((const __m128i *)(brow + k)));
            __m512i sc = _mm512_mask_blend_epi32(m, vm1, vp1);
            __m512i dv = _mm512_add_epi32(
                _mm512_loadu_si512((const void *)(pp + k)), sc);
            __m512i gv = _mm512_add_epi32(
                _mm512_max_epi32(_mm512_loadu_si512((const void *)(pl + k)),
                                 _mm512_loadu_si512((const void *)(pu + k))),
                vgap);
            _mm512_storeu_si512((void *)(cu + k), _mm512_max_epi32(dv, gv));
#else
            __m128i ma = _mm_cmpeq_epi8(
                _mm_loadl_epi64((const __m128i *)(arow + k)),
                _mm_loadl_epi64((const __m128i *)(brow + k)));
            __m256i m32 = _mm256_cvtepi8_epi32(ma);
            __m256i sc  = _mm256_blendv_epi8(vm1, vp1, m32);
            __m256i dv  = _mm256_add_epi32(
                _mm256_loadu_si256((const __m256i *)(pp + k)), sc);
            __m256i gv  = _mm256_add_epi32(
                _mm256_max_epi32(_mm256_loadu_si256((const __m256i *)(pl + k)),
                                 _mm256_loadu_si256((const __m256i *)(pu + k))),
                vgap);
            _mm256_storeu_si256((__m256i *)(cu + k), _mm256_max_epi32(dv, gv));
#endif
        }
        if (d <= n) {
            p2[0] = -2 * d;
            p2[d] = -2 * d;
        }
        int32_t *t = p0; p0 = p1; p1 = p2; p2 = t;
    }
    return (int)p1[n];
}
#endif /* __AVX2__ */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if defined(__AVX2__)
    if (n >= 64) {
        const int PAD = 128;
        int stride = (n + 1 + 128 + 63) & ~63;
        int use16 = (n <= 16000);
        char *sa = (char *)malloc((size_t)n + PAD);
        char *sb = (char *)malloc((size_t)n + PAD);
        void *mem = malloc((size_t)stride * 3 *
                           (use16 ? sizeof(int16_t) : sizeof(int32_t)));
        if (sa && sb && mem) {
            memcpy(sa, a, (size_t)n);
            memset(sa + n, 'N', PAD);
            for (int k = 0; k < n; k++) sb[k] = b[n - 1 - k];
            memset(sb + n, 'M', PAD);
            int r = use16 ? nw_wave16(n, sa, sb, (int16_t *)mem, stride)
                          : nw_wave32(n, sa, sb, (int32_t *)mem, stride);
            free(sa); free(sb); free(mem);
            return r;
        }
        free(sa); free(sb); free(mem);
    }
#endif
    return nw_scalar(n, a, b);
}
```

## PREDICTION

PREDICTION: 10

(Interpretation: ~10× faster than the reference `O(n²)`-table Needleman–Wunsch, identical return value. Reasoning behind the number: the reference's inner loop carries a serial `left → add → max` dependency of ~2–3 cycles/cell and streams an `O(n²)` int table through memory; the wavefront kernel issues ~13 µops per 16 cells (AVX2) or per 32 cells (AVX-512), load-port-bound at roughly 0.15–0.25 cycles/cell, with an `O(n)` working set. That is ~10–16×; I deliberately predict at the low end because short anti-diagonals near the two corners waste lanes — average diagonal length is `n/2`, so lane utilisation is well under 100% for small `n`.)

## MEASUREMENT

**Not measured — reported plainly as a failure, not glossed over.** This session was invoked with no executable tools: `alignment_bench` / `alignment_contract` and the symbolic and unconventional helpers were listed as nominally available but the environment exposes no tool-call surface for them (only unrelated Gmail/Drive/Slack/Docs connectors, two of which are additionally unauthorized). I could not compile, could not run the contract check, and could not time anything. The four permitted improvement rounds were therefore never opened — every design choice above is static reasoning, not an observation.

What I *did* verify by hand, in lieu of measurement:

- **Recurrence identity.** Substituting `cur[i]=dp[i][d-i]` into the reference recurrence gives exactly `max(pp[i-1]+s, max(p[i-1],p[i])-2)`; no operation is reordered in a way that changes an integer max/add result, so the return value is bit-identical, not approximate.
- **Traced `n=2`, `a="AA"`, `b="AA"` cell by cell** through diagonals `d=0..4`, including the reversed-`b` index `br[(n-d)+i]` (which lands on `b[1]` and `b[0]` correctly) — final `p1[n] = 2`, matching NW.
- **Spill safety proof.** The last vector store of a diagonal writes up to `W-1` cells past `ihi`. Diagonal `d+1` reads indices `≤ min(n,d)` and diagonal `d+2` reads `≤ min(n-1,d)`; both are `≤ d`, and index `d` is rewritten with the true boundary `-2d` *after* the spill. So garbage is provably never consumed. Buffer bound: max touched index `n+31 < stride = n+129`.
- **`int16` range.** Scores lie in `[-2n, n]`; the deepest intermediate is `-2n-2 = -32002 > -32768` at the `n ≤ 16000` cutoff, so the narrow path cannot overflow; wider `n` falls to the `int32` path.
- **Read bounds.** `A` is touched at `[0, n+30]`, `BR` at `[0, n+30]`, both inside the `PAD=128` copies — no read of the caller's buffers past `n`, no page-crossing overread of `a`/`b`.

## VERDICT

The mechanism translates cleanly and faithfully: the "slanted line of helpers" is an anti-diagonal SIMD wavefront, the helpers are vector lanes, and the reversed-`b` trick is what makes every operand of the recurrence a unit-stride load so the lanes actually pay off. Nothing about the DP is weakened — no band, no score-range trick, no early exit — so correctness is structural rather than probabilistic.

But I have to be blunt about the state of the deliverable: **it is unverified.** I predicted ~10× and then could not measure, which means the honest status is "plausible and hand-checked, unproven." The specific things I would want a run to tell me, in order: (1) does it compile and pass `alignment_contract` on all three code paths, especially the AVX-512 one, which is the least-exercised branch and the most likely place for a real bug; (2) the actual speedup at the bench's real `n`, since lane utilisation on short corner diagonals is the main gap between my 10× and the ~16× the µop count suggests; (3) whether the AVX-512 path downclocks enough to lose to the AVX2 `int16` path, in which case the right fix is to cap at 256-bit, not to abandon the wavefront. If any of those come back bad, the correction stays inside this mechanism — tile the wavefront or narrow the vectors — rather than switching approach.