## APPROACH

**Mapping the disguised solution onto the real problem.**

| Parade story | DNA aligner |
|---|---|
| Two equal-length lines, one flag each | two length-`n` strings over {A,C,G,T} |
| point for a match, penalty for a differing flag | `+1` / `-1` substitution score |
| a "slip" (shoelace) costing more | gap = `-2` |
| the giant chart, every A-child × every B-child | the full `(n+1)²` Needleman–Wunsch table |
| each box needs above, left, above-left | the exact NW recurrence, unchanged |
| **"only draw a narrow strip along the natural front-to-back matching"** | **band `|i−j| ≤ d` around the main diagonal** |
| "no reasonable amount of slipping reaches the far corners" | corners require many gaps; gaps are expensive, so they can never be optimal |
| "same box-by-box way, read the answer from the end of the strip" | identical recurrence inside the band, answer at `dp[n][n]` |
| "as long as no child drifts further than the strip allows" | this is the one thing the story hand-waves — I make it a **proved** bound, so the result is bit-identical to full NW, not an approximation |

**Making the band provably exact (the one place I refuse to hand-wave).**
With `g` gap columns, aligned pairs `p = n − g/2`, and `X` mismatches, the score is
`S = n − 2X − 2.5g ≤ n − 2.5g`. A path that ever reaches deviation `v = |i−j|` must spend `v` insertions *and* `v` deletions (it starts and ends on the diagonal), so `g ≥ 2v` and `S ≤ n − 5v`.
Let `D = Σ (a[i]==b[i] ? 1 : −1)` — the score of the pure-diagonal alignment, a *valid* alignment, so `S* ≥ D`. Choose `d = floor((n−D)/5)`. Then `5(d+1) ≥ (n−D)+1`, so every path with deviation `≥ d+1` scores `≤ D−1 < S*`. **A half-width-`d` band therefore contains an optimal path exactly.** No doubling, no retry, one pass. (`d` is clamped to `[16, n]`; a wider band is always still exact.)

**Making the strip fast.** The left-neighbour dependency forbids row-wise SIMD, so I sweep **anti-diagonals** `k = i+j`: cell `(i,j)` needs `k−1` at `i−1`/`i` and `k−2` at `i−1`, which are pure unaligned neighbour loads — all lanes of one anti-diagonal are independent. Three rotating score rows of `O(n)` replace the `O(n²)` table (the reference's real hidden cost). `b` is pre-reversed once so the substitution lookup `b[k−i−1] = brev[n−k+i]` becomes a *contiguous ascending* byte load; `pcmpeqb` → sign-extend → `s = (m & 2) − 1` gives ±1 branchlessly. Lanes are 16-bit (32/vector on AVX-512BW, 16 on AVX2) whenever `n + 2d < 31000` guarantees no overflow, else 32-bit; scalar fallback otherwise. Out-of-band cells are enforced with one `−INF` sentinel written at each end of every anti-diagonal (I verified the read window never exceeds `hi+1`, so vector over-writes past the band are never read back).

Work: `≈ 2n(d+1)` cells instead of `n²` — for unrelated random DNA `d ≈ 0.3n` (≈1.7× fewer cells), for similar sequences `d` collapses to the 16-cell floor (asymptotically linear). The rest of the win is SIMD + cache.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__) || defined(__AVX512BW__)
#include <immintrin.h>
#endif

#define NW_PAD 64

/* ---------- 32-bit lane banded anti-diagonal NW (always available) ---------- */
static int nw_band_i32(int n, const char *A, const char *BR, int d)
{
    const int W = n + 160;
    int *buf = (int *)malloc((size_t)3 * (size_t)W * sizeof(int));
    if (!buf) return 0;
    const int NEG = -(1 << 28);
    { size_t tot = (size_t)3 * (size_t)W, t;
      for (t = 0; t < tot; t++) buf[t] = NEG; }
    int *p2 = buf + 16;          /* diagonal k-2 */
    int *p1 = buf + W + 16;      /* diagonal k-1 */
    int *cu = buf + 2 * W + 16;  /* diagonal k   */
#if defined(__AVX2__)
    const __m256i vone = _mm256_set1_epi32(1);
    const __m256i vtwo = _mm256_set1_epi32(2);
#endif
    const int K = 2 * n;
    for (int k = 0; k <= K; k++) {
        int t  = k - d;
        int lo = (t <= 0) ? 0 : ((t + 1) >> 1);
        if (k - n > lo) lo = k - n;
        int hi = (k + d) >> 1;
        if (hi > k) hi = k;
        if (hi > n) hi = n;
        const int off = n - k;
#if defined(__AVX2__)
        for (int i = lo; i <= hi; i += 8) {
            __m128i ca = _mm_loadl_epi64((const __m128i *)(A + i - 1));
            __m128i cb = _mm_loadl_epi64((const __m128i *)(BR + off + i));
            __m256i m  = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
            __m256i s  = _mm256_sub_epi32(_mm256_and_si256(m, vtwo), vone);
            __m256i vd = _mm256_add_epi32(
                             _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), s);
            __m256i vu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i vl = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i vh = _mm256_sub_epi32(_mm256_max_epi32(vu, vl), vtwo);
            _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi32(vd, vh));
        }
#else
        for (int i = lo; i <= hi; i++) {
            int s = (A[i - 1] == BR[off + i]) ? 1 : -1;
            int v = p2[i - 1] + s;
            int u = p1[i - 1], l = p1[i];
            int h = ((u > l) ? u : l) - 2;
            if (h > v) v = h;
            cu[i] = v;
        }
#endif
        if (k <= d) { cu[0] = -2 * k; cu[k] = -2 * k; }  /* first row / first col */
        cu[lo - 1] = NEG;
        cu[hi + 1] = NEG;
        { int *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }
    }
    int res = p1[n];
    free(buf);
    return res;
}

/* ---------- 16-bit lane banded anti-diagonal NW (AVX2 / AVX-512BW) ---------- */
#if defined(__AVX2__)
static int nw_band_i16(int n, const char *A, const char *BR, int d)
{
    const int W = n + 160;
    short *buf = (short *)malloc((size_t)3 * (size_t)W * sizeof(short));
    if (!buf) return nw_band_i32(n, A, BR, d);
    const short NEG = -32000;
    { size_t tot = (size_t)3 * (size_t)W, t;
      for (t = 0; t < tot; t++) buf[t] = NEG; }
    short *p2 = buf + 16;
    short *p1 = buf + W + 16;
    short *cu = buf + 2 * W + 16;
#if defined(__AVX512BW__)
    const __m512i vone = _mm512_set1_epi16(1);
    const __m512i vtwo = _mm512_set1_epi16(2);
#else
    const __m256i vone = _mm256_set1_epi16(1);
    const __m256i vtwo = _mm256_set1_epi16(2);
#endif
    const int K = 2 * n;
    for (int k = 0; k <= K; k++) {
        int t  = k - d;
        int lo = (t <= 0) ? 0 : ((t + 1) >> 1);
        if (k - n > lo) lo = k - n;
        int hi = (k + d) >> 1;
        if (hi > k) hi = k;
        if (hi > n) hi = n;
        const int off = n - k;
#if defined(__AVX512BW__)
        for (int i = lo; i <= hi; i += 32) {
            __m256i ca = _mm256_loadu_si256((const __m256i *)(A + i - 1));
            __m256i cb = _mm256_loadu_si256((const __m256i *)(BR + off + i));
            __m512i m  = _mm512_cvtepi8_epi16(_mm256_cmpeq_epi8(ca, cb));
            __m512i s  = _mm512_sub_epi16(_mm512_and_si512(m, vtwo), vone);
            __m512i vd = _mm512_adds_epi16(
                             _mm512_loadu_si512((const void *)(p2 + i - 1)), s);
            __m512i vu = _mm512_loadu_si512((const void *)(p1 + i - 1));
            __m512i vl = _mm512_loadu_si512((const void *)(p1 + i));
            __m512i vh = _mm512_subs_epi16(_mm512_max_epi16(vu, vl), vtwo);
            _mm512_storeu_si512((void *)(cu + i), _mm512_max_epi16(vd, vh));
        }
#else
        for (int i = lo; i <= hi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(A + i - 1));
            __m128i cb = _mm_loadu_si128((const __m128i *)(BR + off + i));
            __m256i m  = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i s  = _mm256_sub_epi16(_mm256_and_si256(m, vtwo), vone);
            __m256i vd = _mm256_adds_epi16(
                             _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), s);
            __m256i vu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i vl = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i vh = _mm256_subs_epi16(_mm256_max_epi16(vu, vl), vtwo);
            _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi16(vd, vh));
        }
#endif
        if (k <= d) { cu[0] = (short)(-2 * k); cu[k] = (short)(-2 * k); }
        cu[lo - 1] = NEG;
        cu[hi + 1] = NEG;
        { short *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }
    }
    int res = (int)p1[n];
    free(buf);
    return res;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    /* Lower bound on the optimal score: the pure-diagonal alignment. */
    int D = 0;
    for (int i = 0; i < n; i++) D += (a[i] == b[i]) ? 1 : -1;

    /* Any path of deviation v costs >= 2v gaps => score <= n - 5v.
       d = floor((n-D)/5) => every path leaving the band scores < D <= S*,
       so the banded optimum equals the full Needleman-Wunsch optimum. */
    int d = (n - D) / 5;
    if (d < 16) d = 16;
    if (d > n)  d = n;

    size_t rg = (size_t)n + 2 * NW_PAD;
    char *sb = (char *)malloc(2 * rg);
    if (!sb) return 0;
    memset(sb, 'N', 2 * rg);
    char *A  = sb + NW_PAD;
    char *BR = sb + rg + NW_PAD;
    memcpy(A, a, (size_t)n);
    for (int i = 0; i < n; i++) BR[i] = b[n - 1 - i];   /* reversed b */

    int res;
#if defined(__AVX2__)
    if (n + 2 * d < 31000) res = nw_band_i16(n, A, BR, d);
    else                   res = nw_band_i32(n, A, BR, d);
#else
    res = nw_band_i32(n, A, BR, d);
#endif
    free(sb);
    return res;
}
```

## PREDICTION

PREDICTION: 12

(Speedup factor over the reference `kernel` on unrelated random DNA: ~1.7× from the band, ~8–15× from 16-bit anti-diagonal SIMD and the `O(n)` working set; much larger — trending to linear-time — for similar/mutated sequence pairs, where `d` collapses to the 16 floor.)

## MEASUREMENT

**Not measured — and I will not dress that up.** This session was invoked with no tools available (`alignment_bench` / `alignment_contract` were named in the task but are not callable here), so the number above is an unverified analytic estimate, not a result. Nothing below is evidence of speed; it is only the correctness work I could do by hand:

- **Band exactness** proved above from `S ≤ n − 2.5g` and `g ≥ 2v`, with the off-by-one checked: `d = floor(x/5) ⇒ 5(d+1) ≥ x+1 > x`, giving strict inequality, so no tie can escape the band.
- **Index algebra** hand-verified: `b[j−1] = brev[n−k+i]`; the read window of any anti-diagonal never exceeds `hi+1` or falls below `lo−1` (since `lo` is non-decreasing and `hi` grows by ≤1 per diagonal), which is exactly why one sentinel per side suffices and why vector over-stores past `hi+1` are provably never read back.
- **Non-empty windows** for all `k ∈ [0,2n]` given the enforced `d ≥ 1` (the `d = 0` parity hole that would empty odd anti-diagonals is excluded by the floor of 16).
- **Overflow**: 16-bit path gated on `n + 2d < 31000`, and true cell values are bounded below by `−(n + 2d)`; saturating adds keep out-of-band garbage from wrapping.
- **Hand-traced** `n=1` (`"A"/"A" → 1`) and `n=2` (`"AC"/"CA" → −2`) through the anti-diagonal loop including the boundary overrides.

Untested risks I would check first with a real bench: the AVX-512BW path (compiles only under `-march=native` on a 512-capable host, and could down-clock on Skylake-X-class parts — if so, deleting that `#if` branch falls back to the AVX2 16-lane path), and small-`n` behaviour where the 2n-long inter-diagonal dependency chain, not cell count, sets the runtime.

## VERDICT

The mechanism translates cleanly and, importantly, translates *better* than the story tells it: the parade version's caveat "as long as no child drifts further than the strip allows" is an unproven assumption, whereas here the gap cost lets me compute a band half-width `d = floor((n−D)/5)` that is **guaranteed** to contain an optimal path, so the kernel is bit-exact against the reference rather than a heuristic. The band alone is a modest win on unrelated sequences (~1.7×, since expensive gaps only bound deviation at ~0.3n); the bulk of the expected speedup comes from implementing that same strip well — anti-diagonal SIMD over 16-bit lanes with an `O(n)` working set instead of an `O(n²)` table. I predict ~12× and cannot confirm it: **no measurement was possible in this session**, so treat the artifact as reasoned-correct and unbenchmarked.