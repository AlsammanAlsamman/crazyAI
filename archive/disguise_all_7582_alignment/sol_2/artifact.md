## APPROACH

**Mapping the disguised solution onto the real problem**

| Camp-trail story | DNA implementation |
|---|---|
| Giant scoreboard, Mira down the side / Theo across the top | the full `(n+1)×(n+1)` Needleman–Wunsch table |
| "Never write a box before the one above, left, and above-left" | the NW recurrence `max(diag+s, up−2, left−2)` |
| "Both trails walked in the order laid down, never backwards" | both strings scanned forward; no reverse/meet-in-middle trick |
| **Narrow diagonal ribbon corner-to-corner** | band `|i−j| ≤ d`, i.e. only cells whose row and column index are close |
| **"Modest cushion for a reasonable number of hops"** | `d` derived from how many gaps the score budget can afford |
| **"Widen the ribbon if the trails are unusually strange"** | provable widen-and-recheck loop (at most two passes in practice) |
| "A single judge working one box at a time takes forever" | replaced by a **wavefront**: all cells on one anti-diagonal are mutually independent (their three parents live on the two previous anti-diagonals), so 16–32 cells are filled per instruction with SIMD — same three-neighbour rule, same order of dependence, just many boxes at once |

**Why the ribbon is provably wide enough** (this is the part the story hand-waves as "almost every time"; I make it exact). Both strings have length `n`, so any alignment has equal numbers of gaps in each (`g` each), `n−g` substitution columns, hence score `≤ (n−g) − 4g = n − 5g`. Reaching deviation `|i−j| = d` forces `g ≥ d`, so **every path that leaves the band scores `≤ n − 5(d+1)`**. So: run the banded DP, get `S`; if `S ≥ n − 5d`, `S` is provably the *global* optimum and we stop. Starting `d` comes from the gap-free (main-diagonal) alignment score `LB = 2·matches − n`: `d₀ = ⌈(n−LB)/5⌉ ≤ 0.4n`. I first run a cheap `d = 24` probe (this alone finishes for identical or merely *shifted* sequences — band collapses to a handful of cells), then one final pass with the certified width. At most two real passes; the loop is monotone so it cannot spin, and a hard fallback to `d = n` (full DP) is wired in.

**Implementation details that matter**
- Anti-diagonal `k`: cell `(i, k−i)` needs `prev2[i−1]`, `prev1[i−1]`, `prev1[i]` — all contiguous ⇒ pure vector loads, no shuffles, no gather.
- `b` is stored **reversed** once, so the character to compare with `a[i−1]` is `br[n−k+i]`: both operands advance with `i`, so one 16/32-byte load each + `cmpeq` gives all the ±1 substitution scores.
- `int16` lanes (16 with AVX2, 32 with AVX-512BW/VL), guarded by a range check `n + 2d ≤ 31000`; in-band values are provably `≥ −(n+2d)`, so a `−32000` sentinel is strictly below every real value. Outside that range a portable banded row DP runs instead.
- Band edges use two sentinel guard cells rewritten every anti-diagonal; the diagonal parent of an in-band cell is always in-band, so sentinels can never drift or leak into a real value.
- Memory is `O(n)` (three short rows) instead of the reference's `4(n+1)²` bytes — the reference is memory-bound for `n ≳ 2000`, we stay in L1.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define GPEN (-2)

/* ---- portable banded DP over rows (fallback: no AVX2, or huge n) ---- */
static int band_rows(int n, const char *a, const char *b, int d,
                     int *prev, int *cur)
{
    const int NEG = -(1 << 28);
    int i, j, lim;
    if (d > n) d = n;
    if (d < 1) d = 1;
    for (j = 0; j <= n + 1; j++) prev[j] = NEG;
    lim = (d < n) ? d : n;
    for (j = 0; j <= lim; j++) prev[j] = -2 * j;
    for (i = 1; i <= n; i++) {
        int lo = i - d, hi = i + d, start, left;
        const char ai = a[i - 1];
        if (lo < 0) lo = 0;
        if (hi > n) hi = n;
        if (lo > 0) cur[lo - 1] = NEG;
        if (hi < n) cur[hi + 1] = NEG;
        if (lo == 0) { cur[0] = -2 * i; start = 1; } else start = lo;
        left = cur[start - 1];
        for (j = start; j <= hi; j++) {
            int best = prev[j - 1] + ((ai == b[j - 1]) ? 1 : -1);
            int t = prev[j] + GPEN; if (t > best) best = t;
            t = left + GPEN;        if (t > best) best = t;
            cur[j] = best;
            left = best;
        }
        { int *tmp = prev; prev = cur; cur = tmp; }
    }
    return prev[n];
}

#if defined(__AVX2__)
/* ---- banded anti-diagonal wavefront, 16-bit lanes ----
   ap : padded copy of a ; br : padded reverse of b
   A,B,C : three rows, valid indices [-32, n+63]                        */
static int band_diag16(int n, const char *ap, const char *br, int d,
                       short *A, short *B, short *C)
{
    const short NEG = -32000;
    short *p2 = A, *p1 = B, *cu = C;
    int k, t;
#if defined(__AVX512BW__) && defined(__AVX512VL__)
    const __m512i vgap = _mm512_set1_epi16(GPEN);
    const __m512i vpos = _mm512_set1_epi16(1);
    const __m512i vneg = _mm512_set1_epi16(-1);
    const int STEP = 32;
#else
    const __m256i vgap = _mm256_set1_epi16(GPEN);
    const __m256i vneg = _mm256_set1_epi16(-1);
    const int STEP = 16;
#endif
    for (t = -32; t <= n + 63; t++) { A[t] = NEG; B[t] = NEG; C[t] = NEG; }
    p1[0] = 0;                              /* anti-diagonal 0: cell (0,0) */

    for (k = 1; k <= 2 * n; k++) {
        int lo, hi, i0, i1, i, sh;
        lo = (k <= d) ? 0 : ((k - d + 1) >> 1);   /* ceil((k-d)/2), k-d>0 */
        if (k - n > lo) lo = k - n;
        hi = (k + d) >> 1;
        if (hi > n) hi = n;
        if (hi > k) hi = k;
        i0 = (lo > 1) ? lo : 1;                   /* interior cells only   */
        i1 = (hi < k - 1) ? hi : k - 1;
        sh = n - k;
        for (i = i0; i <= i1; i += STEP) {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            __m256i ca = _mm256_loadu_si256((const __m256i *)(ap + i - 1));
            __m256i cb = _mm256_loadu_si256((const __m256i *)(br + sh + i));
            __mmask32 eq = _mm256_cmpeq_epi8_mask(ca, cb);
            __m512i sc = _mm512_mask_blend_epi16(eq, vneg, vpos);
            __m512i dg = _mm512_loadu_si512((const void *)(p2 + i - 1));
            __m512i up = _mm512_loadu_si512((const void *)(p1 + i - 1));
            __m512i lf = _mm512_loadu_si512((const void *)(p1 + i));
            __m512i bs = _mm512_max_epi16(_mm512_add_epi16(dg, sc),
                           _mm512_add_epi16(_mm512_max_epi16(up, lf), vgap));
            _mm512_storeu_si512((void *)(cu + i), bs);
#else
            __m128i ca = _mm_loadu_si128((const __m128i *)(ap + i - 1));
            __m128i cb = _mm_loadu_si128((const __m128i *)(br + sh + i));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i sc = _mm256_sub_epi16(vneg, _mm256_add_epi16(eq, eq));
            __m256i dg = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
            __m256i up = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i bs = _mm256_max_epi16(_mm256_add_epi16(dg, sc),
                           _mm256_add_epi16(_mm256_max_epi16(up, lf), vgap));
            _mm256_storeu_si256((__m256i *)(cu + i), bs);
#endif
        }
        if (k <= d) {                        /* cells (0,k) and (k,0)      */
            short v = (short)(-2 * k);
            cu[0] = v;
            cu[k] = v;
        }
        cu[lo - 1] = NEG;                    /* band guards, written last  */
        cu[hi + 1] = NEG;
        { short *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }
    }
    return (int)p1[n];
}
#endif

int kernel(int n, const char *a, const char *b)
{
    int i, m = 0, lb, dcap, d, S = 0, it;
    size_t cn, sstride;
    char *cbuf;
    short *sbuf = NULL;
    int *ibuf = NULL;

    if (n <= 0) return 0;

    for (i = 0; i < n; i++) m += (a[i] == b[i]);
    lb = 2 * m - n;                 /* score of the gap-free alignment      */
    dcap = (n - lb + 4) / 5;        /* no optimal path can deviate further  */
    if (dcap > n) dcap = n;
    if (dcap < 1) dcap = 1;
    d = (dcap < 24) ? dcap : 24;    /* cheap probe first                    */

    cn = (size_t)n + 128;
    sstride = (size_t)n + 96;
    cbuf = (char *)malloc(2 * cn);
    if (cbuf) {
        memcpy(cbuf, a, (size_t)n);
        memset(cbuf + n, 0, 128);
        { char *br = cbuf + cn;
          for (i = 0; i < n; i++) br[i] = b[n - 1 - i];
          memset(br + n, 1, 128); }
    }

    for (it = 0; ; it++) {
        int did_v = 0;
#if defined(__AVX2__)
        if (cbuf && (n + 2 * d) <= 31000) {
            if (!sbuf) sbuf = (short *)malloc(3 * sstride * sizeof(short));
            if (sbuf) {
                S = band_diag16(n, cbuf, cbuf + cn, d,
                                sbuf + 32, sbuf + sstride + 32,
                                sbuf + 2 * sstride + 32);
                did_v = 1;
            }
        }
#endif
        if (!did_v) {
            if (!ibuf) ibuf = (int *)malloc(2 * ((size_t)n + 8) * sizeof(int));
            if (!ibuf) { free(cbuf); free(sbuf); return lb; }
            S = band_rows(n, a, b, d, ibuf, ibuf + n + 8);
        }
        /* certificate: any path leaving the band scores <= n-5(d+1) */
        { int need = (n - S + 4) / 5;
          if (need <= d || d >= n) break;
          d = (need > n) ? n : need;
          if (it >= 4) d = n; }
    }

    free(cbuf);
    free(sbuf);
    free(ibuf);
    return S;
}
```

## PREDICTION: 12

(Speedup factor over the reference `O(n²)` scalar aligner, at `n` in the low thousands on random DNA; reasoning: certified band ≈ 0.6·n² cells × 16 int16 lanes, minus wavefront bookkeeping, plus a cache-residency bonus the reference loses to its `4(n+1)²`-byte table. With AVX-512BW present I'd expect ~20×; with only SSE fallback, ~1.5×.)

## MEASUREMENT

**Not performed — and I have to report that plainly rather than dress it up.** This session was launched with no tool access (`alignment_bench` / `alignment_contract` were listed in the prompt but are not callable here, and the environment note confirms it), so I could neither compile nor time anything, and consequently used none of my four allotted improvement rounds. Everything above is static reasoning, hand-traced verification, and an op-count estimate:

- Correctness was checked by hand-simulating the wavefront on `n=2` (`AC`/`AC`, all five anti-diagonals, every cell matching the reference table), plus `n=1`, the shifted case `ACGTACGT`/`TACGTACG` (band 4, answer 3), and the near-identical cases where the band collapses to `d=1`.
- The exactness argument does not rest on "almost every time": the widen-and-recheck certificate `S ≥ n − 5d` makes any returned value provably equal to the full DP's value, and `d = n` is a wired-in terminal fallback.
- Residual risks I could not retire without a compiler: the AVX-512 branch (`_mm256_cmpeq_epi8_mask` / `_mm512_mask_blend_epi16`) has never been compiled by me; if the benchmark host is AVX-512-capable and I mistyped an intrinsic, this fails to build rather than merely running slow. If that happens, deleting the `#if defined(__AVX512BW__) …` block and its `#else` guard leaves the tested-by-inspection AVX2 path.

## VERDICT

A faithful translation of the mechanism: the same three-neighbour, forward-only, box-by-box fill, restricted to a corner-to-corner ribbon, with the story's "widen it if the trails are strange" upgraded from a hope to a proof, and the "single judge" replaced by 16–32 boxes filled per instruction along anti-diagonals (which is legal precisely because the three parents of every cell on an anti-diagonal lie on the two preceding ones). I predict ~12× and claim exact agreement with the reference for all inputs; **but the honest bottom line is that this artifact is unmeasured** — the prediction stands as a prediction, and the pipeline's numbers, not mine, decide whether it was right.