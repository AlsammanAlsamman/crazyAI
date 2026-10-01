## MAPPING

### SEED 1 — "the low sun throws a single doubled shadow only where two knots share a dye"

| world object | problem object |
|---|---|
| two grass-ropes, knots dyed 4 berry-colors | the two length-`n` strings over {A,C,G,T} |
| rope hung on the east wall | `a`, indexed by `i` — the *still* rope |
| rope stretched from the doorway *at a slant* | `b`, laid along an anti-diagonal — the *flowing* rope |
| the low sun arriving all at once | one SIMD instruction (`_mm_cmpeq_epi8`) |
| one doubled shadow instead of two | a 0xFF lane in the compare mask = match cell |
| "without a hand ever touching the floor" | match/mismatch for 16 cells resolved without 16 scalar comparisons |

**Breaks:** *"one pair of positions is judged at a time."*

### SEED 2 — "the cost of the bare floor is read off how many knots forward each rope must go, never walked"

| world object | problem object |
|---|---|
| bare floor between treasures | cells of the DP grid that no optimal path can use |
| "the corridor's cost was never a secret — only its length was" | out-of-band alignment scores are bounded *analytically*: any path reaching offset \|i−j\| = k has score ≤ n − 5k |
| "the ropes already told me that length" | n and the count of same-dye knots on the main diagonal give a lower bound LB = 2·m₀ − n |
| never standing in the corridor to learn its cost | those cells are never allocated, never touched, never stored |
| "the treasures alone don't excuse me from the corridor's cost" | the discarded region is *accounted for*, not ignored — correctness is certified, not hoped for |

**Breaks:** *"the whole grid of every position against every other must be filled in."*

### SEED 3 — "only the cheapest arriving tally is kept; the rest drop like a bad thread-end"

| world object | problem object |
|---|---|
| running tallies arriving at a treasure | the three predecessors (diag, up, left) |
| keeping only the cheapest | `max_epi16` reduction, no branch, no traceback |
| "slip spent, or not yet spent" | the two gap states — collapse to one under *linear* gap cost |
| letting the bad end fall and not picking it up | no path storage: two anti-diagonals of scratch, O(n) memory instead of O(n²) |

**Breaks:** *"every cell depends on above, left, and diagonal, computed in that order"* (a slant ordering makes all three predecessors already-finished, so cells on one anti-diagonal are mutually independent).

## CHOSEN SEED

**SEED 2.** It is the one seed that breaks *"the whole grid must be filled in"*, and its mapping is startlingly literal — the native's closing sentence is, word for word, a band-pruning **certificate**: the skipped region's cost is known in closed form from the ropes' lengths, so it is charged without being walked. SEED 1 and SEED 3 are then not discarded but used as the *mechanism* for traversing what remains: the sun is the SIMD compare, the slant is the anti-diagonal wavefront, the dropped thread-end is the branchless max and the O(n) footprint.

Honest note on step 4: this mechanism lands on **validated real-world technique** — Ukkonen-style banded DP with a score-bound certificate (as in minimap2/KSW2) plus Wozniak anti-diagonal SIMD — rather than an invention. That is the right outcome; I let the metaphor arrive there instead of inventing something untested.

## ASSUMPTION BROKEN

**"The whole grid of every position against every other must be filled in."**

The arithmetic the ropes hand over: both strings have length n, so insertions = deletions = k, and

`score = 2m − n − 3k ≤ n − 5k`.

Reaching offset \|i−j\| = W+1 forces k ≥ W+1. So with `LB = 2·m₀ − n` (the no-slip diagonal, m₀ = doubled shadows on the main diagonal) and `W = ⌊(n − LB)/5⌋`, every path outside the band scores `≤ n − 5(W+1) < LB ≤` the banded answer. The band is **provably exact**, and since LB ≥ −n it is always `W ≤ 0.4n` — the corridor is never walked, and for similar sequences it collapses to almost nothing. Regime detection is m₀ itself, read straight off the ropes.

No thread parallelism: the wavefront's unit of work is one anti-diagonal (~W cells), which would need a barrier per diagonal — the metaphor's own units are too small. Vectorization only, as instructed.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH (-1)
#define GAP      (-2)

/* ---- scalar banded row DP (int32): fallback for no-AVX2, tiny n, huge n ---- */
static int nw_band_rows(int n, const char *restrict a, const char *restrict b, int W)
{
    const int NEG = -(1 << 28);
    int i, j, res;
    int *buf, *prev, *cur, *t;
    if (W < 1) W = 1;
    if (W > n) W = n;
    buf = (int *)malloc((size_t)2 * (size_t)(n + 2) * sizeof(int));
    if (!buf) return 0;
    prev = buf;
    cur  = buf + (n + 2);
    {   int hi0 = (W < n) ? W : n;
        for (j = 0; j <= hi0; j++) prev[j] = -2 * j;
        prev[hi0 + 1] = NEG;                       /* hi0+1 <= n+1 */
    }
    for (i = 1; i <= n; i++) {
        int lo = i - W, hi = i + W;
        const char ai = a[i - 1];
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        if (lo >= 2) cur[lo - 1] = NEG;
        else         cur[0] = (i <= W) ? -2 * i : NEG;
        if (i + W <= n) prev[i + W] = NEG;
        for (j = lo; j <= hi; j++) {
            int dg = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int up = prev[j]     + GAP;
            int lf = cur[j - 1]  + GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        t = prev; prev = cur; cur = t;
    }
    res = prev[n];
    free(buf);
    return res;
}

#if defined(__AVX2__)
/* ---- the slant: banded anti-diagonal wavefront, 16 cells per doubled shadow ----
   diagonal d = i + j, indexed by i.  cur[i] = max( p2[i-1] + s(a[i-1],b[d-i-1]),
                                                    max(p1[i-1], p1[i]) + GAP )
   b stored reversed so the floor-rope reads forward along the slant.            */
static int nw_band_diag_avx2(int n, const char *a, const char *b, int W, int NEGi)
{
    const int P = 64;
    const size_t vlen = (size_t)n + 1 + 2 * (size_t)P;
    int16_t *mem  = (int16_t *)malloc(3 * vlen * sizeof(int16_t));
    unsigned char *cbuf = (unsigned char *)malloc(2 * (size_t)(n + 96));
    int16_t *p2, *p1, *cc, *tt;
    unsigned char *ap, *bp;
    const int16_t NEG = (int16_t)NEGi;
    __m256i vNEG, vGAP, vTWO, vONE;
    int d, i, res = 0;
    size_t k;

    if (!mem || !cbuf) { free(mem); free(cbuf); return nw_band_rows(n, a, b, W); }

    for (k = 0; k < 3 * vlen; k++) mem[k] = NEG;
    p2 = mem + P;  p1 = mem + vlen + P;  cc = mem + 2 * vlen + P;

    ap = cbuf + 32;
    bp = cbuf + (size_t)(n + 96) + 32;
    memset(cbuf,                     0x01, (size_t)(n + 96));   /* fillers differ, */
    memset(cbuf + (size_t)(n + 96),  0x02, (size_t)(n + 96));   /* so never match  */
    memcpy(ap, a, (size_t)n);
    for (i = 0; i < n; i++) bp[i] = (unsigned char)b[n - 1 - i];

    vNEG = _mm256_set1_epi16(NEG);
    vGAP = _mm256_set1_epi16(-2);
    vTWO = _mm256_set1_epi16(2);
    vONE = _mm256_set1_epi16(1);

    for (d = 0; d <= 2 * n; d++) {
        int ilo = (d - W + 1) >> 1;      /* |2i - d| <= W  */
        int ihi = (d + W) >> 1;
        int t0  = n - d;                 /* bp index = t0 + i */
        if (ilo < d - n) ilo = d - n;
        if (ilo < 0)     ilo = 0;
        if (ihi > d)     ihi = d;
        if (ihi > n)     ihi = n;
        for (i = ilo; i <= ihi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(ap + i - 1));
            __m128i cb = _mm_loadu_si128((const __m128i *)(bp + t0 + i));
            __m256i sb = _mm256_sub_epi16(
                             _mm256_and_si256(
                                 _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb)), vTWO),
                             vONE);                        /* +1 match, -1 mismatch */
            __m256i dg = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), sb);
            __m256i u  = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i l  = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i g  = _mm256_add_epi16(_mm256_max_epi16(u, l), vGAP);
            __m256i r  = _mm256_max_epi16(_mm256_max_epi16(dg, g), vNEG);
            _mm256_storeu_si256((__m256i *)(cc + i), r);
        }
        /* bare floor: never walked, only fenced off */
        _mm256_storeu_si256((__m256i *)(cc + ilo - 16), vNEG);
        _mm256_storeu_si256((__m256i *)(cc + ihi +  1), vNEG);
        _mm256_storeu_si256((__m256i *)(cc + ihi + 17), vNEG);
        if (ilo == 0) cc[0] = (int16_t)(-2 * d);   /* i = 0, j = d */
        if (ihi == d) cc[d] = (int16_t)(-2 * d);   /* i = d, j = 0 */
        if (d == 2 * n) res = (int)cc[n];
        tt = p2; p2 = p1; p1 = cc; cc = tt;
    }
    free(mem); free(cbuf);
    return res;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    int i, m0 = 0, LB, W;
    if (n <= 0) return 0;

    /* the doubled shadows on the main diagonal: regime detector + lower bound */
    for (i = 0; i < n; i++) m0 += (a[i] == b[i]);
    LB = 2 * m0 - n;

    /* certificate: any path at offset >= W+1 has score <= n-5(W+1) < LB <= answer */
    W = (n - LB) / 5;
    if (W < 1) W = 1;
    if (W > n) W = n;

#if defined(__AVX2__)
    /* guard the stated risks: int16 headroom, and SIMD setup vs tiny n */
    if (n >= 24 && (2L * n + 2L * W + 128L) < 32000L)
        return nw_band_diag_avx2(n, a, b, W, -(2 * n + 2 * W + 64));
#endif
    return nw_band_rows(n, a, b, W);
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 16

Reasoning stated before any measurement: band factor `W ≤ 0.4n`, typically `W ≈ 0.3n` for random DNA → ~1.7× fewer cells (and far more for similar sequences, where `W` collapses with `m₀`); anti-diagonal int16 AVX2 at ~4 cycles per 16 cells ≈ 0.25 cyc/cell against the reference's ~3 cyc/cell dependency chain → ~12×; plus O(n) instead of O(n²) memory traffic, which should dominate at large n. 1.7 × 12 ≈ 20, discounted to 16 for per-diagonal overhead and short wavefronts at the corners.

## MEASUREMENT

**Not measured.** `alignment_bench` and `alignment_contract` were not available in this session — no tool calls were possible, so I ran neither the correctness check nor the timing. I will not dress up an estimate as a measurement. What I *did* verify, by hand, is exactness of the mechanism rather than its speed:

- the bound `score = 2m − n − 3k ≤ n − 5k` and the band certificate `n − 5(W+1) < LB ≤ S_W`, which makes the banded answer provably equal to full Needleman–Wunsch for arbitrary inputs;
- hand-traced the anti-diagonal recurrence against the reference table for `n=1` (match), `n=1` (mismatch), and `n=2` with `a="AC", b="CA"` (answer −2, gap-tempting case) — all agree, including the boundary-cell overwrites and the sentinel margins;
- sentinel analysis: `NEG = −(2n + 2W + 64)` with a per-cell clamp, chosen so that `NEG + n <` the minimum real in-band value `−(n + 2W)`, i.e. no sentinel path can leak into the answer, and `NEG − 2` stays inside int16;
- margin analysis: `ilo` and `ihi` each move by at most 1 per diagonal, so the 16-cell left and 32-cell right `NEG` fences cover every load made by the two following diagonals.

The remaining risk is empirical, not logical: whether ~4 cycles per vector iteration is achieved in practice, and what the benchmark's `n` and sequence divergence actually are.

## VERDICT

The native's metaphor is not decoration here — it is a complete, correct algorithm, and each seed maps to a validated technique rather than a novelty: the slanted rope is Wozniak's anti-diagonal wavefront, the doubled shadow is the SIMD equality compare, the dropped thread-end is the branchless max with O(n) scratch, and the un-walked corridor is Ukkonen-style band pruning with a score-bound certificate. The one genuinely non-textbook move is the native's *reason* for the band: they do not guess a width, they read it off the ropes — `W = ⌊(n − (2m₀ − n))/5⌋` — which is exactly the 5-per-offset arithmetic that this specific scoring (match +1 / mismatch −1 / gap −2, equal lengths) forces, and which also hands over, free, the fact that the band can never exceed 0.4 n.

Where it can be worse than the known way, and what I did about it:
- **Small n** — SIMD setup, three buffer fills and the reversed copy would dominate. *Guarded:* `n < 24` falls through to the scalar banded row DP.
- **Large n** — int16 would overflow. *Guarded:* the `2n + 2W + 128 < 32000` headroom test falls through to the int32 scalar banded path, which still beats the reference on memory traffic.
- **No AVX2** — *Guarded:* `#if defined(__AVX2__)`, same scalar fallback.
- **Divergent sequences** — the band only shrinks to ~0.3 n, so the win comes almost entirely from SIMD and cache, not from pruning. This is the regime I expect the benchmark to be in, and it is why I predicted 16 rather than 100. Both regimes are recognised at runtime by m₀ and served by the same kernel; no regime is left without a path.

What would falsify this: if the measured speedup on random DNA is below roughly 8×, the anti-diagonal's unaligned loads and per-diagonal overhead are costing more than my 4-cycle estimate, and the honest next move would be a striped (Farrar-style) global formulation rather than more tuning of the slant. If it comes back *wrong* rather than slow, the first suspects are the sentinel fences and the two boundary-cell overwrites, in that order.