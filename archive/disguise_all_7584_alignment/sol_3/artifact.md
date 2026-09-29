## APPROACH

**Mapping the disguised solution onto real Needleman–Wunsch.**

The disguised solution has three moving parts: (1) *walk the diagonal straight through first, no grid*; (2) *use the tally you get to decide where/how far a "nudge" could possibly pay for itself*; (3) *only explore nudged line-ups, never the whole square*. Translated literally and made exact:

**1. The straight walk = the main diagonal of the DP table.** For two strings of *equal* length `n`, every alignment satisfies a hard accounting identity. If `k` = number of gap-pairs (a gap in `a` forces a matching gap in `b`, since both strings are fully consumed), and `MM` = mismatched aligned pairs:

```
pairs p = n - k,  gap columns = 2k
score = (p - 2MM) - 4k = n - 5k - 2MM
```

The diagonal walk is exactly `k = 0`: `S₀ = n - 2d`, `d` = Hamming distance. One O(n) auto-vectorized pass.

**2. "How far is a nudge worth testing?" = a provable band radius.** From the identity, any alignment scoring at least `L` obeys `5k ≤ n - L`, so `k ≤ ⌊(n-L)/5⌋`. And the path displacement obeys `|i-j| ≤ k` (a prefix has used at most `k` gaps of either kind). So **any lower bound `L` on the optimum gives a provably sufficient band radius `K = ⌊(n-L)/5⌋`** — outside it, no optimal path can go. The diagonal walk supplies the first `L` for free. This is the rigorous version of "a nudge only pays if the disagreements are dense enough to pay for it."

**3. "Test the nudge on the spot" = a narrow banded DP that bootstraps itself.** I first run a cheap probe band (`K = 16`, cost `32n` cells) — this is literally "nudge by one bead and re-walk." That gives a much better `L`, hence a much smaller guaranteed `K`, and if `K ≥ ⌊(n-L)/5⌋` we are already provably exact and stop. Otherwise escalate ×16 (capped at the guaranteed radius), which terminates in ≤ 3 rounds with ≤ ~7 % overhead. Every round returns a *valid* alignment score (a restriction of the feasible set), so the running `best` is always a legitimate lower bound and the final answer is provably `dp[n][n]`.

This never switches algorithm: it is the disguised mechanism (walk straight, spot the trouble, test the shift, keep the max) upgraded from a heuristic to an exact Ukkonen-style bound.

**Implementation of the band itself.** Anti-diagonal (wavefront) layout: `A[t][i] = dp[i][t-i]` depends on `A[t-2][i-1]` (diag), `A[t-1][i-1]` (up), `A[t-1][i]` (left) — **no dependency within an anti-diagonal**, so it vectorizes with plain shifts, no prefix-scan. `b` is pre-reversed so both character streams are read with stride +1. Three rolling `n+3` buffers with `-INF` guard slots at both band edges (I proved reads never reach past the guards: `lo` is non-decreasing, `hi` grows by ≤1 per step). Scores stay in `[-(n+2K), n] ⊆ [-1.8n, n]`, so **int16 lanes are provably safe for n ≤ 15000** → 16 cells per AVX2 instruction; int32 path otherwise. Working set is 3 rows, cache-resident, versus the reference's `(n+1)²` int table.

No OpenMP: the only natural parallel axis is the anti-diagonal, and 2n barriers would cost far more than the ~1 ms of work they'd split.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define NEG16V (-29000)
#define NEG32V (-(1 << 28))

/* Banded global NW over the anti-diagonals.
   A[t][i] = dp[i][t-i], stored at physical index i+1 (slot 0 / hi+2 are -INF guards).
   Band: |i - j| <= K.  Returns the best in-band alignment score (<= true optimum,
   == true optimum whenever K >= max gap-pair count of some optimal path). */
static int nw_band_i16(int n, const char *a, const char *brev, int K, short *scratch)
{
    const int stride = n + 3;
    short *d2 = scratch, *d1 = scratch + stride, *d0 = scratch + 2 * stride, *tp;
    const int T = 2 * n;
    int res = 0, z;

    for (z = 0; z < 3 * stride; z++) scratch[z] = (short)NEG16V;

    for (int t = 0;; t++) {
        int lo = (t - K + 1) >> 1;      /* ceil((t-K)/2) */
        int hi = (t + K) >> 1;          /* floor((t+K)/2) */
        if (lo < 0) lo = 0;
        if (lo < t - n) lo = t - n;
        if (hi > n) hi = n;
        if (hi > t) hi = t;

        d0[lo] = (short)NEG16V;         /* guard for logical lo-1 */
        d0[hi + 2] = (short)NEG16V;     /* guard for logical hi+1 */

        int vlo = lo, vhi = hi;
        if (lo == 0) { d0[1] = (short)(-2 * t); vlo = 1; }          /* dp[0][t]  */
        if (hi == t) { d0[t + 1] = (short)(-2 * t); vhi = t - 1; }  /* dp[t][0]  */

        int i = vlo;
#if defined(__AVX2__)
        {
            const __m256i vm1 = _mm256_set1_epi16(-1);
            const __m256i vg  = _mm256_set1_epi16(2);
            const int base = n - t;
            for (; i + 15 <= vhi; i += 16) {
                __m128i av = _mm_loadu_si128((const __m128i *)(a + i - 1));
                __m128i bv = _mm_loadu_si128((const __m128i *)(brev + base + i));
                __m256i mk = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(av, bv)); /* -1 / 0 */
                __m256i sc = _mm256_sub_epi16(vm1, _mm256_add_epi16(mk, mk)); /* +1/-1 */
                __m256i dg = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(d2 + i)), sc);
                __m256i u  = _mm256_loadu_si256((const __m256i *)(d1 + i));
                __m256i l  = _mm256_loadu_si256((const __m256i *)(d1 + i + 1));
                __m256i g  = _mm256_sub_epi16(_mm256_max_epi16(u, l), vg);
                _mm256_storeu_si256((__m256i *)(d0 + i + 1),
                                    _mm256_max_epi16(dg, g));
            }
        }
#endif
        {
            const short *__restrict p2 = d2;
            const short *__restrict p1 = d1;
            short       *__restrict p0 = d0;
            const char  *__restrict pa = a;
            const char  *__restrict pb = brev + (n - t);
            for (; i <= vhi; i++) {
                int s = (pa[i - 1] == pb[i]) ? 1 : -1;
                int best = p2[i] + s;
                int u = p1[i] - 2;
                int l = p1[i + 1] - 2;
                if (u > best) best = u;
                if (l > best) best = l;
                p0[i + 1] = (short)best;
            }
        }
        if (t == T) { res = d0[n + 1]; break; }
        tp = d2; d2 = d1; d1 = d0; d0 = tp;
    }
    return res;
}

static int nw_band_i32(int n, const char *a, const char *brev, int K, int *scratch)
{
    const int stride = n + 3;
    int *d2 = scratch, *d1 = scratch + stride, *d0 = scratch + 2 * stride, *tp;
    const int T = 2 * n;
    int res = 0, z;

    for (z = 0; z < 3 * stride; z++) scratch[z] = NEG32V;

    for (int t = 0;; t++) {
        int lo = (t - K + 1) >> 1;
        int hi = (t + K) >> 1;
        if (lo < 0) lo = 0;
        if (lo < t - n) lo = t - n;
        if (hi > n) hi = n;
        if (hi > t) hi = t;

        d0[lo] = NEG32V;
        d0[hi + 2] = NEG32V;

        int vlo = lo, vhi = hi;
        if (lo == 0) { d0[1] = -2 * t; vlo = 1; }
        if (hi == t) { d0[t + 1] = -2 * t; vhi = t - 1; }

        int i = vlo;
#if defined(__AVX2__)
        {
            const __m256i vm1 = _mm256_set1_epi32(-1);
            const __m256i vg  = _mm256_set1_epi32(2);
            const int base = n - t;
            for (; i + 7 <= vhi; i += 8) {
                __m128i av = _mm_loadl_epi64((const __m128i *)(a + i - 1));
                __m128i bv = _mm_loadl_epi64((const __m128i *)(brev + base + i));
                __m256i mk = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(av, bv));
                __m256i sc = _mm256_sub_epi32(vm1, _mm256_add_epi32(mk, mk));
                __m256i dg = _mm256_add_epi32(
                                 _mm256_loadu_si256((const __m256i *)(d2 + i)), sc);
                __m256i u  = _mm256_loadu_si256((const __m256i *)(d1 + i));
                __m256i l  = _mm256_loadu_si256((const __m256i *)(d1 + i + 1));
                __m256i g  = _mm256_sub_epi32(_mm256_max_epi32(u, l), vg);
                _mm256_storeu_si256((__m256i *)(d0 + i + 1),
                                    _mm256_max_epi32(dg, g));
            }
        }
#endif
        {
            const int  *__restrict p2 = d2;
            const int  *__restrict p1 = d1;
            int        *__restrict p0 = d0;
            const char *__restrict pa = a;
            const char *__restrict pb = brev + (n - t);
            for (; i <= vhi; i++) {
                int s = (pa[i - 1] == pb[i]) ? 1 : -1;
                int best = p2[i] + s;
                int u = p1[i] - 2;
                int l = p1[i + 1] - 2;
                if (u > best) best = u;
                if (l > best) best = l;
                p0[i + 1] = best;
            }
        }
        if (t == T) { res = d0[n + 1]; break; }
        tp = d2; d2 = d1; d1 = d0; d0 = tp;
    }
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    /* (1) the straight walk: ungapped diagonal score, O(n), auto-vectorized. */
    int matches = 0;
    for (int i = 0; i < n; i++) matches += (a[i] == b[i]);
    int best = 2 * matches - n;

    /* (2) accounting identity  score = n - 5k - 2MM  =>  k <= (n - best)/5,
           and |i-j| <= k, so this is a provably sufficient band radius.     */
    int Kcap = (n - best) / 5;
    if (Kcap <= 0) return best;              /* no gap can ever pay: exact  */
    if (Kcap > n) Kcap = n;

    char *brev = (char *)malloc((size_t)n + 64);
    if (!brev) return best;
    for (int i = 0; i < n; i++) brev[i] = b[n - 1 - i];
    memset(brev + n, 0, 64);

    const int use16 = (n <= 15000);          /* |scores| <= 1.8n < 29000 */
    size_t cells = 3 * ((size_t)n + 3);
    void *scr = malloc(cells * (use16 ? sizeof(short) : sizeof(int)) + 64);
    if (!scr) { free(brev); return best; }

    /* (3) probe with a narrow band, let the score it finds shrink the band. */
    int K = Kcap < 16 ? Kcap : 16;
    for (;;) {
        int s = use16 ? nw_band_i16(n, a, brev, K, (short *)scr)
                      : nw_band_i32(n, a, brev, K, (int *)scr);
        if (s > best) best = s;
        int need = (n - best) / 5;           /* best <= optimum => need >= k_opt */
        if (K >= need) break;                /* band provably contained an optimum */
        long long nk = (long long)K * 16;
        if (nk > need) nk = need;
        K = (int)nk;
    }

    free(scr);
    free(brev);
    return best;
}
```

## PREDICTION

PREDICTION: 20

## MEASUREMENT

**No measurement was performed — the session provides no tools.** `alignment_bench` / `alignment_contract` were listed but are not callable here, so I could not compile, verify against the reference, or time anything. Reporting that plainly rather than inventing numbers: the prediction above is unvalidated, and the "improve it at most four times" loop did not happen.

What I did instead, in lieu of measurement:

- **Hand-traced the wavefront machinery** on `n=1 (A/A)` → 1; `n=3 (AAA/CCC)`, `K=1`, every cell of `t=0..3` against the textbook table → `dp[1][1]=-1`, `dp[1][2]=-3`, `dp[2][1]=-3`, all matching, with the `-INF` guards landing exactly on the out-of-band cells `(0,2)` and `(2,0)`.
- **Proved the guards are never over-read**: `lo(t)` is non-decreasing and `hi(t) ≤ hi(t-1)+1`, `hi(t) ≤ hi(t-2)+1`, so reads at `[lo(t)-1, hi(t)]` and `[lo(t)-1, hi(t)-1]` stay inside `[lo, hi]` ∪ the two guard slots of the producing diagonal.
- **Proved no `-INF` ever becomes a stored value**: every general cell `(i≥1, j≥1)` has its diagonal predecessor in-band and already computed, and that term exceeds `NEG+1`, so sentinels never win a `max` and never cascade. This is also what makes the int16 range bound `[-(n+2K), n] ⊆ [-1.8n, n]` airtight (`K ≤ 2n/5`).
- **Checked SIMD load bounds**: the vector loop runs only while `i+15 ≤ vhi ≤ t-1` and `vhi ≤ n`, which puts the last `a` byte at `a[n-1]` and the last `brev` byte at `brev[n-1]` — no out-of-bounds read of the caller's buffers.
- **Checked exactness of the stopping rule**: every round's result is a restriction of the feasible path set, so `s ≤ S_opt`; `best ≤ S_opt` ⟹ `need = ⌊(n-best)/5⌋ ≥ ⌊(n-S_opt)/5⌋ ≥ k_opt`, so `K ≥ need` ⟹ the band contained an optimal path ⟹ `best = S_opt`.
- **Checked the interesting adversarial case** (`a = "T"+S`, `b = S+"G"`, diagonal all-mismatch): diagonal gives `-10`, `Kcap=4`, band-4 finds `5 = n - 5·1`, `need = 1 ≤ 4`, exact. This is exactly the "obvious missing bead ten steps away" case the disguised problem complains about, and the band finds it without touching the rest of the grid.

Where the 20× estimate comes from: for two independent random DNA strings, `S₀ ≈ -0.5n` and `S_opt ≈ -0.4n`, giving `K ≈ 0.28n` → ~`0.56n²` cells (1.8× fewer than the reference) × ~10–14× effective SIMD throughput at int16 × a large cache factor (3 rolling rows vs. a streamed `4(n+1)²`-byte table). For *related* sequences — the realistic aligner workload — `K` collapses to O(mutations) and the kernel goes near-linear, i.e. 100×+. 20 is the conservative middle, weighted toward the random case.

## VERDICT

The mechanism translates cleanly and, better than I expected going in, translates **exactly** rather than heuristically: the equal-length accounting identity `score = n − 5k − 2MM` turns "a nudge only pays if the disagreements are dense enough" into a hard, provable band radius, so the disguised solution's hunch-driven shortcut becomes a certified-exact Ukkonen-style banded wavefront. Nothing was swapped out for a different approach — the diagonal walk, the single-nudge test, and the "keep whichever total is bigger" are all still there, just with a proof attached and 16 lanes wide.

The honest caveat dominates the verdict: **this is unmeasured.** I believe it is correct (traced, bounds-proved, sentinel-proved) and I believe it is substantially faster than the reference, but "believe" is doing real work in that sentence and I did not get to falsify either claim. If it is run and comes in under ~5×, the most likely culprits in order are: the escalation paying its ~7 % overhead without ever exiting early on the benchmark's inputs, the per-anti-diagonal prologue dominating when `K` is small (fixable by switching to a row-wise banded loop below `K ≈ 64`), and the untapered band — a dynamic `v(i,j) + (n−i−3δ) ≤ best` prune would roughly halve the cell count on random inputs and is the first thing I would add with a working bench.