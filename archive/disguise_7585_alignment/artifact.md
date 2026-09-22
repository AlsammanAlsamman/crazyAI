## APPROACH

The three disguised solutions map directly onto Needleman–Wunsch score computation:

- **Solution 1** is literally the reference `kernel()` given in the contract: full O(n²) matrix, filled in row-major order using diag/up/left dependencies.
- **Solution 2** is the classic linear-space DP optimization: since we only need the final score (not a traceback), we never need more than the previous row and the current row. This drops memory from O(n²) to O(n), which for any non-trivial n turns the problem from memory-bandwidth-bound (touching a huge matrix that blows out cache) into something that stays resident in L1/L2 the whole time.
- **Solution 3** (Hirschberg meet-in-the-middle) is the right tool when you need the *alignment path itself* in O(n) space — it recomputes each half twice to avoid storing the matrix while still allowing traceback. Since the contract only returns a **score**, that extra machinery buys nothing here; Solution 2 already gives O(n) space for score-only computation without Hirschberg's ~2x recomputation overhead.

So Solution 2 is the direct, correct translation. On top of it, I exploit a structural fact about the recurrence: for a fixed row `i`, the `diag` term (`prev[j-1] + score(a[i-1],b[j-1])`) and the `up` term (`prev[j] + GAP`) depend only on the **previous, already-fully-computed row** — they have no dependency on other cells in the current row, so they're trivially parallel across `j`. Only the `left` term (`curr[j-1] + GAP`) creates a genuine sequential chain within a row. I therefore split each row into two passes:
1. A SIMD (AVX2, 8×int32 lanes) pass computing `cand[j] = max(diag, up)` for the whole row — vectorizable, no cross-lane dependency.
2. A cheap scalar left-to-right scan folding in the `left`/gap term: `curr[j] = max(cand[j], curr[j-1] + GAP)` — just one add + one compare per element, so even though it's serial it's nearly free.

This keeps the total work O(n²) (unavoidable — NW's DP dependency structure is provably sequential/quadratic for exact scoring) but cuts memory traffic ~n-fold vs. the reference and vectorizes the majority of the per-cell arithmetic.

## ARTIFACT

```c
#include <stdlib.h>
#include <immintrin.h>

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *curr = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cand = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !curr || !cand) { free(prev); free(curr); free(cand); return 0; }

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    const __m256i match_v    = _mm256_set1_epi32(MATCH);
    const __m256i mismatch_v = _mm256_set1_epi32(MISMATCH);
    const __m256i gap_v      = _mm256_set1_epi32(GAP);

    for (int i = 1; i <= n; i++) {
        curr[0] = i * GAP;

        __m256i a_v = _mm256_set1_epi32((int)(unsigned char)a[i - 1]);

        int j = 1;
        for (; j + 8 <= n + 1; j += 8) {
            __m256i prev_diag = _mm256_loadu_si256((const __m256i *)(prev + j - 1));
            __m256i prev_up   = _mm256_loadu_si256((const __m256i *)(prev + j));

            __m128i b_bytes = _mm_loadl_epi64((const __m128i *)(b + j - 1));
            __m256i b_v     = _mm256_cvtepu8_epi32(b_bytes);

            __m256i eq_mask = _mm256_cmpeq_epi32(a_v, b_v);
            __m256i sc      = _mm256_blendv_epi8(mismatch_v, match_v, eq_mask);

            __m256i diag = _mm256_add_epi32(prev_diag, sc);
            __m256i up   = _mm256_add_epi32(prev_up, gap_v);
            __m256i best = _mm256_max_epi32(diag, up);

            _mm256_storeu_si256((__m256i *)(cand + j), best);
        }
        for (; j <= n; j++) {
            int diag = prev[j - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int up   = prev[j] + GAP;
            cand[j]  = (diag > up) ? diag : up;
        }

        int left = curr[0];
        for (int jj = 1; jj <= n; jj++) {
            int c  = cand[jj];
            int lg = left + GAP;
            int v  = (c > lg) ? c : lg;
            curr[jj] = v;
            left = v;
        }

        int *tmp = prev; prev = curr; curr = tmp;
    }

    int result = prev[n];
    free(prev); free(curr); free(cand);
    return result;
}
```

PREDICTION: 6

## MEASUREMENT

No benchmarking tool (`alignment_bench`) is actually available in this session — the tool list I was given here contains only unrelated integrations (Gmail/Slack/Drive/Docs), not the alignment/symbolic/unconventional tools the task references. I cannot run or fabricate real numbers, so I'm reporting that plainly rather than inventing a measurement: **no empirical measurement was performed by me**; per the task setup, the external pipeline is expected to compile and measure the artifact above.

## VERDICT

Unverified pending the external harness's run. Reasoning-based expectation: for the reference O(n²)-matrix version, runtime for moderately large n is dominated by allocating and writing a full (n+1)×(n+1) int array (e.g. ~400MB of traffic at n=10000), which thrashes cache. The submitted version does the same O(n²) amount of arithmetic (this is inherent — exact NW scoring cannot be done in sub-quadratic time in general) but with O(n) memory footprint that stays cache-resident, plus AVX2 vectorizing the diag/up half of each cell's computation, leaving only a cheap serial add-compare scan for the gap term. I'd expect a real speedup, plausibly in the several-times range for large n, but I have no measured confirmation and would trust the harness's numbers over my estimate if they diverge.