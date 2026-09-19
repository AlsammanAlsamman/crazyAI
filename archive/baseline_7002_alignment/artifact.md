# APPROACH

The reference solution allocates a full `(n+1)×(n+1)` DP matrix and fills it in strict row/column order. That's O(n²) memory and O(n²) cache-unfriendly traffic for large n, even though only the previous row is ever needed again (linear gap penalty, no affine-gap lookback).

Key restructuring:
1. **O(n) memory** — keep only `prev` (row i-1) and `cur` (row i), plus a scratch `cand` row. This collapses the matrix to three vectors, dramatically shrinking the working set and memory bandwidth for anything beyond small n.
2. **Split the recurrence to expose SIMD.** The only serial dependency in NW with a *linear* gap is the "left" term (`cur[j-1] + GAP`). The "diag" and "up" terms depend only on the *previous* row, which is already fully computed — so they have zero loop-carried dependency and vectorize freely. I compute `cand[j] = max(diag, up)` for the whole row first (branchless match/mismatch via `MISMATCH + 2*eq`, `#pragma omp simd` hint for the auto-vectorizer under `-O3 -march=native`), then do a cheap second pass folding in the serial `max(cand[j], cur[j-1]+GAP)` chain. That second pass is just one compare+add per element, essentially memory-bound and fast even though not vectorized.

This keeps the same O(n²) time complexity (which is unavoidable for exact global alignment with no assumption on score-band boundedness — banding would only be safe/correct for restricted inputs, and the contract requires exact match with the reference for arbitrary A/C/G/T sequences), but cuts memory footprint from O(n²) to O(n) and lets the compiler emit AVX2 `vpmaxsd`/`vpaddd` for the bulk of the per-row work instead of a fully scalar, matrix-bound loop.

# ARTIFACT

```c
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cand = (int *)malloc((size_t)(n + 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    for (int i = 1; i <= n; i++) {
        const char ac = a[i - 1];
        const int *restrict pr = prev;
        const char *restrict bb = b;
        int *restrict cd = cand;

        /* No loop-carried dependency here: diag/up only read the
           already-finished previous row, so this vectorizes freely. */
        #pragma omp simd
        for (int j = 1; j <= n; j++) {
            int eq   = (ac == bb[j - 1]);
            int diag = pr[j - 1] + MISMATCH + 2 * eq;  /* eq? MATCH : MISMATCH */
            int up   = pr[j] + GAP;
            cd[j] = diag > up ? diag : up;
        }

        /* Serial part is now minimal: fold in the left/gap chain. */
        int *restrict cr = cur;
        cr[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int left = cr[j - 1] + GAP;
            cr[j] = cd[j] > left ? cd[j] : left;
        }

        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[n];
    free(prev);
    free(cur);
    free(cand);
    return result;
}
```

# PREDICTION: 3.0

# MEASUREMENT

No tools were available in this session to invoke `alignment_bench`/`alignment_contract` (the environment explicitly disabled tool calls for this turn), so I could not run the live harness myself. The prediction above (~3x over the naive full-matrix baseline) is an estimate based on: (a) removing O(n²) malloc/memory-traffic in favor of O(n) working rows, and (b) exposing the diag/up computation — the majority of the arithmetic per cell — to AVX2 auto-vectorization via `-march=native`, while leaving only a cheap scalar gap-propagation pass. This is an *unmeasured estimate*; the actual number should come from the pipeline's own run of `alignment_bench` against this artifact.

# VERDICT

Correctness: the recurrence is algebraically identical to the reference (`diag = dp[i-1][j-1] + match/mismatch`, `up = dp[i-1][j] + GAP`, `left = dp[i][j-1] + GAP`, `dp[i][j] = max(...)`), with row/column-0 boundary initialization matching (`i*GAP`, `j*GAP`), so it should produce byte-identical scores to the reference for all n, including n=0. Performance claim is a reasoned prediction (O(n) memory instead of O(n²), vectorized candidate row) but is **unverified in this session** since the measurement tools were unavailable — flagging that plainly rather than asserting a measured result I don't have.