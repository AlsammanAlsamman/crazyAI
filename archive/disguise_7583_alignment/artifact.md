APPROACH

The reference kernel returns only `dp[n][n]`, never a traceback, so there is no need to ever materialize the full O(n²) matrix (Solution 1's literal "fill the whole giant scoreboard and keep it"). Each cell only needs the row above and the values already computed to its left in the current row (Solution 3: "keep only a slim moving window — the last row/diagonal — not the whole chart"). That reduces memory from O(n²) to O(n), which for realistic n turns this from a cache-thrashing, malloc-heavy workload into one where both working rows stay resident in L1/L2 cache. This is *always* exactly correct — it's the same recurrence, same fill order, same boundary conditions as the reference, just without keeping cells that will never be read again. It requires no assumption about score bounds or a bounded slip radius, so unlike a banded variant (Solution 2) it stays exact on adversarial inputs where the optimal path could in principle wander.

I additionally hoist the match/mismatch decision for each row into its own short loop over `j` (`sc[j] = (ai==b[j]) ? MATCH : MISMATCH`), separate from the sequential min/max recurrence. That comparison loop has no cross-`j` dependency, so `-O3 -march=native` can auto-vectorize it (SIMD byte compare + select) even though the actual DP recurrence itself is an inherently sequential scan along `j` (the `left` term needs `cur[j-1]` from the same row) and can't be vectorized directly without a full diagonal-wavefront rewrite. This keeps the sequential inner loop to pure integer max/add with no branching on characters, improving pipelining without risking correctness bugs from a more invasive anti-diagonal SIMD rewrite that I can't verify against a test harness in this session.

ARTIFACT

```c
#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    int *restrict prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *restrict cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *restrict sc   = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        const char *restrict bp = b;

        /* independent per-column match/mismatch scores: auto-vectorizable */
        for (int j = 0; j < n; j++) {
            sc[j] = (ai == bp[j]) ? MATCH : MISMATCH;
        }

        cur[0] = i * GAP;
        int diag = prev[0];

        for (int j = 1; j <= n; j++) {
            int d    = diag + sc[j - 1];
            int up   = prev[j] + GAP;
            int left = cur[j - 1] + GAP;

            int best = d > up ? d : up;
            if (left > best) best = left;

            cur[j] = best;
            diag = prev[j];
        }

        int *t = prev; prev = cur; cur = t;
    }

    int result = prev[n];
    free(prev);
    free(cur);
    free(sc);
    return result;
}
```

PREDICTION: 8

MEASUREMENT

No benchmarking tool (`alignment_bench`) was actually invoked or available in this session — the harness stated explicitly that no tools were available and that the pipeline would compile/measure this after the fact. So this is a pre-measurement estimate, not an observed number: the reasoning is that the O(n²)→O(n) memory reduction eliminates the dominant cost for the reference implementation at moderate-to-large n (malloc of an (n+1)² int matrix, plus the cache misses of striding across it), while the per-row score precomputation shaves some branch/compare overhead off the remaining sequential scan. I have not seen an actual timing number to compare against or refine from.

VERDICT

Implementation is a straightforward, provably-equivalent rewrite of the reference recurrence (identical boundary values, identical three-way max, identical fill order row-by-row/left-to-right), just without retaining cells that are never read again, so it should be exact on every input, including adversarial ones where a banded approach would risk being wrong. Speedup is expected but unverified in this session — it rests on memory-locality reasoning rather than a measured comparison, so treat the PREDICTION as an untested estimate rather than a confirmed result.