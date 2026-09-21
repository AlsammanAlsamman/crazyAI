# APPROACH

The contract requires an *exact* full Needleman-Wunsch score (match +1, mismatch -1, gap -2) — no banding or heuristic shortcuts are safe, because a single indel-heavy pair of sequences can force the optimal path arbitrarily far off the main diagonal, and the task demands exact agreement with the reference DP, not an approximation. So the O(n²) cell count is unavoidable; the only lever is constant-factor reduction: less memory traffic, better cache locality, and vectorizing as much of the per-cell arithmetic as the data dependencies allow.

The recurrence `dp[i][j] = max(diag, up, left)` has one genuinely serial dependency per row: `left = dp[i][j-1] + GAP`. But `diag = dp[i-1][j-1] + (match?+1:-1)` and `up = dp[i-1][j] + GAP` depend **only on the already-finished previous row**, so they're fully independent across `j` and can be computed in one branch-free, auto-vectorizable sweep. I split each row into two passes:

1. A vectorizable pass computing `pre[j] = max(diag, up)` for the whole row (SIMD-friendly: broadcast `a[i-1]`, compare against the `b` vector, blend ±1, add, max with `up`).
2. A cheap serial scan folding in the left/gap dependency: `curr[j] = max(pre[j], curr[j-1] + GAP)` — just one add + one compare-select per element, a 2-op dependency chain that pipelines well even though it can't be vectorized.

This also drops memory from the reference's O(n²) allocation to O(n) (two rolling rows), removing a large source of allocation cost and cache/DRAM traffic — for any n large enough to matter, that alone should beat the baseline, independent of the vectorization gain.

# ARTIFACT

```c
#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    size_t width = (size_t)n + 1;
    int *prev = (int *)malloc(width * sizeof(int));
    int *curr = (int *)malloc(width * sizeof(int));
    if (!prev || !curr) { free(prev); free(curr); return 0; }

    for (size_t j = 0; j <= (size_t)n; j++) prev[j] = (int)j * GAP;

    for (int i = 1; i <= n; i++) {
        curr[0] = i * GAP;
        const char ai = a[i - 1];

        const int  * restrict pv = prev;
        int        * restrict cv = curr;
        const char * restrict bp = b;

        /* Pass 1: fully independent across j -> vectorizable.
         * pre[j] = max(diag, up), both derived only from finished row i-1. */
        #pragma omp simd
        for (int j = 1; j <= n; j++) {
            int diag = pv[j - 1] + ((ai == bp[j - 1]) ? MATCH : MISMATCH);
            int up   = pv[j] + GAP;
            cv[j] = diag > up ? diag : up;
        }

        /* Pass 2: serial gap-open propagation from the left.
         * Cheap 2-op chain (add + select), cannot vectorize but is short. */
        for (int j = 1; j <= n; j++) {
            int left = cv[j - 1] + GAP;
            if (left > cv[j]) cv[j] = left;
        }

        int *tmp = prev;
        prev = curr;
        curr = tmp;
    }

    int result = prev[n];
    free(prev);
    free(curr);
    return result;
}
```

Correctness check: `dp[i][j] = max(diag, up, left) = max(max(diag, up), left) = max(pre[j], curr[j-1] + GAP)`, which is exactly what pass 2 computes, seeded from pass 1's `pre[j]` values stored in `cv[j]`. Row/column base cases (`i*GAP`, `j*GAP`) match the reference exactly, and the final answer is `dp[n][n]` = `prev[n]` after the last swap.

PREDICTION: 3

# MEASUREMENT

No tools (including `alignment_bench`) are available in this session — the system explicitly disabled tool access here, so I could not execute the benchmark or the reference comparison myself. I am not going to fabricate a measured number; per the instructions to report failure as plainly as success, I'm flagging this rather than inventing a result. The prediction above (~3x over the naive O(n²)-allocation triple-branch baseline) is reasoning-based only: it comes from (a) removing the O(n²) malloc/write traffic in favor of O(n) rolling rows, and (b) vectorizing the diag/up computation which is the majority of the arithmetic, leaving only a short 2-op serial scan per row.

# VERDICT

Implementation is complete and, by manual trace, correct against the reference recurrence and boundary conditions. Performance claim is unverified in this session due to tool unavailability — treat the "3x" prediction as an estimate pending an actual `alignment_bench` run, not a confirmed result.