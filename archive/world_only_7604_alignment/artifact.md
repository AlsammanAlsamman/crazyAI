## APPROACH

The reference solution is a textbook row-major Needleman-Wunsch: O(n²) time, O(n²) memory, and its inner loop has a **serial dependency along j** (`left = dp[i][j-1]`, computed earlier in the very same inner loop), which blocks auto-vectorization of that loop entirely. It also pays for a full n×n `malloc`, which for larger n means heavy memory traffic and cache misses.

Key restructuring: reindex the DP by **anti-diagonals** (k = i+j) instead of rows. Within one anti-diagonal, every cell's three dependencies (diag, up, left) live on the *previous two* anti-diagonals only — never on another cell of the *same* anti-diagonal. That removes the loop-carried dependency, so the inner loop over cells of one anti-diagonal is embarrassingly parallel/vectorizable, while the total work is still exactly n² cell evaluations with the identical recurrence (so the *value* computed, hence the max score, is unchanged — tie-breaking order doesn't affect the numeric max).

To keep both index streams contiguous and increasing as the anti-diagonal is swept (a[i-1] forward, b[j-1] would normally run backward), I precompute a reversed copy of `b` once (O(n)), so both operands of the match/mismatch test become simple contiguous forward reads. I keep only three O(n) buffers (rolling anti-diagonals) instead of the O(n²) matrix, and split off the two boundary cells (i=0, j=0) per diagonal so the interior loop is branch-free and `#pragma omp simd`-able (the harness already passes `-O3 -march=native -fopenmp`).

Net effect: same O(n²) arithmetic work, but O(n) memory instead of O(n²), and a genuinely vectorizable inner loop instead of a serially-dependent one.

## ARTIFACT

```c
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    /* reversed copy of b: br[m] = b[n-1-m], lets both operand streams
       be read forward/contiguously as the anti-diagonal sweeps i upward */
    char *br = (char *)malloc((size_t)n * sizeof(char));
    for (int m = 0; m < n; m++) br[m] = b[n - 1 - m];

    int *D2 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diag k-2 */
    int *D1 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diag k-1 */
    int *D0 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diag k   */

    int result = 0;
    int twoN = 2 * n;

    for (int k = 0; k <= twoN; k++) {
        int lo = (k > n) ? (k - n) : 0;
        int hi = (k < n) ? k : n;

        if (k <= n) {
            /* both boundary cells (i=0,j=k) and (i=k,j=0) have value k*GAP */
            int bv = k * GAP;
            D0[0] = bv;
            D0[k] = bv;
        }

        int ilo = (lo < 1) ? 1 : lo;
        int ihiCap = k - 1;
        int ihi = (hi < ihiCap) ? hi : ihiCap;

        const char * restrict arow = a;
        const char * restrict brow = br;
        int * restrict d2 = D2;
        int * restrict d1 = D1;
        int * restrict d0 = D0;
        int off = n - k;

        #pragma omp simd
        for (int i = ilo; i <= ihi; i++) {
            int match  = (arow[i - 1] == brow[off + i]) ? MATCH : MISMATCH;
            int diagv  = d2[i - 1] + match;
            int upv    = d1[i - 1] + GAP;
            int leftv  = d1[i]     + GAP;
            int best = diagv;
            if (upv   > best) best = upv;
            if (leftv > best) best = leftv;
            d0[i] = best;
        }

        if (k == twoN) {
            result = D0[n];
        }

        int *tmp = D2;
        D2 = D1;
        D1 = D0;
        D0 = tmp;
    }

    free(D2);
    free(D1);
    free(D0);
    free(br);

    return result;
}
```

Correctness reasoning: hand-traced for n=1 and n=2 by comparing against the direct dp[i][j] recurrence — matches exactly at every step, including the boundary values and the final `dp[n][n]` cell extracted at k=2n.

## PREDICTION: 4

## MEASUREMENT

No tool access was available in this session (explicitly stated: "No tools are available in this session... the pipeline will compile and measure it"), so `alignment_bench`/`alignment_contract` could not be run here. I have not executed or timed this code myself — the number above is a reasoned estimate (O(n) memory vs O(n²), removal of the serial left-dependency enabling SIMD on the interior loop), not an empirical result. The external pipeline is expected to supply the actual measurement.

## VERDICT

Unverified by direct measurement in this session. Expected outcome based on algorithmic analysis: correct (score-only DP value is invariant to the reformulation, verified by hand-trace) and faster than the naive O(n²)-memory, serially-dependent reference, primarily from (a) dropping memory from O(n²) to O(n), and (b) making the inner loop free of loop-carried dependencies so `-O3 -march=native` can vectorize it, versus the reference's inherently serial `left` dependency. If the actual measurement contradicts this, the likely failure points to check first are: anti-diagonal boundary indexing (off-by-one in `ilo`/`ihi`/`off`) and whether the compiler actually vectorizes the branchy `if (upv>best)` reduction as well as hoped.