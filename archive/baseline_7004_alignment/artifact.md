## APPROACH

The reference implementation fills the DP grid row-by-row. Its inner loop has a genuine loop-carried dependency (`dp[i][j]` needs `dp[i][j-1]`, just computed in the same iteration), so a compiler cannot vectorize the `j` loop — every cell is essentially processed as a serial scalar chain with a data-dependent branch.

The recurrence is still exact O(n²) work (no valid shortcut exists for worst-case global alignment — an adversarial input can force every cell to matter), but the *order* of evaluation can change without changing any value: cells on the same anti-diagonal (`i+j = k`) never depend on each other, only on the two previous diagonals. So instead of rows, I sweep diagonals. Within one diagonal the loop over `i` has **no cross-iteration dependency**, which lets `-O3 -march=native` (and an explicit `#pragma omp simd`) vectorize the compare/add/max chain across lanes.

The one wrinkle: for fixed `k`, the two dependent-diagonal reads (`i-1`) are contiguous, but the character comparison `a[i-1]==b[k-i-1]` walks `b` backwards as `i` increases. Reversing `b` once (`rb[m]=b[n-1-m]`) turns that into a forward, contiguous, constant-offset access (`rb[i+(n-k)]`), so the whole inner loop becomes simple aligned-stride array ops — ideal for auto-vectorization.

I verified correctness by hand-tracing the full 2×2 DP table against the diagonal sweep (every intermediate value matched the reference table exactly, including boundary rows/columns which are injected directly as `-2k` rather than recomputed).

I did **not** add OpenMP thread-level parallelism across diagonals: with 2n diagonals each needing a barrier, thread-spawn/sync overhead would dominate for small-to-medium n and this can't be tuned without measurement, so I kept the win that's unconditionally safe (SIMD) and skipped the one that's situational (threading).

## ARTIFACT

```c
#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    /* Reverse b once so the per-diagonal character comparison becomes a
       contiguous, constant-offset access instead of a backward stride. */
    char *rb = (char*)malloc((size_t)n);
    for (int t = 0; t < n; t++) rb[t] = b[n - 1 - t];

    int *buf0 = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *buf1 = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *buf2 = (int*)malloc((size_t)(n + 1) * sizeof(int));

    int *prev2 = buf0; /* diagonal k-2 */
    int *prev1 = buf1; /* diagonal k-1 */
    int *cur   = buf2; /* diagonal k   */

    int total_diag = 2 * n;
    for (int k = 0; k <= total_diag; k++) {
        int rowlo = (k > n) ? (k - n) : 0;
        int rowhi = (k < n) ? k : n;
        int lo = (k - n > 1) ? (k - n) : 1;
        int hi = (k - 1 < n) ? (k - 1) : n;

        if (lo <= hi) {
            int off = n - k;
            const int  * __restrict p2  = prev2;
            const int  * __restrict p1  = prev1;
            int        * __restrict pc  = cur;
            const char * __restrict pa  = a;
            const char * __restrict prb = rb;

            #pragma omp simd
            for (int i = lo; i <= hi; i++) {
                int sc    = (pa[i - 1] == prb[i + off]) ? MATCH : MISMATCH;
                int diagv = p2[i - 1] + sc;
                int upv   = p1[i - 1] + GAP;
                int leftv = p1[i]     + GAP;
                int best  = diagv > upv ? diagv : upv;
                if (leftv > best) best = leftv;
                pc[i] = best;
            }
        }
        if (rowlo == 0) cur[0] = -2 * k;   /* dp[0][k] boundary */
        if (rowhi == k) cur[k] = -2 * k;   /* dp[k][0] boundary */

        int *t = prev2; prev2 = prev1; prev1 = cur; cur = t;
    }

    int result = prev1[n];

    free(rb); free(buf0); free(buf1); free(buf2);
    return result;
}
```

## PREDICTION: 4

## MEASUREMENT

No tool access was available in this session to actually invoke `alignment_bench` — the instructions for this turn explicitly state no tools are available here and that the compile/measure step happens in the external pipeline after this response. I therefore cannot report a real measured number now, and I'm saying so plainly rather than inventing one.

Reasoned expectation (not a run): the diagonal sweep does the same O(n²) arithmetic as the reference, so no algorithmic-class win is possible for exact global alignment. The win is purely constant-factor: removing the inner loop-carried dependency should let GCC emit AVX2 (`vpmaxsd`/`vpaddd`) code processing ~8 int32 lanes per iteration where the reference is forced to scalar, data-dependent-branch execution. Accounting for the extra O(n) bookkeeping (reverse-buffer build, three-way buffer rotation, boundary writes) and the fact real vector width won't be a clean 8x due to remainder handling and short leading/trailing diagonals, I'd expect roughly a 3–5x wall-clock improvement for moderate-to-large n, converging toward the reference's speed for very small n where per-diagonal overhead dominates.

## VERDICT

Pending actual measurement by the pipeline. The implementation is verified correct by hand-trace against the reference DP table (full match on a 2×2 case including all intermediate diagonal/up/left terms), and is structurally set up to benefit from SIMD auto-vectorization without changing the DP's numerical semantics — but I have not been able to confirm the predicted speedup with a real run in this session.