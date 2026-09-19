# APPROACH

The reference implementation is a textbook row-major Needleman-Wunsch: it allocates the *full* `(n+1)×(n+1)` score matrix and fills it row by row. Two things make it slow:

1. **O(n²) memory traffic** — for any non-trivial `n` the matrix blows past L2/L3 cache (and even physical memory for large `n`), so the kernel becomes memory-bandwidth/TLB bound rather than compute bound.
2. **A true loop-carried dependency inside each row** — `dp[i][j]` needs `dp[i][j-1]` (the "left" neighbor), so consecutive iterations of the inner `j` loop cannot be reordered or vectorized; the CPU stalls on the add→compare→max→store latency chain every single cell.

Both problems have the same known fix: process the grid **by anti-diagonal** instead of by row. On diagonal `k = i+j`, every cell `dp[i][j]` depends only on cells from diagonals `k-1` and `k-2`, which are already fully computed — so *all cells on the same diagonal are mutually independent*. That:

- lets me keep only three O(n) diagonal buffers instead of the O(n²) matrix (huge memory-traffic win, and it also removes the giant `malloc`),
- removes the intra-row serial dependency entirely, so the inner loop is embarrassingly parallel/vectorizable (no loop-carried data dependency at all — only reads of already-finalized buffers), which I expose to the compiler with `#pragma omp simd`.

Total arithmetic work is still O(n²) (fundamentally unavoidable for exact global alignment — every cell must be visited, matching the contract's own listed assumptions), but the constant factor drops substantially because the memory footprint shrinks from O(n²) to O(n) and the per-cell latency chain is broken.

Boundary handling: the two edge cells of each diagonal (`i=0` or `j=0`) are set directly from the closed form `i*GAP`/`j*GAP`; the interior cells use the three-way max recurrence. Diagonals `k=0` and `k=1` are seeded explicitly, then `k=2..2n` are computed in a loop, rotating three buffers (no extra allocation per step). The final answer is the single cell of diagonal `k=2n`.

I deliberately did *not* hand-roll AVX2 intrinsics for the character comparison: `b` is walked in the opposite (decreasing) direction from `a` within a diagonal, so a correct manual byte-reversal/widening shuffle sequence is easy to get subtly wrong without being able to compile-test it here. Correctness matters more than a speculative extra factor, so I rely on `#pragma omp simd` (safe — it only asserts something that is actually true here) plus the O(n) memory win, which is the dominant effect anyway.

# ARTIFACT

```c
#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int *prev2 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal k-2 */
    int *prev1 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal k-1 */
    int *cur   = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal k   */

    /* k = 0 : single cell dp[0][0] = 0 */
    prev2[0] = 0;

    /* k = 1 : dp[0][1] = GAP, dp[1][0] = GAP (if n >= 1, always true here) */
    {
        int lo = (1 - n > 0) ? (1 - n) : 0;
        int hi = (1 < n) ? 1 : n;
        for (int i = lo; i <= hi; i++) {
            int j = 1 - i;
            prev1[i] = (i == 0) ? j * GAP : i * GAP;
        }
    }

    for (int k = 2; k <= 2 * n; k++) {
        int lo = (k - n > 0) ? (k - n) : 0;
        int hi = (k < n) ? k : n;
        int iStart = lo, iEnd = hi;

        /* boundary cells (only present while k <= n) */
        if (lo == 0) {
            cur[0] = k * GAP;          /* dp[0][k] */
            iStart = 1;
        }
        if (hi == k && k <= n) {
            cur[k] = k * GAP;          /* dp[k][0] */
            if (iEnd == k) iEnd = k - 1;
        }

        #pragma omp simd
        for (int i = iStart; i <= iEnd; i++) {
            int j = k - i;
            int diag = prev2[i - 1] + ((a[i - 1] == b[j - 1]) ? MATCH : MISMATCH);
            int up   = prev1[i - 1] + GAP;
            int left = prev1[i]     + GAP;
            int m = diag > up ? diag : up;
            m = m > left ? m : left;
            cur[i] = m;
        }

        int *tmp = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur = tmp;
    }

    int result = prev1[n];
    free(prev2);
    free(prev1);
    free(cur);
    return result;
}
```

# PREDICTION: 5

# MEASUREMENT

No benchmarking tools (`alignment_bench`/`alignment_contract`) were exposed in this session — the environment explicitly stated no tools are available here and that the external pipeline would compile and measure the artifact after submission. I have not personally executed a timed run, so I cannot report an observed number; the prediction above (~5x over the naive O(n²)-memory row-major reference) is a reasoned estimate based on: eliminating the O(n²) allocation/memory-traffic (three O(n) buffers instead of one giant matrix), and breaking the per-row loop-carried "left" dependency that forces the reference's inner loop to run at add-compare-store latency rather than throughput. I did not run it myself, so this number should be treated as unverified until the pipeline reports actual timings.

# VERDICT

Correctness: derived and hand-traced against the recurrence for `n=1` and `n=2` boundary diagonals (`k=0,1,2,n,n+1,2n`); the anti-diagonal transform is a standard, provably-equivalent reformulation of the same Needleman-Wunsch recurrence (same match/mismatch/gap constants, same dependency set, just reordered), so it should match the reference exactly for all `n`, including `n=0`.

Performance: not empirically measured in this session (no tool access) — reported honestly as unmeasured rather than fabricated. The prediction (~5x) stands as an estimate; if the harness reports the actual number, the true comparison is O(n) memory / broken dependency chain (this kernel) vs. O(n²) memory / serialized inner loop (reference), so I'd expect the gap to widen further as `n` grows and the reference's matrix stops fitting in cache.