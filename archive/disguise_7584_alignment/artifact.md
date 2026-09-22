# APPROACH

Of the three disguised solutions, **Solution 3 (greedy + spot-check for gaps)** is unsound for this target: general DP alignment scores are not always dominated by long match/mismatch runs, and "detect a slip by noticing a bad streak, then locally test one gap" is a heuristic that will silently diverge from the exact Needleman-Wunsch optimum on many inputs (it's essentially a banded/greedy heuristic, not exact). Since the contract demands *exactly matching* the reference DP, I reject it.

**Solution 1** (crawl the full grid, one cell at a time, in row-major order) is exactly what the reference C code already does — correct, but wasteful: it allocates and touches an O(n²) matrix, which for even moderate n blows past cache and turns the kernel into a memory-bandwidth-bound loop with a genuine sequential dependency (the "left" neighbor) that also blocks vectorization within a row.

**Solution 2** (fill whole anti-diagonals at once, since cells on one anti-diagonal only depend on the previous two anti-diagonals, never on each other) is the one that translates directly into a real speedup:

- Anti-diagonal `k = i+j`: all cells on diagonal `k` depend only on diagonals `k-1` and `k-2`, so within a diagonal there is *no* dependency between lanes → the loop is trivially SIMD-vectorizable (and *exactly* reproduces the same recurrence/order-of-evaluation contract: everything on diagonal k-1 must be finished before diagonal k starts, which the outer sequential k-loop enforces).
- This also lets me drop memory from O(n²) to O(n) (three rolling diagonal buffers), which is the dominant real-world win: for large n the reference's malloc of `(n+1)²` ints thrashes cache/RAM bandwidth; three `O(n)` buffers stay resident in L2.
- To make the inner loop have *unit-stride* accesses on both sequences (needed for the compiler to auto-vectorize the match/mismatch compare), I reverse `b` once (`brev`) so both `a[i-1]` and `brev[n-k+i]` advance by +1 as `i` increases along the diagonal — the standard SIMD-diagonal alignment trick used in tools like SSW/KSW2.
- I use `#pragma omp simd` only (no `parallel for`) — thread-team spin-up/join cost per diagonal (up to `2n` diagonals) is not obviously worth it without being able to measure the actual benchmark's `n`, so I take the safe, robust win (memory + vectorization) rather than risk regressing with threading overhead I can't tune here.

This is Solution 2, applied faithfully: same recurrence, same "must finish everything before" ordering, same final answer — just batching the truly independent cells together instead of visiting them one at a time.

# ARTIFACT

```c
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAPMAG 2   /* gap penalty magnitude; contributes -GAPMAG to the score */

static inline int imax3(int x, int y, int z) {
    int m = (x > y) ? x : y;
    return (m > z) ? m : z;
}

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *buf0 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf1 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf2 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    char *brev = (char *)malloc((size_t)n * sizeof(char));

    for (int t = 0; t < n; t++) brev[t] = b[n - 1 - t];

    int *D0 = buf0; /* diagonal k-2 */
    int *D1 = buf1; /* diagonal k-1 */
    int *D2 = buf2; /* diagonal k, being filled now */

    int totalK = 2 * n;
    for (int k = 0; k <= totalK; k++) {
        int i_min = (k - n > 0) ? (k - n) : 0;
        int i_max = (k < n) ? k : n;
        int lo = i_min, hi = i_max;

        /* boundary cells: i==0 (j==k) or j==0 (i==k), only discoverable
           because diagonal k-1/k-2 (i.e. everything "before" it) is
           already fully computed -- same rule as the reference DP. */
        if (lo == 0) {
            D2[0] = -(k * GAPMAG);
            lo = 1;
        }
        if (hi == k && k <= n) {
            D2[k] = -(k * GAPMAG);
            hi = k - 1;
        }

        #pragma omp simd
        for (int i = lo; i <= hi; i++) {
            int aOff = i - 1;
            int bOff = n - k + i;   /* brev[bOff] == b[j-1], unit stride in i */
            int sc = (a[aOff] == brev[bOff]) ? MATCH : MISMATCH;
            int diag = D0[i - 1] + sc;
            int up   = D1[i - 1] - GAPMAG;
            int left = D1[i]     - GAPMAG;
            D2[i] = imax3(diag, up, left);
        }

        int *tmp = D0;
        D0 = D1;
        D1 = D2;
        D2 = tmp;
    }

    int result = D1[n];

    free(buf0);
    free(buf1);
    free(buf2);
    free(brev);
    return result;
}
```

PREDICTION: 4

# MEASUREMENT

No benchmarking tool (`alignment_bench`) is actually present in this session's tool list, and I was told the pipeline will compile and measure this artifact externally rather than me running it here — so I have not executed any measurement myself. The prediction above (≈4x over the naive O(n²)-memory row-major reference) is based purely on reasoning: eliminating the O(n²) matrix allocation/traffic (dominant cost for anything beyond small n) plus enabling SIMD across each anti-diagonal via unit-stride access on both sequences.

# VERDICT

Unverified in this session — correctness reasoning (traced by hand for n=0, n=1, and general k) supports exact equivalence to the reference recurrence and evaluation order, and the algorithmic change (O(n²)→O(n) memory, vectorized anti-diagonal) is a well-established real speedup technique for Needleman-Wunsch. Actual speedup number depends on the pipeline's external measurement, which I cannot produce here.