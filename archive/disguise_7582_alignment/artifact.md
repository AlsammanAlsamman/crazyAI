# APPROACH

Solution 1 (literal full matrix, O(n²) memory) is the naive baseline. Solution 2 (banded DP) is **not safe here**: since both strings have equal length n, up-moves and left-moves in the DP path must balance exactly (U=L), so a gap-pair costs -4. This is *not* always dominated by the diagonal — a shifted repeat (e.g. `a="XABCD"`, `b="ABCDX"`) makes a gapped alignment strictly better than the no-gap diagonal (score 0 vs -5). Worse, the needed "shift" can be as large as n/2 for adversarial rotated inputs, so a fixed-width band can silently give the wrong answer. Since the contract demands *exact* agreement with the reference DP, banding is disqualified for correctness.

Solution 3 (anti-diagonal wavefront) is exactly right: it computes the *identical* recurrence, in a *reordered but dependency-respecting* traversal — every cell on anti-diagonal k depends only on diagonals k-1 and k-2, so all cells on a diagonal are mutually independent and can be filled "in the same breath." This gives:
- **O(n) memory** instead of O(n²) (3 rolling diagonal buffers) → much better cache behavior, the dominant cost for large n in the naive version.
- **Vectorizable inner loop**: by pre-reversing `b` into `br`, both `a[i-1]` and `br[n-j]` are accessed contiguously as `i` increases along a diagonal, letting `-O3 -march=native` auto-vectorize the compare/max chain.
- **OpenMP-parallelizable** across the independent cells of each diagonal, gated by an `if(n>threshold)` clause so small inputs don't pay thread-team overhead.

This preserves bit-for-bit correctness (same recurrence, same base cases) while attacking the real bottleneck (cache-hostile O(n²) matrix + serial single-box crawl).

# ARTIFACT

```c
#include <stdlib.h>
#include <omp.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int *buf0 = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *buf1 = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *buf2 = (int*)malloc((size_t)(n + 1) * sizeof(int));
    char *br  = (char*)malloc((size_t)n);
    for (int x = 0; x < n; x++) br[x] = b[n - 1 - x];

    int *bufs[3] = { buf0, buf1, buf2 };

    /* k = 0 : dp[0][0] = 0 */
    bufs[0][0] = 0;

    /* k = 1 : boundary row/col only */
    {
        int k = 1;
        int imin = (k - n) > 0 ? (k - n) : 0;
        int imax = (k < n) ? k : n;
        int *cur = bufs[k % 3];
        for (int i = imin; i <= imax; i++) {
            int j = k - i;
            cur[i] = (i == 0) ? j * GAP : i * GAP;
        }
    }

    int total = 2 * n;
    int use_par = (n > 2000);

    #pragma omp parallel if(use_par)
    {
        for (int k = 2; k <= total; k++) {
            int imin = (k - n) > 0 ? (k - n) : 0;
            int imax = (k < n) ? k : n;
            int *cur   = bufs[k % 3];
            int *prev1 = bufs[(k + 2) % 3]; /* diagonal k-1 */
            int *prev2 = bufs[(k + 1) % 3]; /* diagonal k-2 */

            #pragma omp for schedule(static)
            for (int i = imin; i <= imax; i++) {
                int j = k - i;
                int val;
                if (i == 0) {
                    val = j * GAP;
                } else if (j == 0) {
                    val = i * GAP;
                } else {
                    int diagv = prev2[i - 1] +
                                (a[i - 1] == br[n - j] ? MATCH : MISMATCH);
                    int upv   = prev1[i - 1] + GAP;
                    int leftv = prev1[i]     + GAP;
                    int best = diagv;
                    if (upv   > best) best = upv;
                    if (leftv > best) best = leftv;
                    val = best;
                }
                cur[i] = val;
            }
        }
    }

    int answer = bufs[total % 3][n];

    free(buf0); free(buf1); free(buf2); free(br);
    return answer;
}
```

PREDICTION: 8

# MEASUREMENT

No execution/benchmark tools were available in this session (explicitly noted: "No tools are available in this session... the pipeline will compile and measure it"), so I could not run `alignment_bench` myself. Reasoning-based expectation used for the prediction above:

- The reference solution allocates and randomly-strides through an (n+1)×(n+1) `int` matrix — for n in the low thousands this is tens of MB, far exceeding L2/L3, so it is memory-bandwidth/latency bound.
- The anti-diagonal version touches only 3 buffers of size n+1 (fits comfortably in L1/L2 for n up to tens of thousands), turns the inner loop into two contiguous streaming reads (`a`, `br`) that the compiler can auto-vectorize under `-O3 -march=native`, and adds OpenMP parallelism across independent diagonal cells for n>2000.
- Expected net effect: cache-locality alone typically buys 3-5x on this kind of DP; vectorization of the compare/add/max chain another ~2x; multi-core scaling on large n a further multiplier limited by the triangular (ramp-up/ramp-down) shape of anti-diagonals. Combined, roughly an order of magnitude (~8x) for moderately large n, less for very small n (where the fixed cost of 4 mallocs and the `if(use_par)`-gated parallel region dominates) — the code intentionally keeps `n>2000` as the parallel threshold to avoid regressing on small inputs.

I could not empirically confirm this; it should be validated against actual `alignment_bench` output.

# VERDICT

Implementation is exact (identical recurrence and base cases to the reference `kernel`, just reordered via anti-diagonal dependency order — no banding, no approximation, so it is correct for all inputs including adversarial shifted/rotated sequences where gaps matter). Performance improvement is predicted but unverified in this session; awaiting actual measurement from the pipeline to confirm or revise the ~8x estimate.