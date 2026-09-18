# MAPPING

| Seed | World object | Problem object | Assumption it breaks |
|---|---|---|---|
| **1** — hinge-calipers, one leg free to pause | two rows of shard-letters | arrays `a[0..n-1]`, `b[0..n-1]` | "every cell depends on above/left/diag-above-left, **computed in that order**" |
| | the two caliper legs walking together | the DP indices `(i,j)` advancing in lock-step | |
| | one leg pausing at a joint | an "up" or "left" (gap) move, available at *every* joint | |
| | "the string slips a half-step out of the other's shadow" | offset `d = i-j` becoming non-zero for a stretch | |
| **2** — pond that looks the same before and after, sound not ripple | the pond | the accumulator / working memory | "the whole grid ... must be filled in" (i.e. must be *stored*) |
| | pond staying "exactly as still" | fixed-size storage reused every stride, never growing | |
| | "sound, once heard, stays counted" (not a ripple that settles) | a deterministic, order-independent tally rather than a value threaded stateful through the whole table | |
| **3** — every place for the pause is tried, cheapest walk kept, rest thrown into the sand | trying every door | evaluating diag/up/left at a cell | "one pair of positions is judged at a time" |
| | keeping only the cheapest walk, discarding the rest | local max taken, losers discarded (not stored) | |
| | trying every door **at once**, over a whole "market day" | many candidate cells judged **simultaneously** | |

# CHOSEN SEED

Seed 1 — the hinge-calipers with one pausable leg. It is the most literal (a caliper *is* a pair of pointers; a pausing leg *is* a gap-move) and it is the seed that forces a genuinely different computation order, not just a different implementation detail. Seeds 2 and 3 fall out for free once seed 1 is taken literally: a real caliper doesn't remember the whole grid behind it (seed 2 → O(n) rolling storage instead of O(n²)), and many caliper-pairs can walk the *same* stride side by side (seed 3 → all cells of a stride judged at once, in parallel).

# ASSUMPTION BROKEN

"Every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed in that order" — specifically the *row-major* order. The reference kernel's inner loop over `j` has a true loop-carried dependency (`dp[i][j]` needs `dp[i][j-1]`, same row) that forbids vectorization. A caliper with two legs stepping together naturally walks **anti-diagonals** (`s=i+j`): every cell on one anti-diagonal depends only on the *previous two* anti-diagonals, never on a neighbor within the same one. So all cells of a stride are mutually independent — they can be judged at once (SIMD lanes / OpenMP threads), and the pond never needs to hold more than the current and previous two strides (O(n) memory instead of O(n²)). The recurrence itself (diag/up/left, max) is untouched, so the result is mathematically identical to the reference — only the *order and locality* of computation changes.

# ARTIFACT

```c
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    /* reversed b so that, along an anti-diagonal, both a[] and the
       relevant b-index move forward together -> contiguous access */
    char *brev = (char *)malloc((size_t)n * sizeof(char));
    for (int k = 0; k < n; k++) brev[k] = b[n - 1 - k];

    /* the "pond": three fixed, same-size buffers, reused every stride
       (never a full n x n grid) */
    int *buf0 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf1 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf2 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *PD2 = buf0; /* anti-diagonal s-2 */
    int *PD1 = buf1; /* anti-diagonal s-1 */
    int *CUR = buf2; /* anti-diagonal s   */

    int total = 2 * n;

    #pragma omp parallel default(shared)
    {
        for (int s = 0; s <= total; s++) {
            int istart = (s - n) > 0 ? (s - n) : 0;
            int iend   = (s < n) ? s : n;
            int lo = (istart > 1) ? istart : 1;
            int hi = (iend < s - 1) ? iend : (s - 1);

            if (lo <= hi) {
                /* the many caliper-pairs on this stride, judged at once */
                #pragma omp for schedule(static)
                for (int i = lo; i <= hi; i++) {
                    int j = s - i;
                    int m = n - s + i; /* brev index for b[j-1] */
                    int sc   = (a[i - 1] == brev[m]) ? MATCH : MISMATCH;
                    int diag = PD2[i - 1] + sc;
                    int up   = PD1[i - 1] + GAP;
                    int left = PD1[i]     + GAP;
                    int best = diag;
                    if (up   > best) best = up;
                    if (left > best) best = left;
                    CUR[i] = best;
                }
            }

            #pragma omp single
            {
                if (s <= n) {
                    if (s == 0) {
                        CUR[0] = 0;
                    } else {
                        CUR[0] = PD1[0] + GAP;     /* leg-0 paused: i=0 */
                        CUR[s] = PD1[s - 1] + GAP; /* leg-1 paused: j=0 */
                    }
                }
                /* the calipers close: rotate the still pond forward */
                int *tmp = PD2;
                PD2 = PD1;
                PD1 = CUR;
                CUR = tmp;
            }
        }
    }

    int result = PD1[n];

    free(brev);
    free(buf0);
    free(buf1);
    free(buf2);
    return result;
}
```

# PREDICTION: speedup_vs_dp = 5

Reasoning behind the number: the reference's inner `j`-loop has a true scalar dependency chain (`dp[i][j-1]`), which blocks SIMD entirely and forces O(n²) traffic through a heap array that stops fitting in cache once n is a few hundred. The anti-diagonal form removes that dependency (all cells on a stride are independent → auto-vectorizable, contiguous thanks to `brev`) and shrinks live memory from O(n²) to O(n), which should dominate for larger n. Working against it: an OpenMP barrier every one of the `2n+1` strides, which is real, non-trivial overhead (worst near the short strides at the corners) that a plain row-major scan doesn't pay. Netting a large per-cell win against real synchronization overhead, I expect a solid but not spectacular win — around 5×, not an order of magnitude more.

# MEASUREMENT

Not run in this session — no tool access was available to me here (explicitly stated in the task setup: "No tools are available in this session... the pipeline will compile and measure it"). I have hand-verified the recurrence and index bookkeeping by tracing it completely by hand for n=1 (base cell, both edge cells, and the interior diag/up/left cell all exercised and matched the reference's expected `dp` values), but I have not compiled or benchmarked it, so I am not reporting a measured number — only the prediction above, honestly labeled as unmeasured.

# VERDICT

Pending actual measurement by the pipeline. What I can say without measuring: this is not a shortcut on correctness — it computes the exact same recurrence as the reference (verified by hand-trace, not an approximation like a single-gap/banded heuristic would be), so if it fails, the likely failure mode is a bookkeeping bug in the anti-diagonal index math rather than a fundamental algorithmic mismatch, and if it succeeds, the source of any measured speedup should trace cleanly back to the two things the caliper literally does differently from row-major DP: it never holds the whole grid (O(n) vs O(n²) memory) and it judges a whole stride of independent cells at once instead of one pair at a time.