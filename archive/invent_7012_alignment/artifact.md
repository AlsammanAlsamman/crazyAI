# MAPPING

**SEED 1 — "single crawler sweeps the whole grid row by row in one unbroken trail"**

| World object | Problem object |
|---|---|
| Two brass lemur-copies pinned along south/east doors, one letter per peg | Strings `a`, `b`, fixed read-only arrays, one character per index |
| The grid where their reach overlaps | The (n+1)×(n+1) DP score space |
| Single crawling body, one cell wide, cut from the seaweed creature's waist | One sequential cursor visiting one cell `(i,j)` at a time — no second body, no parallel visitor |
| "I do not walk it letter against letter... keeping a fresh tally each time" | Rejection of recomputing each pair from scratch; scores accumulate along the trail (this is DP itself, not brute force) |
| Ox-plow sweep, curling at the wall, row then next row | Row-major traversal, wrapping from the end of row *i* to the start of row *i+1* |
| "Every cell it enters already carries the pigment-drop left by the cells behind and above it" | Standard dependency: `dp[i][j]` needs `dp[i][j-1]` (behind) and `dp[i-1][j]` (above) already settled |
| "I wait always for the drop to fully settle before letting the body move on" | Strict sequential data dependency — no speculative/out-of-order cell computation |
| Storms wash the abandoned, fainter drops into the mountains, unremembered | Old rows are **not kept** — only the minimum state needed (previous row + current row) is retained; everything else is discarded, i.e. O(n) memory instead of O(n²) |
| Read the deepest pigment at the last peg | `dp[n][n]` is the returned score |

Broken assumption: this seed does **not** break "one pair of positions is judged at a time" — a single crawler visiting one cell per step is, if anything, a *more* literal enactment of exactly that assumption. It does erode the *memory* assumption implicit in the reference (`malloc((n+1)*(n+1))`): the "storms wash it away, unremembered" clause is a direct instruction to *not* keep the whole grid.

**SEED 2 — "eight compass-headings + curl-back mark agreement vs. slipped letter within one motion"**

| World object | Problem object |
|---|---|
| Eight compass headings | The (over-)generous direction set nominally available to the crawler |
| Curl-back over its own tail | A self/zero-length move, unused by plain NW |
| Diagonal step = agreement | `dp[i-1][j-1] + match/mismatch` |
| Sideways step = slipped letter, "paid within the same continuous motion" | `dp[i-1][j] + GAP` / `dp[i][j-1] + GAP`, computed inline rather than as a separate branch |

This is a re-skin of the standard max-of-three recurrence; it doesn't break "one pair at a time" either (still one cell, one set of three candidates, per step).

**SEED 3 — "trunk regret apparatus keeps only the deepest drop"**

| World object | Problem object |
|---|---|
| Trunk regret apparatus | The `max()` operator |
| Arriving pigment-drops | The 2–3 candidate incoming scores at a cell |
| Keep deepest, let fainter wash away | `dp[i][j] = max(diag, up, left)`, discarding the losers |
| "The worm remembers no other choice once made" | No backtracking over discarded alternatives — standard optimal-substructure DP |

Again just the ordinary max-selection step; doesn't touch "one pair at a time" either.

# CHOSEN SEED

None of the three seeds breaks "one pair of positions is judged at a time" — every one of them, read literally, describes a *single* body occupying *one* cell at a time, waiting for that cell's value to "settle" before advancing. Saying otherwise would be dishonest to the text. So, per the fallback rule, I take the seed whose mapping is most literal and most different from the given O(n²)-table reference: **SEED 1**, because of its explicit, concrete, unambiguous instruction that most of the pigment ("everything else") is not kept — only what the crawler is currently carrying.

# ASSUMPTION BROKEN

Not "one pair at a time" (none of the three seeds break that — stated plainly, as instructed). SEED 1 instead breaks the *unstated* memory habit baked into the reference implementation: "the whole grid of every position against every other must be filled in" is honored (every cell is still visited), but it need not all be *stored* — "storms of visitors banking past wash back into the mountains of pigment, unremembered" is a direct license to discard everything except the previous row and the row currently being laid down. That's an O(n²) → O(n) memory-footprint change, not an algorithmic-complexity change.

# ARTIFACT

```c
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    /* Single "crawler": one row-wide body that lays one unbroken trail
       across the grid, carrying forward only the pigment (scores) it
       needs -- the row above (prev) and the row it is currently laying
       down (curr) -- and lets everything else wash away, unremembered. */
    size_t w = (size_t)n + 1;
    int *prev = (int *)malloc(w * sizeof(int));
    int *curr = (int *)malloc(w * sizeof(int));

    /* Row 0: rest-state at the first peg -- j gap steps from the origin. */
    for (size_t j = 0; j < w; j++) prev[j] = (int)j * GAP;

    for (int i = 1; i <= n; i++) {
        curr[0] = i * GAP;               /* column-0 boundary for this row */
        int diag = prev[0];              /* dp[i-1][0], carried as the crawler advances */
        char ai = a[i - 1];
        for (int j = 1; j <= n; j++) {
            int up_val   = prev[j];      /* the drop left by the cell above */
            int left_val = curr[j - 1];  /* the drop left by the cell behind */
            int cross    = diag + (ai == b[j - 1] ? MATCH : MISMATCH); /* diagonal: agreement/slip-in-letter */
            int side_up   = up_val   + GAP; /* orthogonal step: slipped letter, paid inline */
            int side_left = left_val + GAP; /* orthogonal step: slipped letter, paid inline */

            int best = cross;
            if (side_up   > best) best = side_up;
            if (side_left > best) best = side_left;

            diag = up_val;   /* the cell above becomes the new diagonal predecessor
                                 once the crawler steps one cell further along the row */
            curr[j] = best;  /* the drop settles here before the body moves on */
        }
        /* Curl at the wall: swap the just-laid trail into "prev" and
           reuse the old prev's memory for the next row. */
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

Correctness check by hand: this is the textbook O(n) linear-space rewrite of the given O(n²) reference — same recurrence, same order of evaluation, same boundary rows/column, only the storage is shrunk. For `n=0` the loop body never runs and `result = prev[0] = 0`, matching the reference. For any `n`, `dp[i][j]` is computed from `diag=dp[i-1][j-1]`, `up_val=prev[j]=dp[i-1][j]`, `left_val=curr[j-1]=dp[i][j-1]` — identical dependencies to the reference, just not materialized as a full 2-D array.

I deliberately did **not** add OpenMP or row-parallel SIMD: the literal seed insists on *one* body laying *one* unbroken trail, and the real recurrence has a genuine serial dependency along each row (`curr[j]` needs `curr[j-1]`) that a single-cell-wide crawler cannot legally skip without becoming a different (non-matching) computation. Faking parallelism here would be quietly swapping in the textbook SIMD/banded method, which the instructions forbid.

PREDICTION: speedup_vs_dp = 4.0

Reasoning for the number: the reference allocates and touches `(n+1)²` ints — for n in the few-hundred-to-few-thousand range that's megabytes to tens of megabytes, overflowing L1/L2 cache and turning the kernel memory-bandwidth-bound with poor reuse. The rolling-row version touches only `2(n+1)` ints, comfortably resident in L1 for all realistic benchmark sizes, at the same O(n²) arithmetic work. I expect the win to come almost entirely from cache locality and eliminated large-malloc/page-fault overhead, not from fewer operations — hence a solid constant-factor speedup (my estimate: ~4×) rather than an order-of-magnitude one, and I'd expect it to grow with n and shrink toward ~1× for very small n where the O(n²) table already fits in cache.

# MEASUREMENT

No tools were available in this session (explicitly stated in the task), so `alignment_bench` could not be invoked here — I cannot report an empirical number I actually measured. The prediction above is submitted for the external pipeline to compile and measure against the reference; I have not silently substituted a fabricated measurement for a real one.

# VERDICT

Pending the pipeline's run. Honestly: this is a conservative, correctness-first, literal translation (O(n) memory, same O(n²) work, no parallel crawlers introduced against the seed's explicit "single body" constraint), so I expect a real but modest constant-factor speedup rather than an asymptotic one — if the reference's dominant cost isn't actually the memory footprint (e.g. if the benchmark harness already reuses/warms the allocation, or n is small enough that the O(n²) table fits in cache), the true speedup could land well below the predicted 4.0, and that should be reported as such rather than reframed as success.