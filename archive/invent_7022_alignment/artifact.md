# MAPPING

**SEED 1 — "Two knot-cords laid crosswise… mark every place their symbols meet."**

| World object | Problem object |
|---|---|
| cord running north, "the horse's length" | string `a`, length `n` (row axis) |
| cord running east, "the mountain's width" | string `b`, length `n` (column axis) |
| lattice of furrow-points | the `(n+1)×(n+1)` grid of DP cells `(i,j)` |
| a furrow-point where symbol from cord A faces symbol from cord B | comparison `a[i-1]` vs `b[j-1]` |

Breaks: none of the five listed assumptions — this seed literally *re-states* "the whole grid… must be filled." It sets the stage rather than changing the mechanism.

**SEED 2 — "A single ink-worm walks the eight desert winds and curls back across its own trail to test a junction twice."**

| World object | Problem object |
|---|---|
| ink-worm, one cell wide | the single active DP-cell being resolved right now |
| eight desert winds | the (at most three relevant) transition directions: N/S (up), E/W (left), NE/SW-diagonal |
| dropping a mound to the height its path earned | writing a candidate score into a cell |
| reading the mound already there, keeping "the beaten mound... entirely" discarded | overwrite semantics — no separate ledger beyond the DP value array itself (→ rolling, not full, memory) |
| curling back to try a junction "from another face" when its own tally doesn't yet beat the standing mound | revisiting/relaxing a cell out of the fixed up→left→diagonal order |
| "done only when… no junction anywhere still holds a mound it could improve by returning" | a relaxation-style (Bellman-Ford/SPFA) convergence criterion instead of one fixed visiting order |

Breaks: "every cell… computed in that (fixed) order" — the worm's value at a junction is not guaranteed correct on first visit; it is only guaranteed correct once nothing can improve it by another pass.

**SEED 3 — "The worm crosses a furrow sideways, spending no sand, to let one cord slip ahead of the other."**

| World object | Problem object |
|---|---|
| sideways crossing | the gap move (horizontal/vertical DP transition) |
| "the one permitted stumble" | the well-known fact that only one string may advance without the other on a single step |

Breaks: nothing new — the sideways move still requires the worm to have already stood on the adjacent junction, i.e. it still needs "the position before it" already settled. This seed *reaffirms* the gap-dependency assumption rather than breaking it.

**Check for "one pair of positions judged at a time":** none of the three seeds breaks this — the worm is explicitly *one cell wide*, singular, never "broken into separate glances." I state this plainly rather than force a batching reading onto the metaphor.

# CHOSEN SEED

SEED 2. It is the most literal (it names concrete mechanics — no separate ledger, revisit-to-improve, termination-by-no-more-improvement) and the most different from the textbook fixed-order sweep.

# ASSUMPTION BROKEN

"Every cell of the comparison depends on the ones above, to the left, and diagonally above-left, **computed in that order**." SEED 2 replaces the fixed row-major visiting order with a relaxation view: a junction's value is only final once no further pass can improve it.

# ARTIFACT

Taking the mechanism completely literally exposes something real, but also something to prune. The DP grid is a DAG with a natural topological order along **anti-diagonals** (`i+j = k`): every cell on diagonal `k` depends only on diagonals `k-1` and `k-2`. On a DAG, visiting nodes in topological order makes every "junction" correct the *first* time — so the worm's "curl back and retest" is provably never useful here (a cell on diagonal `k` can never be improved by a later diagonal). Per the task's own rule (arrive at the validated technique rather than keep the novel-but-useless part), I drop the revisit machinery and keep only the two literal details that *do* carry real weight and correspond to a known, validated technique — **anti-diagonal wavefront DP**, exactly the trick behind SIMD aligners like KSW2/parasail:

- *"keeps no ledger apart from the sand itself; a beaten mound is thrown away entirely"* → only **three rolling diagonals** (`O(n)` memory) are ever alive, not the full `O(n²)` matrix.
- *"one cell wide… never broken into separate glances, the whole crawling path is one continuous reading"* → the cells of one anti-diagonal are mutually independent, so that "single wide stitch" is handed to the compiler as one contiguous, branch-light vector loop (`restrict`, forward strides on both `a` and a **pre-reversed** `b` so the compiler doesn't need a backward gather).

Stated risk, guarded per rule 4: for small `n`, the setup cost of reversing `b`, three-buffer rotation, and diagonal boundary bookkeeping could plausibly cost more than it saves. Guarded with an `n < 64` fallback to a plain two-row row-major DP (no diagonal machinery at all).

Per rule 4 (vectorization before threads) and the metaphor's own insistence on a *single, unbroken* worm, I deliberately do **not** add OpenMP — splitting the worm across threads is exactly "breaking it into separate glances," which the native explicitly says never happens.

```c
#include <stdlib.h>

#define MATCH     1
#define MISM_MAG  1
#define GAP_MAG   2

/* Small-n fallback: plain row-major two-row DP. Cheap and branch-simple;
   used below the point where the diagonal/vector machinery's own setup
   cost (reversing b, three rolling buffers, boundary bookkeeping) could
   plausibly outweigh what it saves. */
static int kernel_small(int n, const char *restrict a, const char *restrict b) {
    int len = n + 1;
    int *restrict prev = (int *)malloc(sizeof(int) * (size_t)len);
    int *restrict cur  = (int *)malloc(sizeof(int) * (size_t)len);
    for (int j = 0; j <= n; j++) prev[j] = j * (-GAP_MAG);
    for (int i = 1; i <= n; i++) {
        cur[0] = i * (-GAP_MAG);
        for (int j = 1; j <= n; j++) {
            int diag = prev[j - 1] + ((a[i - 1] == b[j - 1]) ? MATCH : -MISM_MAG);
            int up   = prev[j] - GAP_MAG;
            int left = cur[j - 1] - GAP_MAG;
            int best = diag;
            if (up > best) best = up;
            if (left > best) best = left;
            cur[j] = best;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int result = prev[n];
    free(prev); free(cur);
    return result;
}

/* Anti-diagonal wavefront DP: the DP grid is a DAG whose natural
   topological order is the anti-diagonal k = i + j, so every "junction"
   the worm settles on is final the first time -- the metaphor's own
   "curl back and retest" is therefore provably never useful here and is
   dropped. What survives, taken literally:
     - "no ledger apart from the sand itself"  -> only 3 rolling
       diagonals (O(n) memory) instead of the full (n+1)x(n+1) tray.
     - "one continuous reading, never broken into separate glances" ->
       every cell of one anti-diagonal is independent of its neighbours
       on that same diagonal, so it is handed to the compiler as one
       contiguous, gather-free vector loop (b is pre-reversed so both
       string indices advance with the same +1 stride as i increases). */
int kernel(int n, const char *restrict a, const char *restrict b) {
    if (n <= 0) return 0;
    if (n < 64) return kernel_small(n, a, b);

    int len = n + 1;

    char *restrict brev = (char *)malloc((size_t)n);
    for (int m = 0; m < n; m++) brev[m] = b[n - 1 - m];

    int *restrict prev2 = (int *)malloc(sizeof(int) * (size_t)len); /* diagonal k-2 */
    int *restrict prev1 = (int *)malloc(sizeof(int) * (size_t)len); /* diagonal k-1 */
    int *restrict cur   = (int *)malloc(sizeof(int) * (size_t)len); /* diagonal k   */

    prev1[0] = 0; /* diagonal k=0: the single cell (0,0) */

    for (int k = 1; k <= 2 * n; k++) {
        int lo = (k - n > 0) ? (k - n) : 0;
        int hi = (k < n) ? k : n;
        int istart = lo, iend = hi;

        if (lo == 0) {                  /* i = 0 : top-row boundary, dp[0][k] */
            cur[0] = k * (-GAP_MAG);
            istart = 1;
        }
        if (k <= n) {                   /* i = k : left-column boundary, dp[k][0] */
            cur[k] = k * (-GAP_MAG);
            if (iend == k) iend = k - 1;
        }

        int off = n - k; /* brev[off + i] == b[k - i - 1], forward stride in i */
        for (int i = istart; i <= iend; i++) {
            int diagScore = prev2[i - 1] + ((a[i - 1] == brev[off + i]) ? MATCH : -MISM_MAG);
            int upScore   = prev1[i - 1] - GAP_MAG;   /* straight along a's grain  */
            int leftScore = prev1[i]     - GAP_MAG;   /* sideways: the one stumble */
            int best = diagScore;
            if (upScore   > best) best = upScore;
            if (leftScore > best) best = leftScore;
            cur[i] = best;
        }

        int *t = prev2;   /* "a beaten mound is thrown away entirely" -- rotate, don't keep */
        prev2 = prev1;
        prev1 = cur;
        cur = t;
    }

    int result = prev1[n];

    free(prev2);
    free(prev1);
    free(cur);
    free(brev);
    return result;
}
```

PREDICTION: speedup_vs_dp = 5.0

# MEASUREMENT

No tools were available in this session (per the environment note, `alignment_bench` could not be invoked here), so no empirical number was collected by me — this reflects prediction only. The reasoning behind the 5.0 estimate: dropping `O(n²)` memory to `O(n)` removes most cache/allocation traffic for the sizes where the reference starts to hurt, and the anti-diagonal inner loop (constant-stride loads on `a`/`brev`, compare-select, two more compare-selects for the max) is a clean auto-vectorization target for AVX2 (8-wide `int32`), so I expect roughly 4–8x on the compute-bound interior with some of that eaten by boundary bookkeeping and the `brev` construction pass; 5.0x is my point estimate. The pipeline's actual measurement should be taken as the honest number, not this one.

# VERDICT

Correctness was hand-traced against the reference recurrence on a worked example (n=2, "AC" vs "AG", boundary and interior cells) and matches exactly. The worm's central "curl back and retest a junction twice" idea does **not** survive contact with the actual structure of the problem — the DP grid is a DAG with a ready-made topological order (the anti-diagonal), so no junction can ever be improved by revisiting once diagonals are processed in order; keeping the revisit machinery would only have added redundant relaxation passes with zero benefit, which is exactly the kind of "own stated risk" rule 4 says must be dropped rather than shipped. What legitimately falls out of taking the metaphor literally — a single unbroken, one-cell-wide sweep with no separate ledger, i.e. anti-diagonal order + `O(n)` rolling memory + a vectorizer-friendly reversed second string — is not novel; it is the same technique behind SIMD-vectorized global/local aligners (KSW2, parasail-style diagonal SIMD), which is the right outcome per rule 4 rather than a reason to discard the exercise. Stated risk (small-`n` setup overhead) is guarded with an explicit `n < 64` fallback to a plain two-row DP, so the mechanism cannot regress below the reference on small inputs by more than that fallback's own (minimal) overhead. No thread parallelism was added, consistent with both the metaphor (a single, unbroken worm) and rule 4's "vectorize before threading, only if the work units are large enough" — here the natural unit of parallel work is a SIMD lane within one diagonal, not an independent thread task.