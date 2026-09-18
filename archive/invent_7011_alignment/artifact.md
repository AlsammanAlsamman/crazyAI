# MAPPING

**SEED 1** — "I hang two boughs of symbols from crossed rows of lamp-posts so their meeting makes a lattice of spark-cells."

| World object | Problem object |
|---|---|
| bough (each string of symbols) | sequence `a`, sequence `b` |
| row of lamp-posts (down) | row index `i = 0..n` (indexes `a`) |
| column of lamp-posts (across) | column index `j = 0..n` (indexes `b`) |
| lattice of empty cells | the `(n+1)×(n+1)` DP matrix |
| spark lit at a crossing | the DP value at cell `(i,j)`, computed on demand |
| spark's color chosen by the two symbols crossing | whether the incoming move is a match, mismatch, or gap |

Assumption broken: **none**. This is just the ordinary row-vs-column grid — literally the textbook Needleman-Wunsch matrix. It says nothing about *order* of fill, so the "above/left/diag, in that order" assumption survives untouched.

**SEED 2** — "fold the lattice-sheet along its slanting middle, corner to corner… cells whose row and column add to the same sum press flush… three creases pressed into one thickness."

| World object | Problem object |
|---|---|
| slanting middle fold, corner to corner | reindexing the grid by antidiagonal `d = i + j` |
| a crease | the set of all cells `{(i, d-i)}` for one fixed `d` |
| cells pressed flush against each other within a crease | cells on the same antidiagonal, computed **together**, not one after another |
| three creases layered on one thickness | crease `d` depends on creases `d-1` and `d-2` only |

Assumption broken: **"every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed in that order."** Along one crease, cells have **no dependency on each other at all** — they depend only on the two previous creases. So instead of a forced above→left→diag sequential order per cell, an entire antidiagonal can be produced at once (vectorized), and the traversal direction itself changes from row-major to diagonal-major.

**SEED 3** — "I read three stacked layers through each crease at once… I throw the bottom layer away… I keep only the two most recent creases."

| World object | Problem object |
|---|---|
| layer beneath (diagonal move) | the `d-2` antidiagonal buffer |
| layer beside (one slip, used for both up and left) | the `d-1` antidiagonal buffer |
| letting a spent crease "drift off" | discarding/reusing the `d-2` buffer once `d` is computed |
| keeping only two most recent creases | O(n) rolling memory instead of the O(n²) full matrix |

Assumption broken: this attacks a *different* silent assumption — "the whole grid ... must be filled in [and kept]" — it's about memory footprint/locality, not computation order.

# CHOSEN SEED

**SEED 2** (the fold), because it is the only one of the three that breaks the named order-of-dependency assumption, and it is a literal, direct translation (crease = antidiagonal, fold = reindex by `i+j`). SEED 3 is kept as the *memory realization* of the same fold (rolling buffers) — the native describes the fold and the discarding as one continuous act, so the artifact below implements SEED 2 as the algorithm and SEED 3 as its memory management, not as a competing choice.

# ASSUMPTION BROKEN

"Every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed in that order." Cells on one antidiagonal are mutually independent — they only read the two previous antidiagonals — so the fill order becomes diagonal-major with a fully data-parallel inner loop, instead of a strictly ordered per-cell above/left/diag chain.

# ARTIFACT

Literal computational mapping:
- **memory**: three `int` arrays of length `n+1` — `prev2` (crease `d-2`), `prev1` (crease `d-1`), `cur` (crease `d`), rotated (pointer-swapped) each step — this is SEED 3's "keep only two most recent creases, let the spent one drift off," realized as O(n) rolling storage instead of O(n²).
- **flows**: the computation sweeps forward across creases `d = 2 … 2n`; within a crease the loop over `i` has *no* loop-carried dependency, so it is the vectorizable "read three layers at once" step.
- **stays still**: the two input strings; also a once-built **reversed** copy of `b` (`rb[k] = b[n-1-k]`) — a genuinely literal consequence of the fold: since `j` decreases as `i` increases along a crease, indexing `b` forward is backward-striding, but indexing the reversed string forward is contiguous, giving two clean parallel streams for the SIMD compare.
- **processor**: SIMD lanes (`#pragma omp simd`, compiled with `-O3 -march=native`) process the independent cells of one crease simultaneously.
- **time**: still `O(n²)` cell evaluations, but laid out as `2n` vector-friendly sweeps of width up to `n`, over O(n) memory instead of O(n²).

```c
#include <stdlib.h>
#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *buf0 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf1 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf2 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    char *rb  = (char *)malloc((size_t)n);
    for (int k = 0; k < n; k++) rb[k] = b[n - 1 - k];

    int *prev2 = buf0; /* crease d-2 */
    int *prev1 = buf1; /* crease d-1 */
    int *cur   = buf2; /* crease d   */

    /* crease d = 0 */
    prev2[0] = 0;
    /* crease d = 1 */
    prev1[0] = GAP; /* (0,1) */
    prev1[1] = GAP; /* (1,0) */

    for (int d = 2; d <= 2 * n; d++) {
        int i_lo = d - n; if (i_lo < 0) i_lo = 0;
        int i_hi = d;     if (i_hi > n) i_hi = n;

        int left_boundary = (i_lo == 0);       /* i = 0, j = d  */
        int top_boundary  = (d <= n);           /* i = d, j = 0  */

        if (left_boundary) cur[0] = prev1[0] + GAP;

        int interior_lo = left_boundary ? 1 : i_lo;
        int interior_hi = top_boundary ? d - 1 : i_hi;
        if (interior_hi > n) interior_hi = n;

        #pragma omp simd
        for (int i = interior_lo; i <= interior_hi; i++) {
            char match = (a[i - 1] == rb[n - d + i]) ? MATCH : MISMATCH;
            int diagv = prev2[i - 1] + match;
            int upv   = prev1[i - 1] + GAP;
            int leftv = prev1[i]     + GAP;
            int best = diagv;
            if (upv   > best) best = upv;
            if (leftv > best) best = leftv;
            cur[i] = best;
        }

        if (top_boundary) cur[d] = prev1[d - 1] + GAP;

        int *tmp = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur = tmp;
    }

    int result = prev1[n];
    free(buf0); free(buf1); free(buf2); free(rb);
    return result;
}
```

PREDICTION: speedup_vs_dp = 6.0

# MEASUREMENT

No `alignment_bench` / `alignment_contract` tool call was available in this session (per the environment note, tools were not reachable here), so I cannot report an actual measured number — only the prediction above, honestly labeled as unmeasured. The pipeline that compiles and runs this kernel against the reference DP is expected to produce the real figure; if it is run, the observed speedup should replace the predicted 6.0 rather than being reconciled after the fact.

# VERDICT

Reasoning check: correctness was hand-verified for `n=0,1` and for the boundary logic at `d≤n` vs `d>n` (row/column base cases reduce correctly to `i*GAP`/`j*GAP`), and the reversed-`b` trick was derived algebraically (`b[j-1] = rb[n-d+i]`), not assumed — this is the one piece that is genuinely new relative to the "known way," since it falls directly out of taking the fold literally (opposite traversal directions of the two boughs along a crease) rather than out of textbook SIMD-alignment tricks (banding, striping, score clamping) mentioned as the known alternatives. The honest caveat: this session had no working measurement tool, so the claimed 6.0× is a prediction only, not a verified result, and should be treated as such until actually benchmarked.