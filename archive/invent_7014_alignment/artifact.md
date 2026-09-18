## MAPPING

| World object | Problem object | Assumption it breaks |
|---|---|---|
| **Two gut-strings held taut** (SEED 2) | Sequences `a`, `b` as two live 1‑D fronts, not a stored 2‑D grid | "the whole grid of every position against every other must be filled in" — only positions actually paired under the current correspondence are touched |
| **Fingers walking down both strings at once, bead against bead** (SEED 2) | Simultaneous traversal where one index increases while the paired index decreases (one "bank" per string), meeting in a shared middle | "every cell depends on the ones above, to the left, and diagonally above-left, computed in that order" — cells that meet at the same instant have *no* dependency on each other |
| **"one bank for each string ... meeting in the shallows"** (SEED 2) | The widest anti-diagonals (k≈n) are entirely interior (no boundary correction needed) — literally where the two "banks" (i=0 edge, j=0 edge) have receded out of range | Same as above; also breaks strict row-major traversal order |
| **A bight pinched into the string, skipping one bead** (SEED 1) | A gap-step (`up`/`left` move) tried as a local alternative to the straight diagonal move at a given point | "a slip (gap) can only be discovered by having already compared the position before it" — here the gap is one of three simultaneously-available local hypotheses, not something built up incrementally |
| **Every bight-point tried, only the tallest stone-pile kept, the rest rot** (SEED 3) | The `max(diag, up+GAP, left+GAP)` taken at every cell; rejected candidates are simply discarded, never touched again | "every cell... computed in that order" — trials/candidates are independent and comparison-only, not chained |

## CHOSEN SEED

**SEED 2** — "Fingers walk two taut strings bead against bead... one bank for each string, one loop of stones gathering in the shallows where the two banks meet."

This is the most literal *and* most structurally different mapping: it translates directly into **anti-diagonal (wavefront) traversal** of the DP table — index `i` (bank A) increases while `j = k-i` (bank B) decreases along a shared "river-time" index `k = i+j`. Cells on one anti-diagonal never depend on each other, only on the previous two anti-diagonals — this is exactly the reverse of the known row-major/banded-SIMD approach, which keeps a left-to-right dependency chain *inside* every row that blocks vectorization.

## ASSUMPTION BROKEN

"Every cell of the comparison depends on the ones above, to the left, and diagonally above-left, **computed in that order**." Re-walking by anti-diagonal instead of by row removes the intra-sweep serial dependency: all cells on a given wavefront are mutually independent (each needs only the two *previous, already-finished* wavefronts), so they can be vectorized/threaded freely — something the row-major recurrence structurally forbids for its inner loop.

## ARTIFACT

```c
#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *prev2 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diagonal k-2 */
    int *prev1 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diagonal k-1 */
    int *cur   = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diagonal k   */

    /* k = 0 : cell (0,0) */
    prev2[0] = 0;

    /* k = 1 : cells (0,1) and (1,0) */
    prev1[0] = GAP;
    prev1[1] = GAP;

    for (int k = 2; k <= 2 * n; k++) {
        int ilo = (k - n > 0) ? (k - n) : 0;
        int ihi = (k < n) ? k : n;

        if (ilo == 0) {                 /* boundary cell i=0, j=k : one "bank" edge */
            cur[0] = k * GAP;
            ilo = 1;
        }
        if (ihi == k) {                 /* boundary cell i=k, j=0 : the other "bank" edge */
            cur[k] = k * GAP;
            ihi = k - 1;
        }

        /* the "shallows": interior of the wavefront, fully independent cells */
        #pragma omp parallel for simd schedule(static) if(ihi - ilo > 256)
        for (int i = ilo; i <= ihi; i++) {
            int j = k - i;
            int diag = prev2[i - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int up   = prev1[i - 1] + GAP;
            int left = prev1[i]     + GAP;
            int best = diag;
            if (up   > best) best = up;
            if (left > best) best = left;
            cur[i] = best;
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

Objects, made literal:
- **Memory**: not the O(n²) grid at all — only two/three rolling O(n) arrays ("two gut-strings" held taut at any instant), matching the native's image that only two strings are ever in hand, never a whole net of them.
- **Flows**: the "river that is time" is the anti-diagonal index `k`; buffers roll downstream (`prev2←prev1←cur`) each tick, carrying stones forward.
- **Stays still**: the raw sequences `a`, `b` — read-only, threaded fixed positions.
- **Processor**: SIMD lanes (and, on wide wavefronts only, OpenMP threads) — the "fingers" walking many bead-pairs at once, since cells within one anti-diagonal are provably independent.
- **The bight/best-trial**: the local `max(diag, up+GAP, left+GAP)` — every cell already *is* a bight-point trial, decided instantly rather than searched globally.
- **Boundary vs. shallows**: the two boundary corners (`i=0`, `j=0`) are peeled off explicitly so the interior loop is branch-free and vectorizes cleanly — literally the "shallows where the two banks meet," widest and least encumbered at `k≈n`.

This computes the *exact* standard Needleman-Wunsch recurrence, merely reordered — so it is bit-for-bit identical to the reference DP.

PREDICTION: speedup_vs_dp = 4.0

## MEASUREMENT

Not run: `alignment_bench` and `alignment_contract` are not available as tools in this session (the environment explicitly disabled tool calls here). I have not fabricated a number for this section — the prediction above is a reasoned estimate only, based on: (1) removing the intra-row dependency chain that blocks auto-vectorization of the reference kernel's inner loop, (2) O(n) working set instead of O(n²), which keeps the rolling buffers cache-resident, and (3) OpenMP parallelism gated to only the wide, fully-interior wavefronts (`ihi-ilo>256`) to avoid fork/join overhead dominating on narrow anti-diagonals or small `n`. The actual number needs the pipeline's real measurement to confirm or refute.

## VERDICT

Correctness: by construction this is the same DP recurrence over the same dependency graph, just swept by anti-diagonal instead of by row, so it should match the reference exactly for every input — this is not a heuristic approximation of the native's single-gap image, but the literal "two-banks-meeting" walk applied to the *full* recurrence (every cell still gets its local bight-trial via the 3-way max), so exactness is preserved. Speed is an honest prediction (4.0×), not a measured fact — the pipeline's own run of `alignment_bench` is needed to say whether the anti-diagonal reordering actually pays for its boundary-splitting and fork/join overhead versus the reference's already cache-friendly row-major loop, and I have not overstated that as measured when it wasn't.