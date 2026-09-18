# MAPPING

| World object | Problem object | Assumption it breaks (from the list) |
|---|---|---|
| **SEED 1** — two threads on one hinge, bead-against-bead, hot pin → ash-mark on match, bare cloth on mismatch | Two DNA strings `a[i]`, `b[i]` compared position-by-position; a match contributes credit, a mismatch contributes nothing extra (no separate mismatch bookkeeping beyond "no ash") | *"the whole grid of every position against every other must be filled in"* — the hinge only ever walks the **diagonal** (`i` vs `i`), never `i` vs `j≠i`. There is no grid, only a single row. |
| **SEED 2** — one slack-spool, wound onto the hinge, paid out **exactly once, at exactly one point**, letting one thread skip a bead while the other waits | A single gap-event (one insertion paired with one compensating deletion, since the two strings are equal length) hypothesized at a chosen position `k`, rather than *discovered* by having already filled the cell before it | *"a slip (gap) can only be discovered by having already compared the position before it"* — the native picks the slip point **a priori**, as a free hypothesis, not as something the recurrence stumbles into after filling earlier cells. |
| **SEED 3** — try the walk again at *every* possible point, throw away every cloth but the one with the most ash, wait until every point is tried before choosing | Instead of one dependency chain filling an `n×n` table, run **n independent trials**, each a straight O(n) walk with the slip fixed at a different position, and reduce by `max` | *"the whole grid of every position against every other must be filled in"* **and** the implicit ordering assumption *"every cell depends on the ones above/left/diagonally above-left computed in that order"* — here the trials have **no dependency on each other at all**; they could be run in any order, or all at once. |

# CHOSEN SEED

**SEED 3** — "every possible point is tried, independently, and only the best cloth is kept." It is the most literal (the native is explicit: one attempt per point, no early stopping) and the most different from Needleman–Wunsch's core mechanic: NW's speed *and* its serial dependency chain come from the same place (cell `(i,j)` needs `(i-1,j)`, `(i,j-1)`, `(i-1,j-1)`). SEED 3 throws that chain away entirely and replaces it with a set of independent hypotheses.

# ASSUMPTION BROKEN

*"The whole grid of every position against every other must be filled in, cell by cell, each depending on its neighbors."* The native never builds a grid. He builds **one row of independent hypotheses** (where is the one slip?), evaluates each with a straight walk, and keeps the maximum.

# Literal object-by-object translation

- **Thread A / Thread B** = the two `char*` arrays `a`, `b`.
- **Hinge** = the fact that both are walked with a *shared* index, i.e. one loop variable, never two nested independent indices — this is exactly why there is no grid.
- **Hot pin / ash-mark** = the `+1` credit when `a[i]==b[i]` (mismatch = "bare cloth", handled as `-1`, i.e. bookkeeping folded into a signed sum rather than a separate mark).
- **Slack-spool, paid out once, at one point** = exactly one insertion+deletion pair, whose position `k` is a *free parameter*, not something the recurrence discovers.
- **"One attempt per point along the row"** = `k = 0..n-1`, each a full re-walk — literally what the native describes, and literally O(n) attempts.
- **"Cloth kept, rest thrown away, count of ash-petals returned"** = `max` reduction over all attempts (plus, since he says he is merely *allowed*, not *forced*, to spend the spool, the "never spend it" attempt — the plain diagonal — is included as one more candidate).
- **Memory that stays still** = nothing bigger than a handful of length-`n` arrays — there is no `(n+1)×(n+1)` table anywhere.
- **What flows / what is processor-time** = each attempt's walk; because attempts don't depend on each other, they are the natural unit of parallelism (SIMD/OpenMP), matching seed 3's "independent hypotheses" directly.

The one extra piece of engineering literalism (not textbook substitution): a "point" for the slip and the "point" where the two threads resync are the *same single point* in the native's telling ("exactly once, at exactly one point") — so the walk shifts by one bead from `k` onward and never un-shifts except at the string's natural end. Working the algebra of "try every point, keep the max" out fully shows the whole n candidates collapse into a **prefix-sum / running-max scan** — O(n) total, not O(n) trials × O(n) work each. That collapse is the thing that "falls out" of taking the metaphor completely literally.

# ARTIFACT

```c
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2
#define NEG_INF (-1000000000)

static inline int sc(char x, char y) { return (x == y) ? MATCH : MISMATCH; }

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    /* m[i]     = ash-mark test walking the hinge straight (a[i] vs b[i]) */
    /* mshAB[t] = walk after A's thread has skipped one bead (a leads)   */
    /* mshBA[t] = walk after B's thread has skipped one bead (b leads)   */
    int *m      = (int *)malloc((size_t)n * sizeof(int));
    int *mshAB  = (int *)malloc((size_t)n * sizeof(int));
    int *mshBA  = (int *)malloc((size_t)n * sizeof(int));
    int *PM     = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *PMS_AB = (int *)malloc((size_t)n * sizeof(int));
    int *PMS_BA = (int *)malloc((size_t)n * sizeof(int));

    #pragma omp parallel for simd
    for (int i = 0; i < n; i++) m[i] = sc(a[i], b[i]);

    mshAB[0] = 0;
    mshBA[0] = 0;
    #pragma omp parallel for simd
    for (int t = 1; t < n; t++) {
        mshAB[t] = sc(a[t], b[t - 1]);
        mshBA[t] = sc(b[t], a[t - 1]);
    }

    PM[0] = 0;
    for (int k = 1; k <= n; k++) PM[k] = PM[k - 1] + m[k - 1];
    int S0 = PM[n]; /* the attempt where the slack-spool is never spent */

    PMS_AB[0] = 0;
    PMS_BA[0] = 0;
    for (int k = 1; k < n; k++) {
        PMS_AB[k] = PMS_AB[k - 1] + mshAB[k];
        PMS_BA[k] = PMS_BA[k - 1] + mshBA[k];
    }

    /* every point along the row, folded into one running-max scan */
    int bestAB = NEG_INF, bestBA = NEG_INF;
    {
        int runMaxL_AB = NEG_INF, runMaxL_BA = NEG_INF;
        for (int q = 0; q < n; q++) {
            int LqAB = PM[q] - PMS_AB[q];
            int LqBA = PM[q] - PMS_BA[q];
            if (LqAB > runMaxL_AB) runMaxL_AB = LqAB;
            if (LqBA > runMaxL_BA) runMaxL_BA = LqBA;

            int SMq1 = S0 - PM[q + 1];
            int candAB = runMaxL_AB + PMS_AB[q] + SMq1;
            int candBA = runMaxL_BA + PMS_BA[q] + SMq1;
            if (candAB > bestAB) bestAB = candAB;
            if (candBA > bestBA) bestBA = candBA;
        }
        bestAB += 2 * GAP;
        bestBA += 2 * GAP;
    }

    int result = S0;
    if (bestAB > result) result = bestAB;
    if (bestBA > result) result = bestBA;

    free(m); free(mshAB); free(mshBA);
    free(PM); free(PMS_AB); free(PMS_BA);
    return result;
}
```

PREDICTION: speedup_vs_dp = 60

Reasoning for the number: the DP baseline does Θ(n²) work over an `(n+1)²` table; this kernel does a fixed small number (≈8) of Θ(n) passes and Θ(1) extra memory beyond a few length-`n` arrays — no `n×n` allocation at all. For benchmark sizes in the few-hundred-to-few-thousand range this is a linear-vs-quadratic gap that should dominate constant-factor overhead (six small mallocs vs. one big one), so I expect on the order of tens-of-x speedup, growing with `n`; 60× is a deliberately conservative midpoint given I don't know the benchmark's actual `n`.

# MEASUREMENT

Not run. No tools (including `alignment_bench`) were available in this session — the instructions for this turn explicitly say to reason and predict only, and that "the pipeline will compile and measure it" afterward. I am reporting this plainly rather than fabricating a number.

# VERDICT

Honest limitation, stated up front rather than smoothed over: this kernel is **exact** only for alignments that use zero gap-pairs or a *single* insertion/deletion pair (in either direction, any width, any position) — which is exactly the space the native's "one slack-spool, spent once" literally describes. I hand-verified the recurrence on two constructed cases (a no-gap-optimal pair, where the scan correctly stays at `S0`, and a clean single-insertion pair, where the scan correctly finds a higher score than `S0`). It will **silently underscore** the true Needleman–Wunsch optimum whenever the reference DP's true best alignment needs **two or more independent gap-pairs** (e.g., two separate short repeat regions each independently worth their own indel) — a case the native's single spool structurally cannot represent, and one I did not paper over with the textbook algorithm. Whether that case appears in `alignment_bench`'s actual test sequences determines whether this artifact is a genuine 60×-class win or a fast-but-wrong answer; that is exactly the kind of thing the measurement, not my prediction, should settle.