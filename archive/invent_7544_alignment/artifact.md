# MAPPING

| World object | Problem object | Assumption it breaks |
|---|---|---|
| shell-beads, 4–5 fixed kinds, strung on bridge-rope | the A/C/G/T alphabet, strung along each DNA string | none — literal renaming only |
| two cords stretched side by side, walked from the near rail to the far rail | the two sequences `a`,`b`, both traversed start→end together | none — confirms, doesn't break, direction |
| pressing matched beads / chalking a mark | recording a match (+1) at the current step | none |
| swapping a chalk mark for an ink drop (costs more) | recording a mismatch (−1) instead of a match | none |
| cutting a bead free, dropping it to the sea-beast | a deletion (gap, −2) at that position | none |
| **paying out a loop of slack rope so one cord "walks ahead or behind its partner for a stretch"** | **an insertion/deletion run handled as one elongated action rather than being rediscovered one position at a time — literally, a whole *stretch* of position-pairs judged in a single move** | **"one pair of positions is judged at a time"** |
| errand-boy's running slat of chalk/ink, kept "cheapest-path only" | the DP running-score memo (optimal substructure) | none — this *is* the standard NW recurrence, stated narratively |
| final rail count, entered once in the cellar ledger | the returned integer score | none |

**Direction check (required first):** none of the three seeds break "both strings are read start to end in the same direction." Every version of the walk goes from the near rail to the far rail on *both* cords together, beads read in strung order. This must be stated plainly, not routed around — so per the fallback rule I pick the most literal seed instead, using "most different from the known way" as the tiebreaker among the literal candidates.

# CHOSEN SEED

*"paying out a loop of slack rope into one cord so it walks ahead or behind its partner for a stretch."*

It is the most literal of the three that also says something a plain DP loop doesn't already say: a slip is not discovered bead-by-bead, it is *paid out as a stretch* — a whole run handled as one action. (Seed 1 is literal but trivial — just names the alphabet. Seed 3 is literal but is simply the DP recurrence restated in different words, offering nothing new.)

# ASSUMPTION BROKEN

"One pair of positions is judged at a time." Read literally, a DP cell `(i,j)` on anti-diagonal `d=i+j` depends only on the two *previous* anti-diagonals, never on another cell of its own diagonal — so every cell on one diagonal is mutually independent and can be "paid out for a stretch," i.e. judged together in one SIMD instruction, instead of one at a time as plain row-major DP forces (its `left` dependency is intra-row, so the compiler can't vectorize it).

This is exactly the validated **anti-diagonal wavefront** technique used in real SIMD sequence-alignment kernels — per the instructions, I let the metaphor arrive at that known-good technique rather than inventing something untested. I also keep it at O(n) memory (three rolling diagonal arrays, mirroring the errand-boy's *single* slat, not an n×n ledger book) and default to vectorization, adding OpenMP threading only behind a size guard, with a scalar-fallback for small `n` per the stated risk (diagonal bookkeeping overhead not worth it when there isn't much of a "stretch" to pay out).

# ARTIFACT

```c
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2
#define SMALL_N 32          /* below this, skip diagonal restructuring: not enough work to amortize it */
#define PAR_THRESHOLD 4096  /* only thread-parallelize a diagonal with this many independent cells */

static int kernel_small(int n, const char *a, const char *b) {
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int diag = prev[j - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int up = prev[j] + GAP;
            int left = cur[j - 1] + GAP;
            int best = diag;
            if (up > best) best = up;
            if (left > best) best = left;
            cur[j] = best;
        }
        int *tmp = prev; prev = cur; cur = tmp;
    }
    int result = prev[n];
    free(prev); free(cur);
    return result;
}

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;
    if (n < SMALL_N) return kernel_small(n, a, b); /* guard: fallback for the risky small-n case */

    /* Anti-diagonal wavefront: every cell of diagonal d depends only on
       diagonals d-1 and d-2, so all cells of one diagonal are mutually
       independent -- a whole "stretch" is judged at once (SIMD, and
       threads once the stretch is big enough), instead of bead by bead. */

    char *brev = (char *)malloc((size_t)n * sizeof(char));
    for (int k = 0; k < n; k++) brev[k] = b[n - 1 - k];

    int *d0 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal d-2 */
    int *d1 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal d-1 */
    int *d2 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal d (being filled) */

    d0[0] = 0;     /* dp[0][0] */
    d1[0] = GAP;   /* dp[0][1] */
    d1[1] = GAP;   /* dp[1][0] */

    int total_diag = 2 * n;

    for (int d = 2; d <= total_diag; d++) {
        int i_lo = (d - n > 0) ? (d - n) : 0;
        int i_hi = (d < n) ? d : n;
        int lo = i_lo, hi = i_hi;

        if (lo == 0) { d2[0] = d * GAP; lo = 1; }               /* top row: dp[0][j] = j*GAP */
        int right_is_boundary = (i_hi == d && d <= n);
        if (right_is_boundary) { d2[hi] = hi * GAP; hi -= 1; }  /* left col: dp[i][0] = i*GAP */

        int off = n - d; /* brev[off + i] == b[j - 1] for j = d - i */
        const int * restrict rd0 = d0;
        const int * restrict rd1 = d1;
        int * restrict rd2 = d2;
        const char * restrict ra = a;
        const char * restrict rb = brev;
        int span = hi - lo + 1;

        if (span > 0) {
            #pragma omp parallel for simd schedule(static) if(span > PAR_THRESHOLD)
            for (int i = lo; i <= hi; i++) {
                int diagv = rd0[i - 1] + (ra[i - 1] == rb[off + i] ? MATCH : MISMATCH);
                int upv = rd1[i - 1] + GAP;
                int leftv = rd1[i] + GAP;
                int best = diagv;
                if (upv > best) best = upv;
                if (leftv > best) best = leftv;
                rd2[i] = best;
            }
        }

        int *tmp = d0; d0 = d1; d1 = d2; d2 = tmp;
    }

    int result = d1[n];
    free(d0); free(d1); free(d2); free(brev);
    return result;
}
```

PREDICTION: speedup_vs_dp = 6

# MEASUREMENT

Not performed. No tools (`alignment_bench`, `alignment_contract`) were available in this session — the system reminder explicitly states reasoning and the artifact must be produced without execution, for the pipeline to compile and measure afterward. I am not fabricating a number here: the prediction above is the only figure I can honestly stand behind before that run happens.

# VERDICT

Hand-verified the recurrence by substitution for `n=1` and `n=2` (both boundary cases and the interior case reduce exactly to the reference recurrence), so I have high confidence in correctness, but it is unverified by compilation/execution.

Mechanism: the "stretch of slack" seed literally names anti-diagonal wavefront restructuring — removing the intra-row `left` dependency so a whole diagonal's cells are independent and vectorizable, with memory collapsed from O(n²) to O(n) (the errand-boy's single slat, not a whole ledger). This converges on a real, validated technique (anti-diagonal SIMD DP), not a novel invention, per the instructions.

Named risk and how it's addressed: bookkeeping (boundary branches, three rolling arrays, reversed-`b` build) adds constant overhead per diagonal that isn't worth it when `n` is small — so for `n < 32` the kernel falls back to a plain O(n)-memory row-major DP (`kernel_small`), identical in structure to the reference recurrence, guaranteeing no regression there. Thread parallelism is similarly guarded: it only activates when a single diagonal has more than 4096 independent cells (`if(span > PAR_THRESHOLD)`), so at ordinary DNA-alignment benchmark sizes the kernel runs single-threaded and vectorized only, with threading reserved for genuinely large `n` where the diagonal is wide enough to amortize fork-join cost — never engaged blindly.