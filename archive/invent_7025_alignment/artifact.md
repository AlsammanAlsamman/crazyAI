## MAPPING

**SEED 1 — "fixed row / slipping row, horse visits every house once, and once nowhere"**

| World object | Problem object |
|---|---|
| two furrows, two rows of pawns | strings `a`, `b` |
| beast carved on a pawn's crown | base identity (A/C/G/T) |
| shared road of lines and dots | the shared index axis / DP grid coordinate |
| first row, "held still, never moves again" | one string treated as a fixed reference axis |
| horse's-head piece | the one mechanism allowed to introduce a shift (gap) |
| horse standing in each doorway once, and once nowhere | trying a gap at each of the candidate positions, plus the no-gap baseline |
| "every pawn behind it shuffles one house forward, opening a gap" | inserting a blank at that position, shifting the remaining characters |

Breaks: *"a slip can only be discovered by having already compared the position before it"* (the gap site is picked up front, not discovered by extending a best partial path) and *"the whole grid... must be filled in"* (only independent linear scans, not a dependency-chained grid).

**SEED 2 — "lotus petal / red knot, one flute note per house"**

| World object | Problem object |
|---|---|
| lotus petal | match token |
| red knot | mismatch token |
| one flute note per house | strict, single-step position counter |

This SEED mostly *reinforces* "one pair of positions is judged at a time" rather than breaking anything — it's the description of the elementary compare op, not the control structure.

**SEED 3 — "fist of knots weighed, all but the lightest flung to the fish"**

| World object | Problem object |
|---|---|
| fist of knots | per-trial accumulated cost |
| weighing the fist | evaluating a trial's total score |
| keep lightest, fling the rest to the fish | reduction (max/argmin) over independent candidates, discarding the losers entirely |

Breaks: *"every cell... depends on the ones above, left, diagonally above-left, computed in that order"* — outcomes are combined only by a final reduction over independent candidates, not by a mandatory sequential per-cell chain.

None of the three seeds breaks **"both strings are read start to end in the same direction."** Stating that plainly — no seed touches directionality at all; falling back to the most literal seed as instructed.

## CHOSEN SEED

**SEED 1** — the fixed row / slipping row / "every doorway once, and once nowhere" structure. It is the most literal, most structurally distinctive image, and it is what actually motivates a different *shape* of computation (independent units of work instead of a sequentially-chained grid).

## ASSUMPTION BROKEN

*"Every cell of the comparison depends on the ones above, left, and diagonally above-left, computed in that [row-major] order"* — and, tied to it, *"the whole grid... must be filled in"* one cell chained to the last. Taken completely literally, SEED 1 says: pick the gap site first (try every house, and "nowhere"), then judge a whole row at once, then reduce. That literal reading, if implemented exactly as told (one one-sided slip, no compensating close), computes a **restricted, approximate** score: it cannot represent alignments needing more than one indel event, and a lone gap-open with no gap-close cannot even reach a valid `(n,n)` endpoint for equal-length strings, so it will not "exactly match the reference DP" whenever any real gap pattern is needed. Since exactness is a hard contract requirement, I let the seed's real payload — *many independent "houses" judged together in one pass, with only the best kept* — point at the well-known, validated technique that delivers exactly that while staying bit-exact: **anti-diagonal wavefront DP**. On the DP grid, all cells sharing `i+j = d` depend only on diagonals `d-1` and `d-2`, so a whole diagonal ("every house of that trial") can be evaluated together, independently, in one SIMD sweep, with the same 3-way max/discard (SEED 3) at each cell — no invented mechanism, this is exactly what Farrar-style/WFA-style vectorized NW/SW aligners do.

## ARTIFACT

Guard decisions (per the risk rule): small `n` falls back to the textbook scalar DP (wavefront bookkeeping overhead isn't worth it below ~64). No OpenMP threads: a diagonal's parallel width is at most `n`, but there are `2n` diagonals each needing a fork/join-style barrier before the next can start — at realistic DNA-alignment sizes that per-diagonal synchronization overhead is not worth it, so I default to SIMD only and drop the thread-parallel path rather than ship its own stated risk unaddressed.

```c
#include <stdlib.h>
#include <immintrin.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

static int scalar_nw(int n, const char *a, const char *b) {
    int *dp = (int *)malloc((size_t)(n + 1) * (n + 1) * sizeof(int));
    for (int i = 0; i <= n; i++) dp[i * (n + 1) + 0] = i * GAP;
    for (int j = 0; j <= n; j++) dp[0 * (n + 1) + j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            int diag = dp[(i - 1) * (n + 1) + (j - 1)] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int up   = dp[(i - 1) * (n + 1) + j] + GAP;
            int left = dp[i * (n + 1) + (j - 1)] + GAP;
            int best = diag;
            if (up > best) best = up;
            if (left > best) best = left;
            dp[i * (n + 1) + j] = best;
        }
    }
    int result = dp[n * (n + 1) + n];
    free(dp);
    return result;
}

int kernel(int n, const char *a, const char *b) {
    if (n < 64) return scalar_nw(n, a, b);

    /* Anti-diagonal wavefront: dp(i,j) depends only on cells at i+j-1 and
       i+j-2, so a whole anti-diagonal (the "houses" of one trial) is
       mutually independent and is judged together with AVX2, 8 lanes
       (int32) at a time -- one "flute note" (loop step) per diagonal,
       many houses per note. */
    size_t buf_sz = (size_t)(n + 1) * sizeof(int);
    int *prev2 = (int *)malloc(buf_sz);
    int *prev1 = (int *)malloc(buf_sz);
    int *cur   = (int *)malloc(buf_sz);

    prev2[0] = 0;     /* d = 0 : (0,0)  */
    prev1[0] = GAP;    /* d = 1 : (0,1)  */
    prev1[1] = GAP;    /* d = 1 : (1,0)  */

    const __m256i vmatch    = _mm256_set1_epi32(MATCH);
    const __m256i vmismatch = _mm256_set1_epi32(MISMATCH);
    const __m256i vgap      = _mm256_set1_epi32(GAP);

    for (int d = 2; d <= 2 * n; d++) {
        int i_lo = d - n; if (i_lo < 0) i_lo = 0;

        if (i_lo == 0) cur[0] = d * GAP;   /* top boundary,  j = d */
        if (d <= n)    cur[d] = d * GAP;   /* left boundary, j = 0 */

        int lo = i_lo < 1 ? 1 : i_lo;
        int hi = (d - 1 < n) ? (d - 1) : n;

        int i = lo;
        while (i <= hi) {
            int remaining = hi - i + 1;
            if (remaining >= 8) {
                int abuf[8], bbuf[8];
                for (int t = 0; t < 8; t++) {
                    int ii = i + t;
                    int jj = d - ii;
                    abuf[t] = (unsigned char)a[ii - 1];
                    bbuf[t] = (unsigned char)b[jj - 1];
                }
                __m256i av = _mm256_loadu_si256((const __m256i *)abuf);
                __m256i bv = _mm256_loadu_si256((const __m256i *)bbuf);
                __m256i eq = _mm256_cmpeq_epi32(av, bv);
                __m256i add = _mm256_blendv_epi8(vmismatch, vmatch, eq);

                __m256i diagv = _mm256_loadu_si256((const __m256i *)&prev2[i - 1]);
                __m256i upv   = _mm256_loadu_si256((const __m256i *)&prev1[i - 1]);
                __m256i leftv = _mm256_loadu_si256((const __m256i *)&prev1[i]);

                diagv = _mm256_add_epi32(diagv, add);
                upv   = _mm256_add_epi32(upv, vgap);
                leftv = _mm256_add_epi32(leftv, vgap);

                __m256i best = _mm256_max_epi32(diagv, upv);
                best = _mm256_max_epi32(best, leftv);

                _mm256_storeu_si256((__m256i *)&cur[i], best);
                i += 8;
            } else {
                for (int t = 0; t < remaining; t++) {
                    int ii = i + t;
                    int jj = d - ii;
                    int diag = prev2[ii - 1] + (a[ii - 1] == b[jj - 1] ? MATCH : MISMATCH);
                    int up   = prev1[ii - 1] + GAP;
                    int left = prev1[ii] + GAP;
                    int best = diag;
                    if (up > best) best = up;
                    if (left > best) best = left;
                    cur[ii] = best;
                }
                i += remaining;
            }
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

PREDICTION: speedup_vs_dp = 3.0

## MEASUREMENT

I do not have `alignment_bench`/`alignment_contract` (or any code-execution tool) available in this session — only document/email/Slack/Drive tools were provided. I cannot honestly claim to have measured this; the prediction above is stated, unmeasured, for the pipeline to run and check. Reporting this plainly rather than fabricating a number.

## VERDICT

The literal SEED-1 reading (a single one-sided slip, tried at every house, keep the trial with the fewest knots) is an *approximate* single-indel heuristic — it cannot reach a valid alignment endpoint on its own for equal-length global alignment and cannot represent optima needing more than one indel event, so it fails the "exactly match the reference DP" contract in general and was rejected rather than shipped. What's shipped instead is the validated real technique the seed's own "independent houses, judged together, discard all but the best" idea converges to: anti-diagonal wavefront DP, which is mathematically identical to the reference recurrence (bit-exact, order of evaluation only reshuffled) and lets AVX2 judge 8 independent cells per step instead of one. Guard applied: below `n=64` the wavefront bookkeeping isn't worth it, so it falls back to the plain scalar DP. Thread parallelism was deliberately **not** added — the stated risk (per-diagonal fork/join overhead across `2n` diagonals dominating at realistic DNA-alignment sizes) is addressed by dropping that mechanism entirely rather than shipping it unguarded. If measurement shows the character-gather overhead eating into the ~8x theoretical SIMD width, the honest expectation is a real but moderate win (my prediction: ~3x), not a dramatic one — that must be checked, not assumed.