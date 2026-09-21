## MAPPING

| World object | Problem object | Assumption it touches |
|---|---|---|
| **SEED 1** — single one-cell cord, crawling in any of eight directions, coiling across its own worn rings | A single sequential fill-pointer moving through the DP grid, cell by cell, in fill order (i,j) increasing; "eight ways" = the Moore neighborhood of a grid cell, of which only forward/sideways/diagonal are used for progress | Reinforces "one pair of positions is judged at a time" — the cord is explicitly *singular* and *one-cell wide*. Does not break it. |
| **SEED 2** — cord curls back on itself to mark a slip, one string holding, one letting go a step | The gap transition: moving to (i,j) via the "up" branch (i increments, j fixed) or the "left" branch (j increments, i fixed) — exactly the two non-diagonal DP transitions, which are gap moves | Reinforces "a gap can only be discovered by having already compared the position before it" — the curl-back needs the prior ring's tally already pressed in. Does not break it. |
| **SEED 3** — cord carries its whole tally coiled in its body, yields only the last ring's weight, then every ring is swept away | dp[i][j] values are read-only history once written (the "coiled body"); only dp[n][n] is ever needed at the end; everything else can be discarded once out of reach | Does not break "whole grid must be filled" (every cell is still visited/computed) — it breaks *storage*, not *coverage*: it says the full O(n²) matrix never needs to exist at once, only a rolling O(n) sliver of it. |

None of the three seeds break **"one pair of positions is judged at a time."** All three, read literally, insist on a *single* cord, one cell wide, moving one ring at a time — that is a restatement of strict serial evaluation, not a relaxation of it. Stating this plainly per the instructions: **no seed breaks the target assumption.**

## CHOSEN SEED

Falling back to the most literal seed, as instructed: **SEED 1** (the fullest, most direct description of the fill mechanism itself — direction set, self-loop for gaps, running tally). SEED 3's "discard every ring but the last coin" is folded in as a literal corollary of the same cord (a single-cell cord has no way to *carry* a whole O(n²) grid on its back — it can only physically hold what's within reach, i.e. the current and previous ring-rows), so it sharpens SEED 1's memory model rather than replacing it.

## ASSUMPTION BROKEN

None of "one pair of positions is judged at a time" is broken — stated plainly, not guessed around. What *is* legitimately licensed by the literal metaphor is a change to assumption implicit in the reference code but not in the five listed ones: that the **entire grid must be simultaneously resident in memory**. SEED 1+3 together say the cord's body ("coiled behind its own head") only ever needs the current and immediately-preceding ring — everything older is "swept from the stone." That is O(n) memory instead of O(n²), a real and literal consequence of "single one-cell cord."

Regarding step 5's dual-regime check: the known_way section does mention two regimes — full O(n²) DP vs. a banded/SIMD variant "exploiting a bounded score range." I considered banding explicitly and rejected it: banding requires knowing in advance that the optimal path stays within some width k of the diagonal, but the contract guarantees nothing about sequence similarity (arbitrary A/C/G/T, no bounded-edit-distance promise), and the required band width is itself an output of the very computation we're trying to shortcut — using a heuristic band risks silently returning a wrong score, which violates "exactly matching a reference aligner." So the single full-grid strategy is the only regime the contract actually admits; the other regime doesn't safely apply here, not because it wasn't considered.

## ARTIFACT

```c
#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    /* The table's two edges are the strings; the cord's body only ever needs
       the ring-row it is on and the one just behind it -- everything older
       is swept from the stone (SEED 3). So: O(n) memory, not O(n^2). */
    int *restrict prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *restrict cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *restrict mrow = (int *)malloc((size_t)(n + 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];

        /* "the girls with ground faces mark each square with the letter it
           must answer to, using chalk cut from the dirt FIRST" -- the
           agreement coin for a whole row is decided before the cord ever
           crawls it. This pass has no loop-carried dependency (each square's
           chalk mark depends only on the two letters meeting there), so it
           is free to auto-vectorize under -O3 -march=native. */
        for (int j = 1; j <= n; j++) {
            mrow[j] = (ai == b[j - 1]) ? MATCH : MISMATCH;
        }

        /* The single one-cell cord crawls this row left to right. At each
           ring it may step forward along one thread (up), sideways along
           the other (left), aslant through both (diag), or curl back onto
           the ring just behind it to record a slip (a gap, via up/left).
           It presses the coin already chalked, adds/spends the carried
           weight, and its whole worn trail behind it is this one row. */
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int diag = prev[j - 1] + mrow[j];
            int up   = prev[j]     + GAP;
            int left = cur[j - 1]  + GAP;
            int best = diag > up ? diag : up;   /* branchless: no compare-and-
                                                    branch drama, just weighing
                                                    two coins against each other */
            best = left > best ? left : best;
            cur[j] = best;
        }

        /* the ring the cord stands on becomes a fixed household (window/
           cylinder) the moment the cord leaves it */
        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[n];   /* the last ring's weight -- the only coin kept */
    free(prev);
    free(cur);
    free(mrow);
    return result;
}
```

PREDICTION: speedup_vs_dp = 3.0

## MEASUREMENT

No tool session is available to me to invoke `alignment_bench`/`alignment_contract` directly this turn — the harness note says the pipeline will compile and measure the artifact above against the prediction. I have not fabricated a number here; the 3.0 prediction above was written before any measurement, as required, and stands to be checked against whatever the pipeline reports.

Reasoning behind the 3.0 estimate: the reference kernel mallocs and touches a full (n+1)×(n+1) `int` matrix (e.g. ~400MB at n≈10000), which for any non-trivial n blows out L1/L2 and turns the fill into a memory-bandwidth-bound streaming walk despite being sequential-friendly row order. The rolling two-row version keeps the entire working set in O(n) ints (a few KB–MB, cache resident for realistic n), eliminates the large allocation/zeroing cost, replaces the reference's repeated `(i-1)*(n+1)+...` index arithmetic with simple pointer walks, and gives the compiler a clean auto-vectorizable comparison loop separated from the serial accumulation. None of this changes what is computed — every cell is still visited in the same order with the same recurrence — so exact correctness is preserved by construction.

## VERDICT

This mechanism is safe everywhere but only *matters* in a specific regime: it wins on memory traffic/cache locality, which is a large effect once n is big enough that the O(n²) matrix stops fitting in cache (roughly n above a few hundred–thousand, machine-dependent), and a negligible-to-zero effect for very small n where the whole reference matrix already fits in cache and malloc cost is trivial. Per the guard-rail rule, I need to address that stated risk rather than ship it silently: here the "risk" is only ever *no speedup*, never *a slowdown* — the rolling-row version uses strictly less memory and strictly fewer bytes touched than the reference at every n, so there is no regime where it should be worse, and therefore no fallback branch is needed (unlike a banding scheme, which I explicitly rejected above because it *could* be wrong, not just slow). I did not reach for thread-level parallelism: the metaphor's own unit of work is one cord, one cell, one thread — nothing in either chosen seed licenses multiple cords, so no OpenMP was added, consistent with defaulting to memory-layout/vectorization hints over threading.