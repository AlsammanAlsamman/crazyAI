# MAPPING (world-object → problem-object, per seed)

**SEED 1 — "Only where two runes share a color does a stroke become a bird and mark a treasure."**

| World object | Problem object |
|---|---|
| owl-headed woman, endless corridor of doors | the DP grid (n+1)×(n+1) |
| ribbon nailed along lintels, going right | string `a`, indexing rows |
| ribbon spooled down through ancestors, going down | string `b`, indexing columns |
| small pot of colors | the 4-letter alphabet {A,C,G,T} |
| a rune sharing a color with another | `a[i]==b[j]` |
| stroke lifting off as a living bird | a DP cell whose value is set via the diagonal *match* transition |
| bare corridor stone, thrown from mind | a DP cell where `a[i]!=b[j]` — never computed |
| "the whole crossing of the two" never built | the full n×n grid of comparisons |

Breaks: **"the whole grid of every position against every other must be filled in."** Explicitly and directly — this is the seed's own stated method.

**SEED 2 — "The creature hops treasure to treasure, always further right and further down, never back."**

| World object | Problem object |
|---|---|
| the creature | a traversal/chaining process over marked cells |
| hopping only further right and down | strictly increasing (i,j) — a monotone chain constraint |
| "no rung is higher than the one born after it" | chain values must be non-decreasing along the path |

Breaks: partially "every cell depends on up/left/diag in that order" — the creature only visits a subsequence of cells, not the full dependency lattice; also implicitly touches "a slip can only be discovered by having compared the position before it," since it skips ahead.

**SEED 3 — "The cracked pillar absorbs exactly one slip before the roof must fall."**

| World object | Problem object |
|---|---|
| the cracked pillar / green-faced man | a fault-tolerance budget of size 1 |
| one stumble, ribbons out of step | one place where the chain order is violated (one out-of-turn treasure) |
| second crack → roof falls, chain thrown out | any chain using ≥2 such violations is discarded entirely |

Breaks: "a slip (gap) can only be discovered by having already compared the position before it" — a bounded number of slips is tolerated *without* re-checking every intervening position.

# CHOSEN SEED

SEED 1. It is the only one of the three that directly breaks **"the whole grid of every position against every other must be filled in,"** and it is the most literal — the native states the non-filling as the method itself, not a side effect.

# ASSUMPTION BROKEN

"The whole grid of every position against every other must be filled in" — *attempted*, but see the honest finding below: for this specific scoring scheme and this specific alphabet, it cannot be broken **and remain exact**, so the final artifact reports that finding rather than quietly reverting to the textbook banded/SIMD method.

Working through the literal mapping rigorously (memory = DP cells, flow = score propagation, processor = one "hop" per treasure, time = row order):

- If "bare stone" cells truly contribute *nothing* (as the seed says — "thrown from mind without a second look"), the computed quantity is an LCS-style match count, not the required NW score, because `MISMATCH=-1 ≠ 0`. A lone mismatch (-1) is *always* cheaper than a detour of two gaps (-4), so almost every non-matching cell is in fact used by the true optimal path via its diagonal mismatch transition — it is not "stone the lemur never sniffs at," it is load-bearing.
- Making the sparse chain *exact* (deriving a closed-form edge cost between two "treasures" q=(i',j'), p=(i,j) with no intervening match: `cost = -min(u,v) - 2|u-v|`, u=i-i'-1, v=j-j'-1) is possible, but the number of treasures r (positions where a rune shares a color) is `Θ(n²/4)` for a 4-letter alphabet — DNA's matches are *dense*, not sparse. Even the best known exact sparse-alignment data structure (O(r log r)) is then `Θ(n² log n)`, strictly worse than the O(n²) baseline; a naive O(r²) chain DP is `Θ(n⁴)` and would not finish.
- Conclusion: for this alphabet size, the "sparse treasure amid empty corridor" premise is empirically false (25% of the grid *is* treasure), and genuinely honoring it while staying exact costs more than filling the grid. This mirrors the instructions' own fallback clause in spirit: it textually breaks the assumption, but doing so correctly and beneficially is not achievable here, so I fall back to the most literal *safe* elements of the seed (SEED1's color-sharing detection isolated and vectorized; SEED2's "ancestor row holds up the corridor" as a rolling dependency) inside an exact full-grid computation.

# ARTIFACT

```c
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    int *prev = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *diag = (int*)malloc((size_t)(n + 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * GAP;

        /* "I dip my brush only where the two runes share a color": for every
           column this comparison is independent of every other column, so
           it vectorizes freely (no left/up dependency here at all). */
        #pragma omp simd
        for (int j = 1; j <= n; j++) {
            int match = (ai == b[j - 1]);
            diag[j] = prev[j - 1] + (match ? MATCH : MISMATCH);
        }

        /* "the creature hops rung to rung, always further right, resting on
           the ancestor row that holds up the corridor": this step is the
           genuinely sequential part (left-dependency), so it stays serial. */
        for (int j = 1; j <= n; j++) {
            int best = diag[j];
            int up   = prev[j] + GAP;
            int left = cur[j - 1] + GAP;
            if (up   > best) best = up;
            if (left > best) best = left;
            cur[j] = best;
        }

        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[n];
    free(prev);
    free(cur);
    free(diag);
    return result;
}
```

This computes the *identical* recurrence as the reference (same three transitions, same order of dependency), so it is exact by construction — it does not skip any (i,j); it only stops materializing the full (n+1)² matrix (O(n) memory instead of O(n²)) and isolates the "shares a color?" comparison into a branch-free, vectorizable pass.

PREDICTION: speedup_vs_dp = 5.0

# MEASUREMENT

No tools were available in this session (per the task setup, "the pipeline will compile and measure it" after this answer is submitted — I did not get to call `alignment_bench` myself). I cannot report an observed number; the prediction above is what I expect the pipeline to find, reasoned as follows:
- Correctness: expected to be an **exact match** to the reference DP for every input, since the recurrence, transition order, and final cell are unchanged — only storage and the score lookup are restructured.
- Speed: expected gain comes entirely from (a) O(n) vs O(n²) memory traffic — the reference kernel does two multiplies and an add per index into a huge malloc'd block, ours walks three small rolling arrays sequentially, and (b) the match/mismatch computation and diagonal add being pulled out of the dependency chain so `-O3 -march=native`/`#pragma omp simd` can vectorize them, versus the reference's fully scalar, branchy inner loop. No asymptotic gain is expected or claimed.

# VERDICT

The literal seed — "don't build the whole crossing, only mark where colors share a rung" — was taken completely seriously and pushed through to a precise, checkable claim, not just used as flavor text. That check failed *honestly*: for DNA's 4-symbol alphabet, roughly a quarter of the grid is a "treasure," so the corridor isn't mostly bare stone, and because `mismatch=-1≠0`, the untouched cells actually carry real, non-discardable score. Doing the sparse-treasure idea *exactly* costs at best the same order of work as filling the grid, and at worst far more (O(r²)=O(n⁴) for a naive chain DP). Rather than quietly swapping in the textbook banded/SIMD aligner to "fix" this, the delivered kernel keeps the two literal, *cheap and safe* pieces of the metaphor that do hold up under scrutiny — color-sharing detection as an independent, vectorizable pass, and the ancestor-row as the sole carrier of sequential dependency — inside an unmodified, exactly-correct recurrence. The honest result of this exploration is therefore a modest, memory/vectorization-driven speedup rather than an asymptotic one, and that gap between the native's premise and the alphabet's actual statistics is itself the reportable finding.