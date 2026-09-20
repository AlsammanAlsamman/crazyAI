# MAPPING

**SEED 1** — "The low sun strikes both ropes at once and throws a single doubled shadow only where two knots share a dye, marking every treasure without a hand ever touching the floor."

| World object | Problem object |
|---|---|
| wall-rope (knots dyed 1-of-4) | string `a`, one char per knot, alphabet {A,C,G,T} |
| floor-rope, hung at a slant | string `b`, laid so its knots meet `a`'s knots along the **anti-diagonal** (i+j=const) |
| the empty room / floor never tiled | the full n×n DP matrix — never materialized in memory |
| every knot-crossing (mind's eye) | every cell (i,j) of the conceptual comparison grid |
| the cat sitting where colors match | the equality test `a[i]==b[j]` |
| the one hour when low sun strikes | a single bulk/vector operation that evaluates many crossings simultaneously |
| the doubled shadow | the match/mismatch score for a whole batch of positions, produced at once, not knot-by-knot |

Breaks: **"the whole grid of every position against every other must be filled in"** (attempted) and "one pair of positions is judged at a time."

**SEED 2** — "The cost of the bare floor... read straight off how many knots forward each rope must go, never walked or marked knot by knot."

| World object | Problem object |
|---|---|
| bare floor between two treasures | the run of DP cells between two match-cells |
| "knots forward" on each rope | Δi, Δj between two matches |
| reading the cost off the ropes' lengths | a closed-form gap-cost formula instead of cell-by-cell DP |

Breaks: **"a slip (gap) can only be discovered by having already compared the position before it."**

**SEED 3** — "At each treasure only the cheapest of the arriving tallies — slip spent or slip still whole — is kept..."

| World object | Problem object |
|---|---|
| running tally | a DP score carried between matches, not between adjacent cells |
| "slip spent / not yet spent" | a 2-state (gap-used / gap-unused) chain DP over matches only |
| dropping a costlier tally | standard DP pruning (keep max, discard the rest) |

Breaks: **"every cell depends on the ones above/left/diag-above-left, computed in that order"** (order becomes match-to-match, not row-by-row) and "one pair of positions is judged at a time."

# CHOSEN SEED

SEED 1 is the most literal and the only one that targets the assumption we're told to prefer breaking ("whole grid... filled in"). But taking it fully literally means: detect matches, keep only match-cells ("treasures"), do sparse chain-DP between them (SEEDs 2+3), skip the rest.

Checked honestly against the actual problem: DNA has a **4-letter alphabet**, so `P(a[i]==b[j]) ≈ 1/4`. "Treasures" are *not* sparse — they cover ~n²/4 of the grid, not "few... in a mostly empty corridor." The sparse-chain shortcut that makes SEED-1-as-anchoring work for text diff (Hunt–Szymanski, minimap2 chaining) collapses here: n²/4 anchor nodes is not cheaper than n² DP cells. Worse, SEED 3's "slip past each other **once**" means *at most one* gap event — but exact global NW allows arbitrarily many indels, so a one-slip chain would silently give **wrong** scores on any input needing ≥2 gap events. That fails the hard contract ("exactly matching a reference DP aligner").

So: **none of the three seeds can safely break "whole grid must be filled" while staying exact** for this alphabet. Per the instructions, I say that plainly and fall back to the most literal seed (SEED 1), but keep only its safe, validated core — bulk/simultaneous match detection across many crossings at once — and let it converge on the known, validated technique this literally *is*: **anti-diagonal wavefront vectorized NW with O(n) rolling memory**, the same idea underlying KSW2/SSW-style SIMD aligners (explicitly named as acceptable "Known way" in the prompt).

# ASSUMPTION BROKEN

Not "whole grid filled" (every cell's *value* is still computed — DNA's match density makes skipping cells unsafe, so I say this honestly rather than fake it).

What genuinely breaks:
- **"every cell depends on... computed in [row-major] order"** — cells are computed along the anti-diagonal (the "slant" in the story), batching all mutually-independent cells of a wavefront together.
- **"one pair of positions is judged at a time"** — a contiguous vector of `a`/reversed-`b` positions is compared and scored together (`#pragma omp simd`), literally "the sun strikes many knots at once."
- Literal reading of "**I do not mark the floor**" — the O(n²) matrix ("floor") is never allocated; only 3 rolling O(n) diagonal buffers exist, cutting memory traffic from O(n²) to O(n).

# ARTIFACT

```c
#include <stdlib.h>
#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *restrict a, const char *restrict b) {
    if (n <= 0) return 0;

    int  *prev2 = (int  *)malloc((size_t)(n + 1) * sizeof(int));
    int  *prev1 = (int  *)malloc((size_t)(n + 1) * sizeof(int));
    int  *cur   = (int  *)malloc((size_t)(n + 1) * sizeof(int));
    char *rb    = (char *)malloc((size_t)n);

    /* "the floor rope hung at a slant": reverse b so that walking the
       anti-diagonal reads BOTH ropes forward and contiguously. */
    for (int k = 0; k < n; k++) rb[k] = b[n - 1 - k];

    /* seed diagonal d = 0 : dp[0][0] = 0 */
    prev2[0] = 0;
    /* seed diagonal d = 1 : dp[0][1] = GAP, dp[1][0] = GAP */
    prev1[0] = GAP;
    prev1[1] = GAP;

    for (int d = 2; d <= 2 * n; d++) {
        int ilo = (d - n > 0) ? d - n : 0;
        int ihi = (d < n) ? d : n;

        if (ilo == 0) cur[0] = prev1[0] + GAP;         /* i=0,j=d: left only  */
        if (ihi == d) cur[d] = prev1[d - 1] + GAP;      /* i=d,j=0: up only    */

        int mid_lo = (ilo == 0) ? 1 : ilo;
        int mid_hi = (ihi == d) ? d - 1 : ihi;

        const char *restrict aa  = a;
        const char *restrict rbb = rb;
        int  *restrict p2 = prev2;
        int  *restrict p1 = prev1;
        int  *restrict c  = cur;
        int base = n - d;   /* rb[base+i] == b[d-i-1] : contiguous forward */

        #pragma omp simd
        for (int i = mid_lo; i <= mid_hi; i++) {
            int sc    = (aa[i - 1] == rbb[base + i]) ? MATCH : MISMATCH;
            int diagv = p2[i - 1] + sc;
            int upv   = p1[i - 1] + GAP;
            int leftv = p1[i]     + GAP;
            int best  = diagv;
            if (upv   > best) best = upv;
            if (leftv > best) best = leftv;
            c[i] = best;
        }

        int *tmp = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur   = tmp;
    }

    int result = prev1[n];

    free(prev2);
    free(prev1);
    free(cur);
    free(rb);

    return result;
}
```

Hand-verified against the reference recurrence on `a="AC"`, `b="AG"` (n=2), tracing every diagonal (d=0..4): produced dp[0][2]=-4, dp[1][1]=1, dp[2][0]=-4, dp[1][2]=-1, dp[2][1]=-1, dp[2][2]=0 — all match the textbook table cell-for-cell, final score 0. The reordering only changes *when* a cell is computed, never the recurrence, so it is exact by construction (still every dependency is a diagonal-minus-1 or diagonal-minus-2 read).

PREDICTION: speedup_vs_dp = 4.0

# MEASUREMENT

No tools were invokable in this session (explicit environment note: "No tools are available in this session"). I have not fabricated a measurement — the prediction above (4.0×) is reasoned from: (a) O(n²)→O(n) memory footprint removing most cache/allocation pressure the reference's full malloc'd n×n array incurs at moderate-to-large n, and (b) the branch-light inner loop (contiguous `a` and reversed-`b` loads, no aliasing) being amenable to AVX2 auto-vectorization under `-O3 -march=native` via the `omp simd` hint (4–8 int32 lanes). This must be confirmed by `alignment_bench`; I report that it was not run here rather than inventing a number.

# VERDICT

The literal seed (sparse "treasure" chaining) does **not** survive contact with the actual problem: DNA's 4-symbol alphabet makes matches dense (~25%), so "few treasures in an empty corridor" is false here, and SEED 3's "one slip" would silently break exactness on any multi-indel case — a correctness failure I'm not willing to ship. I dropped that risky part entirely rather than guard it, since no size/threshold makes a one-gap-limited aligner exact. What survives and is shipped is the honest, narrower part of the metaphor — simultaneous multi-knot illumination — realized as vectorization + O(n) memory on an anti-diagonal traversal, which is a known, validated real-world pattern (KSW2/SSW-style SIMD NW), not a novel unproven mechanism. Remaining risk: for very small n (loop trip counts of 1–2 per diagonal) there's no vectorization win and some small constant overhead (extra buffers, reversed-`b` precompute) — this is self-guarding, since at small n the reference is already sub-microsecond and any constant-factor difference is immaterial; no separate fallback path was needed. Complexity remains O(n²) in both versions — the honest gain here is constant-factor (memory bandwidth + SIMD), not asymptotic, which is the correct and defensible outcome once the sparse shortcut was ruled unsafe.