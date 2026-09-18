# MAPPING

**SEED 1 — tube / two planks / sill**

| World object | Problem object |
|---|---|
| symbol, "give each symbol its own smell" | DNA base A/C/G/T mapped to a small fixed code (bounded 4-symbol alphabet) |
| dyer's shelf, "hatful of smells... kept cool" | a small, cache-resident, bounded amount of live state — not the whole sequence's worth |
| the two planks, laid side by side | sequence `a` and sequence `b`, laid out linearly |
| the tube, "feed it a smell from each plank at once, moving down the line" | the arithmetic unit that scores a pair of aligned positions; many pairs fed "at once" rather than one at a time |
| bead set on the sill, count going up | the running DP value for a diagonal cell, held in a thin rolling accumulator, not a stored matrix |

Breaks: *"the whole grid... must be filled in [and stored]"* and *"computed in [strict row-major] order"* — the plank/tube/sill picture only ever keeps a **thin, bounded slice** alive, and processes many position-pairs **simultaneously** ("at once, moving down the line") rather than cell-by-cell in row order.

**SEED 2 — pinch of sand / single peg**

| World object | Problem object |
|---|---|
| pinch of sand held on one peg | a single active gap (insertion *or* deletion, never both at once) |
| "never two pegs... a house with two held back stops telling you anything true" | formal fact: for this scoring scheme, an insertion move immediately followed by a deletion move (or vice versa) is always dominated by a single diagonal move, so optimal paths never hold both planks back at once |
| carried to the raspberry stand, uncounted | a speculative gap-branch that never resolves and is pruned/discarded |

Breaks: *"a slip can only be discovered by having already compared the position before it"* — suggests gaps are tracked as one discrete state variable (open/closed), not read off an already-filled 2-D array.

**SEED 3 — run the plank twice, keep the cheaper**

| World object | Problem object |
|---|---|
| running the plank once each way the strings could lean | a forward DP pass (start→end) and a backward/mirrored DP pass (end→start) |
| keep whichever used less sand / stranded fewer smells | compare the two computations, keep the better one |

Breaks: *"both strings are read start to end in the same direction"* — introduces a second, opposite-direction pass.

# CHOSEN SEED

SEED 1 (tube / planks / sill). It is the most literal (every noun maps onto a concrete DP object) and the most structurally different from both the known O(n²)-stored-table DP *and* the known banded/SIMD variant, which still walk row-major and materialize a matrix (full or banded). SEED 1 instead changes **both** the memory shape (O(n) not O(n²)) **and** the traversal order (anti-diagonal simultaneity instead of row-major sequence), while leaving the recurrence itself untouched — so exactness is preserved for free.

# ASSUMPTION BROKEN

"every cell of the comparison depends on the ones above, to the left, and diagonally above-left, **computed in that order**" and "the whole grid of every position against every other must be **filled in** [and kept]." SEED 1 keeps the causal dependency (a cell still needs up/left/diag) but reorders execution into anti-diagonals — cells sharing a diagonal are mutually independent and get "fed to the tube at once" — and only three O(n) diagonals ("the hatful of smells kept cool") are ever resident, never the full O(n²) grid.

# ARTIFACT

```c
#include <stdlib.h>
#include <omp.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

/* Anti-diagonal ("wavefront") Needleman-Wunsch: the two sequences are two
 * "planks" of symbol-codes; only three O(n) diagonals ("the hatful of
 * smells kept cool on the sill") are ever alive in memory at once, never
 * the full O(n^2) grid. Every position pair that shares a diagonal is fed
 * to the scoring "tube" simultaneously, so the sweep can be split across
 * threads/SIMD lanes instead of forced into strict row-major order. The
 * recurrence itself is untouched, so the result is bit-exact with the
 * reference O(n^2) table. */
int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int m = n + 1;
    int *bufA = (int *)malloc(sizeof(int) * (size_t)m);
    int *bufB = (int *)malloc(sizeof(int) * (size_t)m);
    int *bufC = (int *)malloc(sizeof(int) * (size_t)m);

    int *dp2 = bufA; /* diagonal s-2 */
    int *dp1 = bufB; /* diagonal s-1 */
    int *dp0 = bufC; /* diagonal s   (being written) */

    #pragma omp parallel
    {
        for (int s = 0; s <= 2 * n; s++) {
            int i_lo = (s - n > 0) ? (s - n) : 0;
            int i_hi = (s < n) ? s : n;

            #pragma omp for simd schedule(static) if(i_hi - i_lo > 256)
            for (int i = i_lo; i <= i_hi; i++) {
                int j = s - i;
                int val;
                if (i == 0) {
                    val = j * GAP;
                } else if (j == 0) {
                    val = i * GAP;
                } else {
                    int diagv = dp2[i - 1] + ((a[i - 1] == b[j - 1]) ? MATCH : MISMATCH);
                    int upv   = dp1[i - 1] + GAP;
                    int leftv = dp1[i]     + GAP;
                    int best  = diagv;
                    if (upv   > best) best = upv;
                    if (leftv > best) best = leftv;
                    val = best;
                }
                dp0[i] = val;
            }

            #pragma omp single
            {
                int *tmp = dp2;
                dp2 = dp1;
                dp1 = dp0;
                dp0 = tmp;
            }
        }
    }

    int result = dp1[n];
    free(bufA); free(bufB); free(bufC);
    return result;
}
```

# PREDICTION

PREDICTION: speedup_vs_dp = 5.0

Reasoning: the reference kernel mallocs and touches an O(n²) matrix — for even modest n this blows past L2/L3, so it is memory-bandwidth-bound. This kernel touches only 3×O(n) ints, which stays cache-resident, so it should be compute-bound instead; combined with anti-diagonal work-sharing across threads (guarded by the `if(i_hi-i_lo>256)` so short diagonals near the corners stay serial and avoid barrier overhead), I expect a real but moderate win rather than a spectacular one — the O(2n+1) barrier-per-diagonal cost is a genuine tax this architecture has to pay that a plain row-major SIMD/banded aligner does not.

# MEASUREMENT

No tools were available in this session (the environment explicitly disabled `alignment_bench`/`alignment_contract` calls here), so I could not execute the compile-and-measure step myself. I am reporting this plainly rather than inventing numbers: the PREDICTION above is the honest pre-registered estimate: it has **not** been empirically checked in this session, and the external pipeline referenced in the task is expected to supply the actual measured `speedup_vs_dp` against the reference DP.

# VERDICT

Unverified pending measurement. The artifact is a literal, exact (same recurrence, same result) translation of the "tube/planks/sill" object: O(n) memory instead of O(n²), anti-diagonal simultaneity instead of row-major order, single-fork OpenMP region with a size-gated work-sharing construct to limit synchronization overhead on short diagonals. If the measured speedup comes in well below the predicted 5.0×, the most likely honest culprit — per the reasoning above — is that the 2n+1 per-diagonal barriers cost more than the anti-diagonal parallelism buys back, especially if the benchmark's `n` is small; that would be worth reporting as plainly as a win would be.