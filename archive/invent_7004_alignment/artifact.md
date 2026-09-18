Since no tools are reachable in this session, I've done the derivation by hand below, exactly as instructed — reasoning first, artifact and prediction stated *before* any measurement, honest reporting of what the literal translation can and cannot guarantee.

## MAPPING

**SEED 1 — twin-ringed wheel, straight-through spoke reading**

| World object | Problem object |
|---|---|
| Twin-ringed stone wheel | The two arrays `a[0..n-1]`, `b[0..n-1]` |
| Outer ring / inner ring | Sequence `a` / sequence `b` |
| Teeth / spokes | Character positions `0..n-1` |
| Alphabet marks, "never more than fingers on two hands" (≤10 symbols) | DNA alphabet `{A,C,G,T}` (4 symbols) |
| Clock-bodied watchdog at the hub, ticking one notch | Loop index `i`, advancing one position per iteration |
| "Read straight across, spoke to spoke" | Compare `a[i]` to `b[i]` — **position-aligned**, not all-pairs |
| Bone-colored stone on agreement / bare clay on disagreement | Score contribution `+MATCH` / (translated to the fixed contract) `+MISMATCH` |
| Stone-count on the shard | Running total `S0` for the no-gap (diagonal) alignment |

Broken assumption: *"the whole grid of every position against every other must be filled in."* The wheel explicitly refuses a full spin ("that would be cheating time itself") — it is one O(n) walk, never the O(n²) grid.

**SEED 2 — the watchdog's one seam strike**

| World object | Problem object |
|---|---|
| Watchdog's single "valve," one seam | One perturbation index `s`, chosen from `n` candidates, tried one at a time |
| "Push the inner ring forward by one tooth for everything past it" | Read `b[i+1]` instead of `b[i]` for all `i ≥ s` — a persistent +1 offset |
| "The two rings walk out of step from there to the end" | The offset is *never* resynced within the pass — it is a single indel-pair, tail-to-end alignment |
| "Never two seams in the same pass" | Only one perturbation explored per candidate; no nested/compound gap structures |
| "Each count goes up on its own shard" | One candidate score per seam `s`, computed independently |

Broken assumption: *"a slip (gap) can only be discovered by having already compared the position before it"* — i.e. the DP's forced sequential cell-order (up/left/diag, in that order). SEED 2 instead enumerates every candidate gap location directly and evaluates each one as a free-standing pass, with zero dependency between different seams' computations.

**SEED 3 — owls picking the best shard**

| World object | Problem object |
|---|---|
| Shelf of shards | Array of `n+1` candidate scores |
| Owls pecking the best shard | Max-reduction |
| Burning lesser shards | Discarding all but the winning candidate — no traceback needed, matching the contract (score only) |

This seed doesn't break a listed assumption on its own; it's the reduction that SEED 1+2 require.

## CHOSEN SEED

**SEED 2** — most literal and most different from the known way. It is the one that actually restructures the computation: instead of an O(n²) grid filled in a fixed dependency order, it proposes `n+1` independent, self-contained O(n) linear scans.

## ASSUMPTION BROKEN

"Every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed in that order" **and** "a slip can only be discovered by having already compared the position before it." SEED 2 tries every possible single-gap-pair location as its own independent pass with no cross-dependency.

**The literal-translation payoff:** the story describes `n+1` separate full walks (sounds like O(n²) again — "I do this for every seam"). But being maximally literal about what each walk actually computes shows almost all of it is shared: seam `s`'s count is just "the straight prefix up to `s`" plus "the shifted suffix from `s`," both of which are the *same two running totals* re-used at every seam, not re-walked. Memory that stays still = `a[]`, `b[]`, and the shard shelf (realized literally as two prefix-sum arrays). What flows = the watchdog's tick, i.e. the two O(n) accumulation chains. What's a processor = each seam evaluation, since seams don't depend on each other, can be one SIMD lane/thread doing a lookup+add+compare. Time collapses from the story's apparent O(n²) (n seams × O(n) each) to a real O(n): two sequential prefix passes + one parallel max-reduction.

## ARTIFACT

```c
#include <stdlib.h>
#include <omp.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

/*
 * Literal translation of the wheel:
 *  - prefix0[k] = straight-through "stone count" for the first k spokes (a[i] vs b[i])
 *  - prefix1[k] = shifted "stone count" for the first k spokes after one tooth-advance
 *                 (a[i] vs b[i+1])
 *  - S0         = the shard from the straight-through pass (no seam struck)
 *  - candidate(s) = left-of-seam straight count + right-of-seam shifted count
 *                   + two GAP charges (the seam costs one gap "in", and the
 *                   ring must close back to (n,n) with one gap "out" at the end)
 *  - best       = the shard the owls peck: max over the straight pass and
 *                  every single-seam pass.
 *
 * NOTE (measured honestly, not hidden): this only equals the true global
 * Needleman-Wunsch optimum when the optimal alignment is either gap-free or
 * uses exactly one indel pair that persists to the end of the sequence
 * (a single "tooth advance", never resynced). Alignments needing a local
 * bubble (insertion then a later, separate deletion that resyncs before the
 * end) or more than one indel pair are outside what "one seam, one valve"
 * can represent, and on those inputs this kernel will return a value <= the
 * reference DP score, not necessarily equal to it.
 */
int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int *prefix0 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *prefix1 = (int *)malloc((size_t)n * sizeof(int));

    prefix0[0] = 0;
    for (int i = 0; i < n; i++) {
        int m = (a[i] == b[i]);
        prefix0[i + 1] = prefix0[i] + (m ? MATCH : MISMATCH);
    }

    prefix1[0] = 0;
    for (int i = 0; i < n - 1; i++) {
        int m = (a[i] == b[i + 1]);
        prefix1[i + 1] = prefix1[i] + (m ? MATCH : MISMATCH);
    }

    int S0 = prefix0[n];
    int total_shifted = prefix1[n - 1];
    int best = S0;

    #pragma omp parallel for reduction(max:best) schedule(static)
    for (int s = 0; s < n; s++) {
        int left  = prefix0[s];
        int right = total_shifted - prefix1[s];
        int candidate = left + right + 2 * GAP;
        if (candidate > best) best = candidate;
    }

    free(prefix0);
    free(prefix1);
    return best;
}
```

PREDICTION: speedup_vs_dp = 80

Reasoning for the number: reference DP is O(n²) work with a strict dependency chain (limits vectorization/parallelism); this kernel is O(n) work (two linear prefix passes + one parallel max-reduction), a genuine asymptotic win, not just a constant-factor SIMD win. For benchmark-scale `n` (low thousands), raw O(n)-vs-O(n²) suggests a ratio in the thousands, but two `malloc`s, OpenMP thread-launch overhead on a short parallel region, and the DP's very cache-friendly, branch-predictable inner loop all eat into that — so I expect the realized speedup to land roughly two orders of magnitude, not three.

## MEASUREMENT

Not performed in this session — no tool access was available here (explicitly noted in the environment). The kernel above is handed to the pipeline as-is for `alignment_bench` to compile and measure; I have not adjusted the artifact in response to any real measurement, and the four-iteration improvement budget was not used for that reason.

## VERDICT

Speed: on complexity grounds alone this should be dramatically faster than the O(n²) DP — that part of the native's story translates into a real algorithmic win, not just a vectorization trick, because SEED 2's "independent seams" collapse into two prefix sums.

Correctness — the honest caveat, stated plainly rather than smoothed over: this kernel is *not* a general Needleman-Wunsch solver. "One seam, one valve, never two in the same pass, shifted to the end" is a strictly narrower search than the DP's — it covers only the gap-free alignment and single-indel-pair alignments whose gap persists to the sequence's end. It cannot find alignments with a resyncing bubble or with more than one indel pair, and on any benchmark input where the true optimum needs those, this kernel will return a lower score than the reference and fail the exact-match requirement. I am not aware of a way to know, without running `alignment_bench`, whether its test sequences stay inside the restricted case this wheel can represent — that is exactly the kind of failure this exercise asks to report plainly rather than quietly patch by falling back to the textbook DP.