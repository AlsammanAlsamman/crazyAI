## MAPPING (per seed)

**SEED 1 — "I double every finished stripe by melting a wax‑bound twin onto it and tying the new notch's knot only on that twin."**

| World object | Problem object |
|---|---|
| shape laid flat on the mat | the working array of `n` samples (`out`, initialised from `in`) |
| notches along its edge | the `log2(n)` bit positions / transform stages |
| a stripe (blank both faces, stitched one side, raw the other) | a value at array index `j` at a given stage: "stitched" = bits already combined in earlier stages, "raw" = the bit for this stage not yet folded in |
| stripe binary, never wavering | each combination step produces exactly `a+b` or `a-b` — never a weighted/partial value |
| "single stripe, bare everywhere" = choice of taking nothing | the raw copy of `in[]` before any stage runs (0 notches decided yet) |
| doubling: take a fallen wax bead, press it onto the stitched face, bind an identical twin, tie a knot only at the new notch, leave the original bare | the butterfly at stride `len`: `base[j] = a+b` (original stays bare — no sign flip), `base[j+len] = a-b` (twin gets the new notch's knot — sign flip) |
| "a flash is a knot" | the minus sign — commitment to `-1` at that notch, nothing in between |
| finished stripes stay still, pinned; only newest twins keep moving | values produced by an earlier stage become fixed inputs consumed once more to build the next stage's outputs, never revisited after the final stage |
| thief‑tied: no unpicking, only add or discard | each stage only ever *adds* new notch information via `+`/`-`; it never undoes a prior combination |
| Timesavings Knowledge ledger, checked count | the a‑priori known relationship `n = 2^(#notches)`, used to bound `for (len = 1; len < n; len <<= 1)` |
| burn stripes that are half‑knotted / fused wrong | a correctness check (not on the hot path) — discard/assert if a partial sum is malformed |
| tokens' skin, one token per finished stripe | the final `out[]` array, one transformed value per index |

Breaks: **"there is no way to reuse work between different outputs."** Each partial sum `a` at a stage is reused to generate *two* outputs (`a+b`, `a-b`) instead of every output being recomputed from scratch.

**SEED 2 — "A flash is a knot for full inclusion, bare cord for full exclusion, no wavering."**
Maps directly to the ±1 nature of Hadamard entries. Breaks "combining two contributions always means an ordinary addition" (it's forced binary ±), but this constraint is *already* implicit in the textbook recursive butterfly too — it doesn't suggest a structurally different construction, just restates why `+`/`-` suffice.

**SEED 3 — "verify count against the Timesavings ledger, burn bad stripes."**
Maps to a correctness/assertion step (`n` matches `2^m`, discard malformed intermediate stripes). It touches none of the five listed silent assumptions directly — it's about validation, not about restructuring the computation.

## CHOSEN SEED
**SEED 1** — most literal (it *is* the bit‑by‑bit doubling butterfly, object‑for‑object) and structurally the most different framing from the "known way": the known way is described as top‑down divide‑and‑conquer recursion; SEED 1 is explicitly bottom‑up — iterative stripe‑doubling, stage by stage, never recursing, only ever taking what's already on the mat and doubling it.

## ASSUMPTION BROKEN
"There is no way to reuse work between different outputs" — broken by reusing every partial sum `a` (already-finished stripe) to build both `a+b` and `a-b` in one melt-and-knot step, instead of recomputing `sum_j in[j]*(-1)^popcount(j&k)` independently for every `k`.

## ARTIFACT

```c
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Walsh-Hadamard transform of a length-n (power of two) real sequence.
 * out[k] = sum_j in[j] * (-1)^popcount(j & k)
 *
 * Bottom-up "stripe doubling": the mat starts as a bare copy of `in`
 * (the stripe with no notch decided). For each notch/bit position,
 * every stripe already on the mat is doubled: the original stays bare
 * (a+b, no sign flip) and a wax-bound twin gets the new notch's knot
 * (a-b, sign flip). Finished stripes from a stage are only ever read
 * once more to build the next stage's outputs, never re-derived from
 * scratch — the reuse the naive O(n^2) formula forbids.
 */
void kernel(int n, const double *in, double *out) {
    double * restrict o = out;

    memcpy(o, in, (size_t)n * sizeof(double));

    for (int len = 1; len < n; len <<= 1) {
        int step = len << 1;
        #pragma omp parallel for schedule(static) if(n >= 65536)
        for (int i = 0; i < n; i += step) {
            double *base = o + i;
            for (int j = 0; j < len; j++) {
                double a = base[j];
                double b = base[j + len];
                base[j]       = a + b;   /* bare cord: notch left out */
                base[j + len] = a - b;   /* knotted twin: notch tied in */
            }
        }
    }
}
```

PREDICTION: speedup_vs_naive = 600

## MEASUREMENT
No tools are available in this session (the harness `wht_bench` cannot be invoked here), so I cannot produce an actual measured number now — per the task setup, the pipeline will run `wht_bench` against this artifact externally. Reasoning behind the prediction above: the naive reference is O(n²) with an expensive per-element `popcount` inner loop (branchy, ~log₂n extra iterations per output element), while the kernel above is O(n log n) using only two flops (`+`/`-`) per butterfly, in place, with no popcount at all, plus optional OpenMP parallelism over independent blocks for large n. For n in the low tens of thousands (a typical benchmark size), the raw op-count ratio alone is already ~n/log₂n (~1000+), and the naive kernel's per-element overhead (bit-loop, branch, no vectorization) multiplies that further at the wall-clock level, tempered by the fast kernel being memory-bandwidth-bound with two full array passes per stage. 600× is a middle-of-the-road estimate reflecting those two effects pulling in opposite directions.

## VERDICT
Structurally, this is a correct and literal translation of SEED 1: it is exactly the standard iterative (non-recursive) Fast Walsh–Hadamard Transform, arrived at by taking "doubling a finished stripe by wax-melting an identical twin and knotting only the new notch" as a literal instruction rather than a metaphor for recursion — giving the bottom-up, stage-by-stage butterfly instead of the top-down divide-and-conquer shape implied by "same shape as an FFT." I did not get to run the four allowed measure/improve cycles in this session (no tool access), so the number above is a prediction only; if/when `wht_bench` is run, the loop-guarded OpenMP threshold (`if(n >= 65536)`), unrolling the inner `j` loop, or switching to a radix-4 butterfly would be the natural next three improvements if the measured speedup falls short of the prediction.