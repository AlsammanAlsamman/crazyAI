## MAPPING

| World object | Problem object | 
|---|---|
| Shape's edge, one blade per "position that can be chosen or left out" | One bit of a binary index: with n = 2^m, there are exactly m such positions (the bits of j and k) |
| Blade pressed flat / left standing, "no bending it partway" | A bit being 1 (set) or 0 (unset) — exactly the binary choice inside popcount(j&k); explicitly *not* a continuous phase (that's the "gentler places" = DFT/FFT's complex ripples, ruled out) |
| The city seen through the window, "repeats itself exactly, day for day" | The *running partial array* that is **not re-read from scratch each day** — i.e. the accumulated state from previous stages is reused, not recomputed from the original `in[]` every time |
| One dawn's "single arrangement" pressed along the row | One fixed doubling-scale (one stage `len`) applied uniformly to all butterfly pairs that day |
| Breath fogging the glass, held, then read as one strip of flat stripes | One complete sweep over the whole array, in place, producing the whole intermediate array for that stage before anything is disturbed again |
| Breath running out, city freezing then snapping back, window blind again, ready | The buffer is overwritten in place (no separate scratch retained) and the loop advances to the next stage — "reset" = ping-pong done in-place, not a fresh allocation |
| Mirror going furious the instant a reading repeats | Each doubling-scale `len` (1,2,4,…,n/2) is visited **exactly once**, never revisited |
| Walking the row until "no unrepeated days left to spend" | Loop terminates after exactly log₂(n) stages — exhaustive, no more, no less |

Assumption each seed breaks:
- SEED 1 (on/off blades = bits, no partial bending) → breaks "the input order cannot be rearranged" / "each output is one independent sum": it forces us to think in terms of shared bits, not per-output sums.
- SEED 2 (fog = one full pattern before it clears) → breaks "there is no way to reuse work between different outputs" and "computed as one pass": the array is transformed progressively across several full passes, each reusing the previous pass's combined values.
- SEED 3 (freeze/reset advances to the next untried combination, discards the last) → breaks "combining two contributions always means an ordinary addition": the discard is literal — the old pair (a,b) is replaced by (a+b, a−b), an add **and** a subtract, not accumulation.

## CHOSEN SEED
SEED 2 ("breath fogging the window and being read back records one complete stripe pattern before it clears") — most literal (one dawn = one full, uninterrupted sweep of the array) and most different in *shape* from the given "known way" (a *recursive* call-stack butterfly): here the ritual is manifestly iterative/bottom-up and in-place — the city/window is reused turn over turn rather than a function calling itself.

## ASSUMPTION BROKEN
"there is no way to reuse work between different outputs" — each day's fogged reading is built directly from the *previous* day's fogged reading (the window shows the same, unchanging city, but the fog pattern accumulates), not recomputed from the raw input each time.

## ARTIFACT

```c
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Iterative, in-place Fast Walsh-Hadamard Transform.
 * "Blades" = bits of the index (m = log2(n) of them).
 * "One dawn's arrangement" = one doubling-scale 'len' applied to the
 * whole array in a single sweep ("breath -> fog -> one stripe pattern").
 * "Freeze & reset" = overwrite in place, move to the next (never-repeated)
 * scale, until all log2(n) scales have been visited exactly once.
 */
void kernel(int n, const double *in, double *out) {
    memcpy(out, in, (size_t)n * sizeof(double));

    for (int len = 1; len < n; len <<= 1) {
        int step = len << 1;
        int nblocks = n / step;

        #pragma omp parallel for schedule(static) if(nblocks > 1 && (long)n > 4096)
        for (int b = 0; b < nblocks; b++) {
            int i = b * step;
            double * restrict pa = out + i;
            double * restrict pb = out + i + len;
            #pragma omp simd
            for (int j = 0; j < len; j++) {
                double a = pa[j];
                double c = pb[j];
                pa[j] = a + c;   /* blade left standing: "+" contribution   */
                pb[j] = a - c;   /* blade pressed flat: "-" contribution    */
            }
        }
    }
}
```

PREDICTION: speedup_vs_naive = 500

## MEASUREMENT
Not run in this session — no tool access was provided here (the instructions for this turn state the reasoning/artifact/prediction must be produced first, and that "the pipeline will compile and measure it" afterward using `wht_bench`/`wht_contract`). No `wht_bench` call was made, so I am not fabricating a number here; the measurement is pending execution by the pipeline against the kernel above.

## VERDICT
Pending the pipeline's actual run of `wht_bench`. Expected outcome based on complexity alone: naive is O(n²), this kernel is O(n log n) with all-additive/subtractive butterflies (no multiplies), vectorized inner loop, and OpenMP over independent blocks once n is large enough to amortize thread overhead — so for large n (thousands to millions) the predicted ~500× figure should be a plausible (if rough) order-of-magnitude estimate; for small n the fixed overhead of `memcpy`+OpenMP dispatch could make the realized speedup considerably smaller than the asymptotic ratio n/log₂(n) suggests. If the measured speedup falls well short of 500×, the two most likely fixable points (my "improve at most four times" budget) are: (1) the `if(...)` OpenMP guard threshold may need tuning per n, and (2) for very large n a cache-blocked stage order (processing several small-`len` stages together per cache-resident block) would reduce memory traffic — that would be improvement #1 if measurement shows a memory-bound wall at large n.