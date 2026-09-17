# MAPPING (per seed)

**Seed 1 — "A pipe's cut length stands in for a number, so multiplication becomes matching two reeds by resonance."**

| World object | Problem object |
|---|---|
| Reed cut to a length+bore | a stored `double` value (one element of A or B) |
| Reed bed (pre-cut pipes) | memory holding A, B — values pre-existing before use |
| Resonance matching | the multiply primitive reinterpreted as an analog/lookup match (e.g. log-length addition, à la a slide rule) instead of an IEEE `*` |

Assumption broken: **"numbers are IEEE doubles and multiply is the primitive."** (Rejected as a build target below — log/exp per element costs ~10-50× a hardware FMA, so a literal reading of this seed would make the kernel *slower*, not faster. Noted, not chosen.)

**Seed 2 — "A bow-drawn finch peeled wet off the page is the recorded product of one paired reed-crossing."**

| World object | Problem object |
|---|---|
| One bow-stroke across two reeds | one FMA-lane multiply of one `(a_k,b_k)` pair |
| The finch (ephemeral, wet ink) | the scalar product, held only in a register, never written to an array |
| "I never keep the finches separate — that would be a table nobody asked for" | rejection of materializing the length-n array of partial products |

Assumption broken: **"one product is one problem; many products are many problems"** — the n products for a cell are never treated as n separate stored results.

**Seed 3 — "A flock of finches converging on a single cry performs the summation across the shared index before being released and discarded."**

| World object | Problem object |
|---|---|
| The flock (many finches airborne at once, not one at a time in a chain) | several **independent** partial-sum accumulators, computed with no serial dependency on each other |
| Converging to "one voice"/cry | a tree/pairwise reduction of those independent accumulators into the single scalar dot product |
| Release through the window, "prism must stay clear for next sheaf" | the accumulator registers are zeroed/reused for the next cell — O(1) resident state per cell, not O(n) |

Assumption broken: **"one product is one problem; many products are many problems"**, specifically its corollary that the sum must be a *single sequential* fold (one accumulator threaded through all n steps). The story explicitly computes many products with no chain between them and only *afterward* merges them — breaking the implicit serial-dependency model of accumulation, not just its storage.

# CHOSEN SEED
Seed 3 (the flocking/convergence seed), built together with the discard-and-reuse detail. It is the most literal (every clause maps onto a concrete micro-architectural act: independent products → independent accumulator registers → tree-reduce → register reuse) and the most different from the known way: OpenBLAS's speed comes from *reusing loaded operands across many output cells* (panel packing + register-tiled microkernel); the native process instead explicitly **discards everything after each cell** ("a finch that has already sung its number is dead weight") and gets its speed purely from how **one** cell's sum is combined, not from cross-cell reuse of loaded data.

# ASSUMPTION BROKEN
"One product is one problem; many products are many problems" — reinterpreted at the hardware level as: the textbook/naive kernel folds the k-sum through **one** dependent accumulator (`C[i][j] += a*b[k]`, a serial chain bound by FMA latency, ~4-5 cycles/step ⇒ far under 1 FMA/cycle throughput). The native process computes several products with no dependency between them (the flock airborne simultaneously) and reduces them together at the end, breaking the serial-accumulation assumption and unlocking full FMA-port throughput.

# ARTIFACT
Literal mapping of every world-object to a computational one:
- **Reed bed** → main memory holding A, B (row-major, read-only).
- **"Every cell already cut into a pipe"** → a pre-processing pass: transpose B into `Bt` once, so "column-reeds" become contiguous ("pre-cut") instead of strided.
- **Row-reeds carried to the desk** → row `i` of A, kept resident in registers/cache for the whole row.
- **Column-reeds** → row `j` of `Bt` (== column j of B), now contiguous.
- **Sheaf** → the paired `(Arow, Bcol)` vectors for one output cell.
- **Owl-headed woman's desk / prism** → one CPU core's AVX2 FMA unit(s); OpenMP gives each core its own "hut/desk," parallel over `i` (matrices are read-only, C rows disjoint ⇒ no race, "the marsh has reeds enough").
- **Bow-stroke, one finch per pair** → one `_mm256_fmadd_pd` producing one partial-product vector.
- **The flock (not kept separate, not chained)** → four *independent* `__m256d` accumulators unrolled across k, so no accumulator depends on the previous FMA's result.
- **Converging to one cry** → tree-reduce the 4 accumulators, then horizontal-add the 4 SIMD lanes to one scalar.
- **Write into the third table, ink dries** → `Crow[j] = cry`.
- **Release the flock / prism stays clear** → accumulators are re-zeroed per cell; no state carried beyond the stored scalar.
- **Time** → the outer `(i,j)` loop nest, one "sheaf" processed at a time per thread; "she never tires" → no special-casing, same procedure for all n² cells.

```c
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;
    size_t N = (size_t)n;

    /* "every cell has already been cut into a pipe": pre-cut the column-reeds
       of B into contiguous rows (Bt) so the sheaf (row-reed + column-reed)
       can be carried to the desk without straying through strided memory. */
    double *Bt = (double *)malloc(N * N * sizeof(double));
    if (!Bt) {
        memset(C, 0, N * N * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                double a = A[(size_t)i * N + k];
                for (int j = 0; j < n; j++) C[(size_t)i * N + j] += a * B[(size_t)k * N + j];
            }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        const double *Brow = B + (size_t)k * N;
        for (int j = 0; j < n; j++) Bt[(size_t)j * N + k] = Brow[j];
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *Arow = A + (size_t)i * N;
        double *Crow = C + (size_t)i * N;

        for (int j = 0; j < n; j++) {
            const double *Bcol = Bt + (size_t)j * N;

            /* the flock: four independent accumulators, no bow-stroke
               waits on the one before it */
            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            __m256d acc2 = _mm256_setzero_pd();
            __m256d acc3 = _mm256_setzero_pd();

            int k = 0;
            for (; k + 16 <= n; k += 16) {
                acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k),      _mm256_loadu_pd(Bcol + k),      acc0);
                acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 4),  _mm256_loadu_pd(Bcol + k + 4),  acc1);
                acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 8),  _mm256_loadu_pd(Bcol + k + 8),  acc2);
                acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 12), _mm256_loadu_pd(Bcol + k + 12), acc3);
            }

            /* the flock converges to one voice */
            __m256d acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));

            for (; k + 4 <= n; k += 4)
                acc = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k), _mm256_loadu_pd(Bcol + k), acc);

            __m128d lo   = _mm256_castpd256_pd128(acc);
            __m128d hi   = _mm256_extractf128_pd(acc, 1);
            __m128d sum2 = _mm_add_pd(lo, hi);
            __m128d sum1 = _mm_hadd_pd(sum2, sum2);
            double cry = _mm_cvtsd_f64(sum1);

            for (; k < n; k++) cry += Arow[k] * Bcol[k];   /* stragglers of the flock */

            Crow[j] = cry;   /* write into the matching cell before the ink dries */
        }
    }

    free(Bt);   /* release the flock; the prism stays clear for the next sheaf */
}
```

PREDICTION: speedup_vs_blocked = 4.0

# MEASUREMENT
No `kernel_bench` (or any tool) was actually invocable in this session — the tool list available to me contained none of `kernel_bench`/`kernel_contract`/the symbolic or unconventional tools named in the prompt, only unrelated Docs/Gmail/Drive/Slack tools. I am reporting this plainly rather than fabricating a benchmark result: **the kernel above was not executed or timed by me.** Per the task instructions, the pipeline is expected to compile and measure it against the cache-blocked triple loop and OpenBLAS.

# VERDICT
Unmeasured, so no honest performance verdict can be given yet — only the reasoning behind the prediction:
- Transposing B removes the strided-access penalty that a naive/lightly-blocked triple loop suffers on the shared index, which should already beat a modest blocked loop.
- Four independent accumulators break the serial FMA-latency chain (the direct, literal payoff of "the flock, not kept separate"), which should let the loop approach FMA-port throughput instead of latency-bound throughput — a real, well-established technique, not a rediscovery of the naive method.
- OpenMP over rows adds near-linear core scaling since A, B, Bt are read-only and C rows are disjoint.
- It is *not* expected to beat OpenBLAS: there is no reuse of loaded row/column data across multiple output cells (register-tiling for reuse), so Bt is re-streamed from memory/L3 once per row of A — exactly the cross-cell reuse the native story explicitly discards ("dead weight… prism must stay clear"). That discarding is the literal, honest cost of following the seed instead of quietly substituting the textbook panel-packing method.