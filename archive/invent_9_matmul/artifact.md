# MAPPING

**SEED 1 — "Sailors flung at every crossing drive shadow into height as a product."**

| World object | Problem object |
|---|---|
| bone-colored stretch / facing stretch | RAM, laid out for A and B |
| furrow (row, notched with larva) | row *i* of A |
| larva / shadow-length | element A[i,k], its value is the "shadow" |
| waterfall (column), fall-height at a level | element B[k,j], its value is the "height" at level k |
| crossing of furrow *i* / waterfall *j* at level k | the index triple (i,k,j) |
| sailor, flung not walked | one scalar multiply, dispatched as an independent unit of work (a lane/thread), not iterated |
| driven mark / depth | the product A[i,k]·B[k,j] |

Breaks: **"one product is one problem; many products are many problems"** — sailors are flung "in the same breath," i.e. many products are issued as one collective SIMD/parallel event, not n³ separate serial problems.

**SEED 2 — "A cairn gathers one stone per mark along a furrow/waterfall pair; its final height is the summed cell."**

| World object | Problem object |
|---|---|
| landmark cairn | accumulator for cell C[i,j] |
| stone dropped per mark | one += of a product into the accumulator |
| "gather every sailor along one whole furrow against one whole waterfall" | reduction over the shared index k for fixed (i,j) |
| rank of waterfalls | a panel/block of columns j₀..j₀+W |
| "whole plane of cairns rises at once" | all (i, j-in-panel) accumulators updated in the same parallel wave, across all furrows (rows) simultaneously |
| "one stillness" when last sailor/waterfall finishes | a barrier: k-reduction complete for every cell in the panel before anything is read out |

Breaks: **"the output is produced one cell at a time, row by row"** and **"one processor holds both matrices"** — many cells (a whole panel) are being summed by many independent flingers at once, synchronized only at the panel boundary, not cell-by-cell in row order by one actor.

**SEED 3 — "Spent sailors, folded shadows, and counted cairns are discarded/leveled the instant their number passes into the third table."**

| World object | Problem object |
|---|---|
| spent sailor, folded shadow/height | the scalar product and its operands — not retained after being folded into the accumulator |
| cairn counted into third table | accumulator flushed (memcpy) into C |
| cairn leveled flat | scratch accumulator memory zeroed and reused for the next panel |
| "a million cairns left standing would become a landmark" | explicit refusal to keep n² live partial sums resident — only one panel's worth of scratch exists at a time |

Breaks: **"a matrix is a two-dimensional grid living in one memory"** — C is never accumulated *in place*; it is repeatedly assembled from a small disposable scratch buffer that is destroyed and recreated per panel.

# CHOSEN SEED
SEED 2 (the cairn/plane), read together with its own elaboration in the passage (ranks, planes, the one stillness) — it is the most literal description of *how the reduction over k is organized in time and space*, and it is the piece that differs most from the textbook cache-blocked loop: cells are not filled one at a time in row order by a single actor; a whole panel of cells is summed in lock-step by many flingers, using disposable scratch, then flushed.

# ASSUMPTION BROKEN
"the output is produced one cell at a time, row by row" and "one processor holds both matrices."

# ARTIFACT — literal object map used to build the kernel
- desert (memory) → scratch RAM
- furrow (row of larva) → row of A
- waterfall / fall-height → column of B
- rank of waterfalls → a column panel of width `NC`
- block of furrows processed "in the same breath" → a row block of height `MC`, parallelized over OpenMP threads
- sailor flung → one FMA lane (AVX2, 4 doubles/instruction)
- cairn / plane of cairns → a private, cache-resident `MC×NC` scratch accumulator per thread, zeroed ("leveled") at the start of each panel
- "one stillness" → the k-loop completing fully for the whole panel before it is read
- cairn counted into third table, then leveled → `memcpy` the plane into C, then reuse (zero) the same buffer for the next panel — C is never touched during accumulation, only at the flush

```c
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

#define MC 64    /* furrows in a block ("a rank's worth of furrows")   */
#define NC 256   /* waterfalls in a rank (column panel width), mult of 4 */
#define KC 256   /* depth of one "breath" (k-block)                     */

void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    #pragma omp parallel
    {
        /* the cairn plane: disposable scratch, reused every panel */
        double *plane = (double *)aligned_alloc(32, (size_t)MC * NC * sizeof(double));

        #pragma omp for collapse(2) schedule(dynamic)
        for (int jc = 0; jc < n; jc += NC) {
            for (int ic = 0; ic < n; ic += MC) {
                int nc = (jc + NC <= n) ? NC : (n - jc);
                int mc = (ic + MC <= n) ? MC : (n - ic);

                /* level the plane flat before this furrow-block/rank begins */
                memset(plane, 0, (size_t)MC * NC * sizeof(double));

                for (int kc = 0; kc < n; kc += KC) {
                    int kcb = (kc + KC <= n) ? KC : (n - kc);

                    /* every furrow in this block flings sailors, one breath per k */
                    for (int ii = 0; ii < mc; ii++) {
                        int i = ic + ii;
                        const double *Arow = A + (size_t)i * n;
                        double *prow = plane + (size_t)ii * NC;
                        for (int k = kc; k < kc + kcb; k++) {
                            double a = Arow[k];
                            __m256d va = _mm256_set1_pd(a);
                            const double *Brow = B + (size_t)k * n + jc;
                            int jj = 0;
                            for (; jj + 4 <= nc; jj += 4) {
                                __m256d vb = _mm256_loadu_pd(Brow + jj);
                                __m256d vc = _mm256_loadu_pd(prow + jj);
                                vc = _mm256_fmadd_pd(va, vb, vc);
                                _mm256_storeu_pd(prow + jj, vc);
                            }
                            for (; jj < nc; jj++) prow[jj] += a * Brow[jj];
                        }
                    }
                }

                /* cairns reach final height: count stones into the third table,
                   the plane is left ready to be leveled for the next rank */
                for (int ii = 0; ii < mc; ii++) {
                    int i = ic + ii;
                    memcpy(C + (size_t)i * n + jc,
                           plane + (size_t)ii * NC,
                           (size_t)nc * sizeof(double));
                }
            }
        }

        free(plane);
    }
}
```

Numerically this is the *same* summation order as the reference kernel (k accumulated in increasing order per fixed i,j), so results should match the naive reference to the same rounding as any reordering-free implementation — no accuracy loss is introduced by the scratch-plane detour.

PREDICTION: speedup_vs_blocked = 6.0

(Reasoning behind the number, stated honestly before any run: AVX2 FMA gives up to 4× over scalar double ops, and OpenMP over an 8-core-class machine could give another multiplicative factor, but the kernel is memory-bandwidth bound for large n and the panel/flush scheme adds an extra write-then-copy pass over C that a direct in-place accumulation doesn't pay — so I do not expect the full 4×cores theoretical ceiling. I do not expect this to beat OpenBLAS, which additionally packs A/B into contiguous micro-panels and uses register-blocked multi-accumulator microkernels this version lacks.)

# MEASUREMENT
No tools were available in this session to invoke `kernel_bench` — this response was produced as pure reasoning plus artifact, per the session's explicit instruction that the pipeline (not me) will compile and measure it. I have not fabricated a number here; the prediction above is the only figure I can honestly stand behind until an actual run is reported back.

# VERDICT
Pending real measurement. Honest expectation: a clear win over the naive/blocked reference from AVX2+OpenMP alone, a genuinely different (not textbook) internal organization (disposable panel-scratch accumulation rather than direct in-place C accumulation, driven literally by the cairn/leveling imagery), but very unlikely to beat OpenBLAS since it has no A/B packing and no multi-register microkernel — those would be the next two of the "at most four" improvements if the first measurement comes in below the cache-blocked baseline or well below OpenBLAS.