# MAPPING

| SEED | World object → Problem object | Silent assumption it breaks |
|---|---|---|
| **SEED 1** — "The tied mirror fuses two facing pebble-piles into a single product heap at each doorway." | *mirror* = one FMA lane; *facing piles* = `A[i][k]` and `B[k][j]` (via the transposed room, see below); *doorway* = the index point where these two scalars meet; *product heap* = the fused multiply-add result, folded straight into a register, never stored as two separate numbers. | "numbers are IEEE doubles and multiply is the primitive" / "one product is one problem" — the mirror doesn't produce a *multiply result* that is later *added*; it names one fused heap. Multiply-add is the primitive, not multiply alone. |
| **SEED 2** — "The cat carries every product heap along the unclosing book's spine into the nest… only when every doorway along that shared rung has poured into the same nest do I press it flat into one cell." | *cat* = the CPU's vector-accumulate hardware; *the book that never closes / its spine* = the k-loop kept alive in a register across the *entire* reduction, never flushed to memory mid-way; *nest* = the accumulator register; *pressing flat into a cell* = the single terminal store to `C[i][j]`. | "a matrix is a two-dimensional grid living in one memory" — the running sum lives in a *third* place (a register/scratch nest), never inside the A/B grids, and touches the C grid exactly once, at the very end. |
| **SEED 3** — "All doorways of the apartment open together along untangling wires… a hundred mirrors catch their hundred pairs in the same breath, and a hundred cats run at once, for the wires never tangle." | *apartment* = the whole compute; *all doorways open together* = every output cell's reduction proceeds concurrently; *wires that never tangle* = provably disjoint write-sets (each thread owns distinct rows of `C`, no locks, no false sharing). | "the output is produced one cell at a time, row by row" — nothing here is sequential; every cell (and every mirror inside it) is scheduled to run in the same breath as every other. |

Read together with the frame narration ("first table's rows laid along shelves, each cell at its own rung" / "second table's rows turned crosswise so column meets row at the doorway"): Room 1 stores `A` row-major, unchanged (shelf = row `i`, rung = column `k`). Room 2 stores **`B` transposed** (shelf = column `j` of `B`, rung = row `k` of `B`) — this is the literal meaning of "turned crosswise so column meets row." The doorway at rung `k` between shelf `i` (room 1) and shelf `j` (room 2) is exactly the term `A[i][k]·B[k][j]`. Summing all rungs into one nest and pressing it into cell `(i,j)` of the third table is precisely `C[i][j] = Σ_k A[i][k]·Bᵗ[j][k]` — a **dot product**, not the outer-product rank-1 update BLAS uses.

# CHOSEN SEED

**SEED 2** — most literal (nest = accumulator register that never round-trips to memory until the whole rung is exhausted) and most different from the known way: OpenBLAS's microkernel is an **outer-product** scheme (broadcast one `A` element, FMA into a strip of `C` accumulators across `k`-blocks). SEED 2, taken literally with the crosswise-room detail, describes the opposite: an **inner-product/dot-product** scheme — pack `B` transposed once, then compute each `C[i][j]` as one uninterrupted dot product held entirely in a register until it's done.

# ASSUMPTION BROKEN

"A matrix is a two-dimensional grid living in one memory" and "the output is produced one cell at a time, row by row." The reduction lives in a third place (a register nest, decoupled from both operand grids and from `C`), and — per SEED 3 — every cell's reduction is scheduled to run concurrently with every other cell's, not sequentially.

# ARTIFACT

Literal object map used to write the code:
- **Memory that stays still**: `A` (room 1, untouched, row-major) and `Bt` (room 2, `B` "turned crosswise" = transposed once, packed).
- **What flows**: products fused into vector registers (`vacc[t]`, the nests-on-the-bed), never written to memory until final.
- **Processor**: one OpenMP thread per block of 4 rows of `A` ("a hundred cats running at once"); inside each, 4 FMA lanes (AVX2, `_mm256_fmadd_pd`) are the "hundred mirrors."
- **Time**: outer loop over row-blocks (parallel, no ordering), then `j`, then the never-closing `k` reduction; only one synchronization point exists — the OpenMP schedule boundary, never per-multiply ("I wait only for the slowest nest, never for the mirror").
- **Discarding stale reflections**: the vector accumulator is horizontally reduced to a scalar exactly once, right after the k-loop ends, never carried forward or reused.
- **Discarding strays**: the `n % 4` tail of `k` is handled by a separate scalar loop so no lane ever reads past index `n-1`.

```c
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    /* Room across the hall: B laid crosswise so that shelf j, rung k
       holds Bt[j][k] = B[k][j] -- column meets row at the doorway. */
    double *Bt = (double *)malloc((size_t)n * n * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        const double *Brow = B + (size_t)k * n;
        for (int j = 0; j < n; j++) {
            Bt[(size_t)j * n + k] = Brow[j];
        }
    }

    const int IB = 4; /* four doorways open onto the same rung together */

    #pragma omp parallel for schedule(dynamic, 8)
    for (int i0 = 0; i0 < n; i0 += IB) {
        int ib = (i0 + IB <= n) ? IB : (n - i0);

        for (int j = 0; j < n; j++) {
            const double *Brow = Bt + (size_t)j * n;
            __m256d vacc[4];
            double acc[4] = {0.0, 0.0, 0.0, 0.0};
            for (int t = 0; t < ib; t++) vacc[t] = _mm256_setzero_pd();

            int k = 0;
            /* the book that never closes: the nest lives in vacc[t]
               for the whole rung, never spilled to memory mid-way */
            for (; k + 4 <= n; k += 4) {
                __m256d vb = _mm256_loadu_pd(Brow + k);
                for (int t = 0; t < ib; t++) {
                    const double *Arow = A + (size_t)(i0 + t) * n;
                    __m256d va = _mm256_loadu_pd(Arow + k);
                    vacc[t] = _mm256_fmadd_pd(va, vb, vacc[t]);
                }
            }

            /* the mirror's face is thrown away the instant its heap is
               named: fold the register to a scalar right away, once */
            for (int t = 0; t < ib; t++) {
                double tmp[4];
                _mm256_storeu_pd(tmp, vacc[t]);
                acc[t] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }

            /* strays outside their own rung count for nothing */
            for (; k < n; k++) {
                double b = Brow[k];
                for (int t = 0; t < ib; t++) {
                    acc[t] += A[(size_t)(i0 + t) * n + k] * b;
                }
            }

            for (int t = 0; t < ib; t++) {
                C[(size_t)(i0 + t) * n + j] = acc[t];
            }
        }
    }

    free(Bt);
}
```

PREDICTION: speedup_vs_blocked = 5

Reasoning behind the number: multi-core OpenMP parallelism (unknown core count, conservatively assume a modest fraction of ideal scaling due to memory-bandwidth sharing) plus AVX2/FMA vectorization plus a 4-row register-blocking factor that quarters the re-reads of each `Bt` row — set against the packing overhead of transposing `B` (O(n²), one-time) and the fact that a *dot-product* scheme still re-streams `A` rows from memory for every block of `j`, unlike BLAS's much deeper cache hierarchy of blocking. I do not expect this to beat OpenBLAS, which additionally blocks for L2/L3 and typically runs closer to peak FLOPs.

# MEASUREMENT

No tools were available in this session (`kernel_bench`, `kernel_contract`, etc. could not be invoked), so **no empirical measurement was taken**. I am not going to invent a number here — per the stated discipline, absence of a measurement must be reported as plainly as a result would be. This kernel and its stated prediction (`speedup_vs_blocked = 5`) are handed off for the pipeline to actually compile and run with `kernel_bench`; the prediction stands or falls on that run, not on my say-so.

# VERDICT

Unverified pending an actual `kernel_bench` run. The design is a literal, checkable translation of SEED 2 (dot-product-per-cell with a never-flushed register nest, `B` packed crosswise/transposed) combined with SEED 1 (FMA as the fused mirror-primitive) and SEED 3 (fully independent, race-free parallel cells) — genuinely different in structure from OpenBLAS's outer-product microkernel, so it is a fair test of whether the native's picture yields something real rather than a relabeled standard trick. If the pipeline measures it, the honest outcomes are: (a) it beats the cache-blocked loop by roughly the predicted margin — the dot-product/transpose idea has real legs; (b) it beats it by much less, or not at all — packing overhead or bandwidth-bound row-streaming dominates and the literal translation underperforms; either result should be reported exactly as measured, not adjusted toward the prediction.