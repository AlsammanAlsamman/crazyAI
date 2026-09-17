# MAPPING

**SEED 1 — "Knotting hairs at each crossing to set that crossing's resistance from one table's values."**

| World object | Problem object |
|---|---|
| Row-hair (one per row of first table) | Row index i of A |
| Column-hair (one per column of second table) | Column index j of B |
| Crossing of row-hair × column-hair | A candidate (i,j,k) product site, gated by k |
| Knot tightness, set by "shared-index numbers" | A per-k gate/selector that decides whether A[i,k] and B[k,j] are allowed to meet at this layer of the mound |
| Loose vs. hard knot | High vs. low conductance = how much of the product is let through |

Assumption broken: **"n³ multiplications are needed."** A resistance that is *set*, rather than *computed*, from the shared index suggests some k-terms can be gated off before ever forming a product (skip/prune), not merely summed after being formed.

**SEED 2 — "Holding pressure down row-hairs against current along column-hairs so their meeting yields a product."**

| World object | Problem object |
|---|---|
| Pressure (voltage) held along a row-hair | A[i,k], broadcast, constant while it acts on the whole column-hair |
| Current carried along a column-hair | B[k, j] for j = 0..n-1, i.e. a whole row of B |
| The crossing itself | An outer-product update: one k produces a full i×j sheet of contributions at once, not one cell at a time |
| "current that gets through is their product, no more, no less" | Exact fused multiply, no accumulation folded in at that instant |

Assumption broken: **"the output is produced one cell at a time, row by row."** A crossing here updates a whole plane of cells (all i, all j) for a fixed k — the natural unit of work is a *k-layer*, not a *cell*.

**SEED 3 — "Draining crossings into still pools gauged by stitched pillows until each pool stops rising."**

| World object | Problem object |
|---|---|
| Pool at the foot of a hair | Accumulator for a block of C, living off to the side, not in "the grid" |
| Pillow/gauge next to the pool | The live register/cache value of that accumulator — watched, not stored back yet |
| "a pool does not move until every crossing has fed it" | No partial write-back to C until the full k-sum for that block is done |
| "cut that hair loose... throw the spent length away" | Once a tile's sum is complete: flush once to C, then discard/reuse the scratch buffer immediately, bounding live memory to the size of one tile forever, independent of n |

Assumption broken: **"a matrix is a two-dimensional grid living in one memory."** The partial C for a tile does not live in the C array at all while it's being built — it lives in a separate, small, ephemeral pool that is thrown away the moment it's read, so the "grid" is only ever touched twice (memset + one flush), never repeatedly read-modify-written in place.

# CHOSEN SEED

SEED 3, driving the buffering strategy, combined mechanically with SEED 2 for the update rule (a k-layer updates a whole tile at once, which is exactly what makes "the pool doesn't move until every crossing has fed it" meaningful — the pool must be big enough to catch a whole tile's worth of crossings per layer, but small enough that it never grows past one tile). SEED 3 is the most literal *and* the most different from "the known way": OpenBLAS's packed microkernel also blocks, but it still write-accumulates into a C micro-tile held in registers during the microkernel — the *novel* part taken literally from the text is refusing to let any partial state exist inside the C array's own memory at all, ever, until the single terminal flush, and discarding the pool the instant it's read rather than reusing it as "the next cell's initial value."

# ASSUMPTION BROKEN

"A matrix is a two-dimensional grid living in one memory" — during accumulation, the relevant slice of C is not the C array; it is a small stack/heap "pool" buffer bounded to tile size (independent of n), memset once, fed by every k-layer, flushed to C exactly once, then discarded. C-memory itself is touched only for the final `memcpy`, never for read-modify-write.

# ARTIFACT

```c
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define MC 64    /* row-hairs per pool  */
#define NC 128   /* column-hairs per pool (multiple of 4 for AVX2) */
#define KC 256   /* mound depth kept live before it must have fully drained */

void kernel(int n, const double *A, const double *B, double *C) {
    #pragma omp parallel
    {
        /* the pool: one per thread, reused tile after tile, never larger
           than MCxNC regardless of n -- "the tangle never outgrows what's
           still unread" */
        double *pool = (double *)aligned_alloc(32, (size_t)MC * NC * sizeof(double));

        #pragma omp for schedule(dynamic) collapse(2)
        for (int i0 = 0; i0 < n; i0 += MC) {
            for (int j0 = 0; j0 < n; j0 += NC) {
                int mc = (i0 + MC <= n) ? MC : (n - i0);
                int nc = (j0 + NC <= n) ? NC : (n - j0);

                /* pool starts empty: no pillow has risen yet */
                memset(pool, 0, (size_t)mc * NC * sizeof(double));

                for (int k0 = 0; k0 < n; k0 += KC) {
                    int kc = (k0 + KC <= n) ? KC : (n - k0);

                    /* one k = one layer of the mound: a single crossing
                       plane feeds the WHOLE (i,j) pool at once */
                    for (int kk = 0; kk < kc; kk++) {
                        int k = k0 + kk;
                        const double *Brow = B + (size_t)k * n + j0; /* current, shared by the layer */

                        for (int ii = 0; ii < mc; ii++) {
                            int i = i0 + ii;
                            double a = A[(size_t)i * n + k];          /* pressure, held for this row-hair */
                            __m256d va = _mm256_set1_pd(a);
                            double *prow = pool + (size_t)ii * NC;

                            int jj = 0;
                            for (; jj + 4 <= nc; jj += 4) {
                                __m256d vb = _mm256_loadu_pd(Brow + jj);
                                __m256d vp = _mm256_loadu_pd(prow + jj);
                                vp = _mm256_fmadd_pd(va, vb, vp);      /* the product, no more no less */
                                _mm256_storeu_pd(prow + jj, vp);
                            }
                            for (; jj < nc; jj++) prow[jj] += a * Brow[jj];
                        }
                    }
                    /* pool keeps rising across k0 panels -- it must not
                       move (be written to C) until every crossing has fed it */
                }

                /* pool has stopped rising: read its level, write it down,
                   then the pool is thrown away (memset next iteration) */
                for (int ii = 0; ii < mc; ii++) {
                    int i = i0 + ii;
                    memcpy(C + (size_t)i * n + j0, pool + (size_t)ii * NC,
                           (size_t)nc * sizeof(double));
                }
            }
        }

        free(pool); /* hair cut loose from the mound for good */
    }
}
```

Note on SEED 1's "wall and puppeteers, for checking only" (Freivalds-style: pick a random vector r, compare C·r against A·(B·r) from two independent random projections): deliberately **not** included in the timed hot path above, exactly as the native insists it's for checking, not computing — folding it in would corrupt the speed measurement. If wanted, it belongs in a separate `#ifdef VERIFY` pass outside `kernel()`.

# PREDICTION: speedup_vs_blocked = 5.0

Reasoning: AVX2 FMA gives ~4x over scalar double throughput; the pool discipline removes the repeated read-modify-write of C that a plain cache-blocked single-threaded loop still performs; OpenMP spreads independent (i0,j0) tiles across cores. Expect most of the win from core count (assume ~8 usable cores) tempered by memory-bandwidth sharing and dynamic-scheduling overhead, so well under linear (8x) but comfortably above a single-threaded cache-blocked reference. I do **not** expect this to beat OpenBLAS/MKL, which additionally pack A/B into contiguous panels and use a register-resident micro-kernel tuned per microarchitecture — this kernel has neither packing nor a true register micro-kernel, only a coarser pool buffer.

# MEASUREMENT

Not performed in this session — no `kernel_bench` / `kernel_contract` tool was available here (the harness explicitly disabled tool calls and asked for reasoning + artifact only, with the pipeline compiling and measuring separately). Reporting this plainly rather than fabricating numbers: **no empirical speedup value can be honestly stated by me right now.** The `speedup_vs_blocked = 5.0` above is a prediction only, to be checked by the pipeline's actual run of `kernel_bench` against the cache-blocked reference (and, separately, against OpenBLAS).

# VERDICT

Unverified. The design is a literal, traceable translation of SEED 3 (pool lives outside the C array, fed once per k-layer, flushed exactly once, discarded) fused with SEED 2 (a crossing = an outer-product layer, not a per-cell dot product), and deliberately keeps SEED 1's verification idea (Freivalds double-projection check) out of the timed path per the native's own framing. If the pipeline's measurement comes in below the predicted 5.0x, the first three things I'd try (of the four allowed improvements) would be: (1) shrink KC / grow MC×NC to fit L2 more precisely per target cache size, (2) unroll the `ii` loop by 2–4 to get register-level micro-tiling instead of relying on the pool array for the innermost accumulation, (3) add explicit prefetch of the next k's `Brow`. I have not yet spent any of those four iterations since no measurement has come back to react to.