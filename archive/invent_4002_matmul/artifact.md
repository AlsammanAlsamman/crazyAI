# MAPPING

| Seed | World-object | Problem-object | Assumption broken |
|---|---|---|---|
| **Seed 1** – funnel-throat scaling | stamped coin-stone number; funnel throat width; poured sand | `A[i][k]`; `B[k][j]`; the scalar product `A[i][k]*B[k][j]` | "one product is one problem; many products are many problems" — the pour is a continuous *measured cup*, i.e. many matched stone/funnel pairs get poured in one motion (a vector lane), not counted one at a time. |
| **Seed 2** – marble tiling / stampede | field cut into 40×40 marble tiles; a stampede dragging one tile of stones under one tile of funnels | blocking A and B into `BI×BK` / `BK×BJ` sub-blocks; one core (or thread team) owns one tile-pair | "a matrix is a two-dimensional grid living in one memory" and "one processor holds both matrices" — no single processor ever sees more than one tile of each table at once. |
| **Seed 3** – ledger / discarded tiles / one-time stamped coin | bowl; librarian's running mark; spent tiles hauled to shade and never returning; mason weighing each bowl exactly once | a private accumulator `acc[i][j]`, decoupled from C's real memory; A/B tile values are read once and never revisited; C in memory is written exactly once, after all k has settled | "the whole sum over the shared index is finished before the next cell is started" *and*, more sharply, the hidden corollary that **C lives in the same grid memory throughout the accumulation** — here the running sum lives in a third, private, disposable field, and the real C is touched with a single write, never a read-modify-write. |

# CHOSEN SEED
**Seed 3** — the ledger/bowl/one-time-stamp seed. It is the most literal (bowl = accumulator, spent stone/funnel = discarded operand, fresh coin = one-shot store) and the most different from "cache-blocked triple loop" in spirit: the standard blocked loop still does `C[i][j] += ...` directly into C's memory on every k-block, relying on the cache to make that cheap. Seed 3 explicitly forbids ever reading C back — the sum is kept in a private "third field" until it is completely settled, then stamped once.

# ASSUMPTION BROKEN
"The whole sum over the shared index is finished before the next cell is started" is generalized to block granularity, and the deeper hidden assumption — that C's own memory cells are the place where accumulation happens — is broken: accumulation happens in a disposable private buffer; C memory is write-only, once.

# ARTIFACT

```c
#include <string.h>
#include <immintrin.h>
#include <omp.h>

/* marble-tile sizes: "forty stones by forty funnels" rounded to
   SIMD-friendly, cache-friendly dimensions */
#define BI 32   /* rows of the coin-stone tile      -> accumulator rows   */
#define BJ 64   /* funnel-lattice width, mult. of 4 -> AVX2 lanes         */
#define BK 256  /* how many funnels/stones we drag through before resting */

void kernel(int n, const double *A, const double *B, double *C) {
    int nBI = (n + BI - 1) / BI;
    int nBJ = (n + BJ - 1) / BJ;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < nBI; bi++) {
        for (int bj = 0; bj < nBJ; bj++) {

            int i0 = bi * BI, i1 = i0 + BI; if (i1 > n) i1 = n;
            int j0 = bj * BJ, j1 = j0 + BJ; if (j1 > n) j1 = n;
            int ib = i1 - i0, jb = j1 - j0;

            /* the third field of bowls: private, never the same memory
               as A, B, or the final C — a stampede (this thread) owns it
               alone, so no ledger is ever contended */
            double acc[BI][BJ];
            for (int ii = 0; ii < ib; ii++)
                for (int jj = 0; jj < jb; jj++)
                    acc[ii][jj] = 0.0;

            /* drag one tile of stones under one tile of funnels at a time */
            for (int k0 = 0; k0 < n; k0 += BK) {
                int k1 = k0 + BK; if (k1 > n) k1 = n;

                for (int ii = 0; ii < ib; ii++) {
                    int i = i0 + ii;
                    double *accrow = &acc[ii][0];

                    for (int k = k0; k < k1; k++) {
                        /* one coin-stone's stamped number, read once,
                           then it is chest-junk */
                        double a = A[(size_t)i * n + k];
                        __m256d avec = _mm256_set1_pd(a);
                        const double *Brow = B + (size_t)k * n + j0;

                        int jj = 0;
                        for (; jj + 4 <= jb; jj += 4) {
                            /* the funnel throat scales the pour; the
                               librarian adds it to the running mark    */
                            __m256d bvec = _mm256_loadu_pd(Brow + jj);
                            __m256d cvec = _mm256_loadu_pd(accrow + jj);
                            cvec = _mm256_fmadd_pd(avec, bvec, cvec);
                            _mm256_storeu_pd(accrow + jj, cvec);
                        }
                        for (; jj < jb; jj++)
                            accrow[jj] += a * Brow[jj];
                    }
                }
                /* this k-tile of stones and funnels never returns;
                   only the bowls' running mark survives */
            }

            /* the mason weighs every bowl exactly once and stamps the
               fresh coin into the new table — a single write, no
               read-modify-write of C is ever performed */
            for (int ii = 0; ii < ib; ii++)
                memcpy(C + (size_t)(i0 + ii) * n + j0, &acc[ii][0],
                       (size_t)jb * sizeof(double));
        }
    }
}
```

PREDICTION: speedup_vs_blocked = 5.0

# MEASUREMENT
No `kernel_bench` / `kernel_contract` tools were reachable in this session (none were exposed to me here — only Docs/Gmail/Drive/Slack tools were available). Per the task instructions for this situation, I am reporting the prediction and artifact only; the stated pipeline is expected to compile with `gcc -O3 -march=native -fopenmp -lm` and run `kernel_bench` externally. I have not fabricated a measured number.

# VERDICT
Unmeasured by me. Honest expectation, stated plainly: the write-once accumulator plus AVX2 FMA plus OpenMP tiling should beat a naive/lightly-blocked reference by a solid margin (multi-core + vectorization + eliminated C read-modify-write traffic), so the ~5x prediction is plausible against a *single-threaded* cache-blocked baseline, less so if the baseline is already multithreaded. I do **not** expect this to beat OpenBLAS — it lacks A/B panel packing, register-level micro-kernel unrolling beyond 4-wide AVX2, and any NUMA/thread-affinity tuning, all of which OpenBLAS has. That gap should be reported as-is once actually measured, not minimized.