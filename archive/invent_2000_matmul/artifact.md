## MAPPING

**SEED 1** — "Two staircases tilted to one matching angle so riser meets riser; the shared index becomes a physical alignment rather than a count."

| World object | Problem object |
|---|---|
| Staircase for table A / table B | Matrix A / matrix B, each seen as a stack of rows (risers) |
| The tilt angle | The way the shared index k is walked through memory — a stride, not a counted loop variable |
| "Riser meets riser" | For a fixed k, A's row-segment at k and B's row at k are brought into the *same* register/cache line at the *same* instant |
| "No step of one flight is a stranger to the step below it on the other" | A[i,k] and B[k,j] for the same k are always fetched together, never independently re-derived |

Breaks: **"the whole sum over the shared index is finished before the next cell is started"** — many cells advance their partial sums together, in lock-step over k, rather than one dot-product finishing before the next starts.

**SEED 2** — "An eye per cell filled with milk to a depth equal to its number; two eyes locking is the multiplication, one drop into a well; wells draining into one basin is the summing."

| World object | Problem object |
|---|---|
| Eye, milk depth | Memory cell holding a scalar |
| Two eyes locking → one drop | A[i,k]*B[k,j] produced as a single fused event |
| Wells draining into one shrine-basin | Reduction over k into C[i,j], done by hardware accumulation (FMA/register), not an explicit counted add |

Breaks: **"one product is one problem; many products are many problems"** — the many (i,k,j) products belonging to one output corner are treated as *one* draining event, not n separate additions the mason must perform.

**SEED 3** — "A doubled sun, sky-face and drowned-face, must agree before a basin is trusted; disagreement empties it and pours again."

| World object | Problem object |
|---|---|
| Doubled sun (sky/drowned) | Two independently derived scalars that must match for the same claimed result |
| Basin disagreeing → tipped out, poured again | Cheap verification (e.g. Freivalds check) triggering a full, safe recomputation |

Breaks: **"every product is computed exactly, once"** — trust is not assumed, it is checked, and only re-earned by redoing the work if the check fails.

## CHOSEN SEED

SEED 1. It is the most literal (the shared index *is* a memory/register alignment, not a loop counter) and the most structurally different starting point from "packed panels + microkernel" — it forces the design question "what does it mean for k to be a place, not a count?" rather than starting from panel-packing directly. SEED 3 is kept as a secondary, cheap correctness gauge riding on top of it (it costs O(n²), not part of the hot loop), since the sketch keeps the doubled sun visible the whole time regardless of which staircase-seed is chosen.

## ASSUMPTION BROKEN

"The whole sum over the shared index is finished before the next cell is started" (SEED 1), and, as a safety net, "every product is computed exactly, once" (SEED 3).

## ARTIFACT

```c
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MR 4      /* rows per micro-tile: one flight of 4 risers            */
#define NR 4      /* cols per micro-tile: width of one AVX riser (256-bit)  */
#define BI 128    /* row super-block: one staircase segment kept in cache   */

static inline int imin(int a, int b) { return a < b ? a : b; }

/* safe, unblocked reference path used only if the doubled-sun check
   (SEED 3) ever disagrees -- kept deliberately simple/trustworthy */
static void safe_gemm(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        const double *Arow = A + (size_t)i * n;
        double *Crow = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            double a = Arow[k];
            const double *Brow = B + (size_t)k * n;
            for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
        }
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;

    /* --- the two staircases tilted to one matching angle ---
       ib walks a super-block of rows of A ("staircase A");
       jt walks a tile of columns of B ("staircase B");
       for every (row tile, col tile) the FULL shared index k is
       walked at once, riser meeting riser, before either flight is
       let go -- k is a physical alignment (a SIMD lane / register
       accumulator), not a count we increment and forget. */
    #pragma omp parallel for schedule(dynamic)
    for (int ib = 0; ib < n; ib += BI) {
        int ihi = imin(ib + BI, n);

        for (int jt = 0; jt < n; jt += NR) {
            int jw = imin(NR, n - jt);

            for (int it = ib; it < ihi; it += MR) {
                int iw = imin(MR, ihi - it);

                if (iw == MR && jw == NR) {
                    __m256d acc0 = _mm256_setzero_pd();
                    __m256d acc1 = _mm256_setzero_pd();
                    __m256d acc2 = _mm256_setzero_pd();
                    __m256d acc3 = _mm256_setzero_pd();
                    const double *a0 = A + (size_t)(it + 0) * n;
                    const double *a1 = A + (size_t)(it + 1) * n;
                    const double *a2 = A + (size_t)(it + 2) * n;
                    const double *a3 = A + (size_t)(it + 3) * n;

                    for (int k = 0; k < n; k++) {
                        __m256d bvec = _mm256_loadu_pd(B + (size_t)k * n + jt);
                        acc0 = _mm256_fmadd_pd(_mm256_set1_pd(a0[k]), bvec, acc0);
                        acc1 = _mm256_fmadd_pd(_mm256_set1_pd(a1[k]), bvec, acc1);
                        acc2 = _mm256_fmadd_pd(_mm256_set1_pd(a2[k]), bvec, acc2);
                        acc3 = _mm256_fmadd_pd(_mm256_set1_pd(a3[k]), bvec, acc3);
                    }

                    _mm256_storeu_pd(C + (size_t)(it + 0) * n + jt, acc0);
                    _mm256_storeu_pd(C + (size_t)(it + 1) * n + jt, acc1);
                    _mm256_storeu_pd(C + (size_t)(it + 2) * n + jt, acc2);
                    _mm256_storeu_pd(C + (size_t)(it + 3) * n + jt, acc3);
                } else {
                    /* edge tile: scalar remainder, still full-k at once */
                    for (int i = it; i < it + iw; i++) {
                        const double *Arow = A + (size_t)i * n;
                        for (int j = jt; j < jt + jw; j++) {
                            double s = 0.0;
                            for (int k = 0; k < n; k++)
                                s += Arow[k] * B[(size_t)k * n + j];
                            C[(size_t)i * n + j] = s;
                        }
                    }
                }
            }
        }
    }

    /* --- the doubled sun: a standing correctness gauge (SEED 3) ---
       sky-face = A*(B*r), drowned-face = C*r for one fixed random r.
       Only if the two faces disagree beyond floating slack do we
       distrust every basin and pour the whole staircase again with
       the safe, unblocked path. This never fires on a correct run;
       it exists to catch a wrong one. */
    double *r  = (double *)malloc((size_t)n * sizeof(double));
    double *Br = (double *)malloc((size_t)n * sizeof(double));
    if (r && Br) {
        unsigned int seed = 0x9e3779b9u;
        for (int i = 0; i < n; i++) {
            seed = seed * 1664525u + 1013904223u;
            r[i] = ((double)(seed & 0xFFFFu) / 65536.0) - 0.5;
        }
        for (int k = 0; k < n; k++) {
            double s = 0.0;
            const double *Brow = B + (size_t)k * n;
            for (int j = 0; j < n; j++) s += Brow[j] * r[j];
            Br[k] = s;
        }
        double maxdiff = 0.0, normref = 0.0;
        for (int i = 0; i < n; i++) {
            const double *Arow = A + (size_t)i * n;
            const double *Crow = C + (size_t)i * n;
            double sA = 0.0, sC = 0.0;
            for (int k = 0; k < n; k++) sA += Arow[k] * Br[k];
            for (int j = 0; j < n; j++) sC += Crow[j] * r[j];
            double d = fabs(sA - sC);
            if (d > maxdiff) maxdiff = d;
            if (fabs(sC) > normref) normref = fabs(sC);
        }
        double tol = 1e-6 * (normref + 1.0) * (double)n;
        if (maxdiff > tol) {
            safe_gemm(n, A, B, C);
        }
    }
    free(r);
    free(Br);
}
```

PREDICTION: speedup_vs_blocked = 6.0

## MEASUREMENT

Not performed in this session — no tool access (`kernel_bench`, `kernel_contract`, etc.) was available here; the instructions for this turn explicitly state the pipeline compiles and measures the artifact separately. I am not fabricating a number for this section: the prediction above is the pre-registered estimate (≈4× from AVX2/FMA register tiling + modest gains from OpenMP threading over the row super-blocks, net of the cache-blocked baseline already getting some auto-vectorization), and the Freivalds check (SEED 3) is expected to add negligible overhead (O(n²), single pass) and to never trigger the `safe_gemm` fallback on a correct build.

## VERDICT

Pending actual measurement by the pipeline. Honestly reported: this artifact is a competent but not radically novel realization of SEED 1 (register-tiled, AVX2, OpenMP-parallel outer-product-per-tile GEMM) — it is close in spirit to what a hand-rolled BLAS microkernel does, just without explicit A/B panel packing, so I do **not** expect it to beat OpenBLAS, only to beat a plain cache-blocked scalar/auto-vectorized triple loop. If measurement shows it does not clear even that bar, the honest next step (not yet taken, per the "at most four improvements" budget) would be to add explicit B-panel packing into a contiguous scratch buffer per row super-block, which is the most direct next literal step of "tilting the two staircases so no step is a stranger to the one below it."