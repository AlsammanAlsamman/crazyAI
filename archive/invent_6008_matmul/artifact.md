Note: per the explicit override in this task ("No tools are available in this session"), I did not call `kernel_bench` or any symbolic tool — the reasoning below is worked by hand, and measurement is left to the external pipeline.

## MAPPING

| World object | Problem object | Silent assumption it breaks |
|---|---|---|
| **Bone-bead string** ("a number is a string of bone-beads laid side by side, one bead per notch of tenths") | A double `x` decomposed into two float32 limbs `xhi=(float)x`, `xlo=(float)(x-(double)xhi)` (a "double-single" split); a product is reconstructed from *sub-products of the limbs*, not one atomic multiply | **"numbers are IEEE doubles and multiply is the primitive"** — the number stops being one indivisible 64-bit mark and the "multiply" becomes a small schoolbook combination of bead-products |
| **Larva, strip, brain-machine fatigue/memory** | A worker (OpenMP thread) that is handed one row-block of A × one k-block of B, sized so both fit the thread's cache before "the bone dims to gray" (cache/TLB thrashing) | **"the whole sum over the shared index is finished before the next cell is started" / "a matrix is a 2-D grid living in one memory"** — the table is tiled into independent strips, each living briefly in its own (cache) memory, not the whole array at once |
| **Slate of small products already worked out once, matched against it** | The base bead-product `ahi*bhi` computed once and reused as the anchor term; the two correction terms (`ahi*blo`, `alo*bhi`) are added to it rather than re-deriving a 4th cross term (`alo*blo`, judged negligible) | **"every product is computed exactly, once"** — here one product is *reused/looked up* as a building block for a different cell's answer instead of being computed fresh and discarded |
| **Tail-eating loop / snake calendar** | The `k`-reduction loop: `acc[j] += phi + corr` accumulating around the shared index until it "closes on itself" into `C[i][j]` | **"the output is produced one cell at a time, row by row"** — actually this one is *not* broken (it's the ordinary reduction); noted for completeness |
| **Nightingale vs. rose** | A per-strip checksum: one spot-checked cell of the fast (bead) result is compared against a slow, plain, un-split double dot-product; on disagreement beyond tolerance the whole strip is silently redone the plain way | **"every product is computed exactly, once"** (again) — trust in the fast path is *earned*, not assumed; a second, disposable, cheap computation gate-keeps a persistent one |
| Discarded larva ("belly wiped bone-clean") | Per-thread scratch accumulator `acc[]`, `memset` to zero and reused for the next row | — (implementation detail, no assumption broken) |

## CHOSEN SEED

**SEED 1** — "A number is a string of bone-beads laid side by side, never one indivisible mark." This is the only one of the three that touches **"numbers are IEEE doubles and multiply is the primitive"** at all (SEED 2 is cache-blocking, SEED 3 is a verification/reuse scheme layered on the ordinary reduction) — so per the instructions it wins by default, without needing to argue it's "more literal."

## ASSUMPTION BROKEN

`numbers are IEEE doubles and multiply is the primitive`. A double is treated as a two-limb bead-string (hi/lo float32), and `a*b` is reconstructed as `ahi*bhi + ahi*blo + alo*bhi` (Karatsuba-style, dropping the doubly-small `alo*blo`) instead of one hardware `a*b`.

**Honest math check before writing code (this is the finding, not a caveat):** Karatsuba-style bead decomposition only pays for itself when the cost of multiplying grows with operand width (bignum/polynomial multiplication, where a native multiply for the full width doesn't exist in hardware). An IEEE double multiply is a **fixed-cost, O(1) hardware primitive** — there is no "wider" multiply being avoided. Worse: to reconstruct the product *correctly* the three bead cross-terms must be summed in double precision (computing them in float32 would round away exactly the bits the correction terms exist to recover — this only comes "for free" on hardware with a genuine narrow-multiply/wide-accumulate pipeline, e.g. tensor cores or `int8` VNNI, which a plain AVX double-FMA does not have). So the literal, numerically-correct translation of this seed does **3 double multiplies + 2 double adds** to reproduce what `a*b` gets from 1 double multiply — a pure ~3× redundancy with no compensating width or bandwidth benefit (the bead tables are 4 float arrays = 16 B/element, the *same* footprint as the two original double arrays, so there's no cache-traffic win either). I build it anyway, faithfully, per the exercise's rule not to swap in a better idea — and predict accordingly.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Bead-split GEMM.
 * A number is not one indivisible double but a string of two bone-beads:
 *   x = xhi + xlo   (xhi = (float)x, xlo = the float32 residual)
 * A product of two bead-numbers is not one fresh multiply: the base bead
 * product ahi*bhi is worked out once and reused ("slate of small products
 * already worked out") and only the two correction beads ahi*blo, alo*bhi
 * are added; alo*blo is judged negligible and dropped.
 * The great tables are never swallowed whole: larvae (OpenMP threads) each
 * carry one row-strip of A paired with a k-strip of B, sized to the
 * brain-machine's memory/fatigue (cache blocking), and the strip's belly
 * (scratch accumulator) is wiped clean between rows.
 * The shared-index reduction is the tail-eating loop: each partial sum
 * dissolves into the next until it closes on itself into one cell.
 * A nightingale checks one spot-checked cell of every strip's declared
 * total against a slower, plainer (un-split, exact) reckoning of the same
 * numbers; only on disagreement is the strip silently redone exactly.
 */

#ifndef BEAD_BLOCK_I
#define BEAD_BLOCK_I 32
#endif
#ifndef BEAD_BLOCK_K
#define BEAD_BLOCK_K 128
#endif
#ifndef BEAD_TOL
#define BEAD_TOL 1e-9
#endif

static inline void split_bead(double x, float *hi, float *lo) {
    float h = (float)x;
    float l = (float)(x - (double)h);
    *hi = h;
    *lo = l;
}

void kernel(int n, const double *A, const double *B, double *C) {
    const size_t N = (size_t)n;
    if (n <= 0) return;
    memset(C, 0, N * N * sizeof(double));

    float *Ahi = (float*)malloc(N * N * sizeof(float));
    float *Alo = (float*)malloc(N * N * sizeof(float));
    float *Bhi = (float*)malloc(N * N * sizeof(float));
    float *Blo = (float*)malloc(N * N * sizeof(float));

    if (!Ahi || !Alo || !Bhi || !Blo) {
        /* no scratch available: fall back to the plain, slow, always-true reckoning */
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                double a = A[(size_t)i * N + k];
                for (int j = 0; j < n; j++)
                    C[(size_t)i * N + j] += a * B[(size_t)k * N + j];
            }
        free(Ahi); free(Alo); free(Bhi); free(Blo);
        return;
    }

    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < (long)(N * N); idx++) {
        split_bead(A[idx], &Ahi[idx], &Alo[idx]);
        split_bead(B[idx], &Bhi[idx], &Blo[idx]);
    }

    int nBI = (n + BEAD_BLOCK_I - 1) / BEAD_BLOCK_I;

    #pragma omp parallel for schedule(dynamic)
    for (int bi = 0; bi < nBI; bi++) {
        int i0 = bi * BEAD_BLOCK_I;
        int i1 = i0 + BEAD_BLOCK_I; if (i1 > n) i1 = n;

        double *acc = (double*)malloc(N * sizeof(double));

        for (int i = i0; i < i1; i++) {
            memset(acc, 0, N * sizeof(double));

            for (int k0 = 0; k0 < n; k0 += BEAD_BLOCK_K) {
                int k1 = k0 + BEAD_BLOCK_K; if (k1 > n) k1 = n;

                for (int k = k0; k < k1; k++) {
                    double ahi = (double)Ahi[(size_t)i * N + k];
                    double alo = (double)Alo[(size_t)i * N + k];
                    const float *bhi_row = &Bhi[(size_t)k * N];
                    const float *blo_row = &Blo[(size_t)k * N];

                    /* tail-eating loop: partial sums dissolve into the next
                       around the shared index k */
                    #pragma omp simd
                    for (int j = 0; j < n; j++) {
                        double bhi = (double)bhi_row[j];
                        double blo = (double)blo_row[j];
                        double base = ahi * bhi;              /* looked-up base product */
                        double corr = ahi * blo + alo * bhi;  /* correction beads */
                        acc[j] += base + corr;                /* alo*blo dropped: negligible */
                    }
                }
            }

            /* nightingale: spot-check one cell of this strip's declared total
               against a slower, plainer reckoning of the same beads */
            {
                unsigned h = (unsigned)i * 2654435761u;
                int jchk = (int)(h % (unsigned)n);
                double slow = 0.0;
                for (int k = 0; k < n; k++)
                    slow += A[(size_t)i * N + k] * B[(size_t)k * N + jchk];
                double denom = fabs(slow) > 1.0 ? fabs(slow) : 1.0;
                if (fabs(acc[jchk] - slow) / denom > BEAD_TOL) {
                    /* her complaint: redo this strip the plain, trusted way */
                    for (int jj = 0; jj < n; jj++) {
                        double s = 0.0;
                        for (int k = 0; k < n; k++)
                            s += A[(size_t)i * N + k] * B[(size_t)k * N + jj];
                        acc[jj] = s;
                    }
                }
            }

            memcpy(&C[(size_t)i * N], acc, N * sizeof(double));
        }

        free(acc);
    }

    free(Ahi); free(Alo); free(Bhi); free(Blo);
}
```

PREDICTION: speedup_vs_blocked = 1.8

**Reasoning behind the number** (stated before any measurement, as required): the bead mechanism itself is a **structural ~3× redundancy** (3 double multiply-adds reconstructing what 1 double FMA already gives exactly, with no width or bandwidth compensation — the bead tables cost the same 16 B/element as the original doubles). The only lever left that can plausibly beat a single-core cache-blocked reference is the OpenMP parallelism over row-strips ("many larvae"). Assuming a modest multi-core machine (~8 threads) and a single-threaded blocked baseline, net ≈ threads / arithmetic-penalty, discounted for bandwidth contention and the O(n²) nightingale overhead → ≈1.8×. If the baseline reference is itself multi-threaded, I expect this to invert to well below 1 — that would be the honest failure mode of this literal translation, not a bug to hide.

## MEASUREMENT

Not run in this session — no tools (including `kernel_bench`) are available here. The kernel above is the literal, complete artifact; the external pipeline will compile it (`gcc -O3 -march=native -fopenmp -lm`) and measure `speedup_vs_blocked` against the prediction of 1.8.

## VERDICT

The literal translation of "a number is a string of bone-beads" onto an IEEE double, done *correctly* (bead cross-terms summed at double precision, since float32 sub-multiplies would silently discard the very bits the beads exist to recover), exposes a genuine, unglamorous fact rather than a speed trick: **a hardware double multiply is already an O(1) primitive**, so Karatsuba-style bead decomposition — which only earns its keep when the width being avoided has a super-linear native cost (bignums, polynomials) — buys nothing here and costs ~3× the arithmetic. Any win this kernel shows will come entirely from the seed-2 element (larvae/strips → OpenMP threading), not from the seed-1 mechanism itself; if the baseline is already parallel, I expect this kernel to plainly lose, and that negative result should be reported as such rather than reinterpreted as success.