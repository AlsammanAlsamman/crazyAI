# MAPPING

**SEED 1 — "Wells sunk at the crossings of two stacked floors carry the shared index."**

| World object | Problem object |
|---|---|
| bowler / vessel of the working | the whole `kernel(n,A,B,C)` call |
| lower floor, lattice, one hollow per number of the first table | matrix A, one hollow = `A[i*n+k]` |
| upper floor stacked above, **cut crosswise** | matrix B **re-laid-out transposed** into `Bt[j*n+k] = B[k*n+j]` |
| crossing of a lower hollow and an upper hollow | one output cell `(i,j)` |
| a well per step of the shared index, sunk at that crossing | one multiply site along `k` for that `(i,j)` |
| two draughts poured in, one from each floor | the read of `A[i][k]` and `Bt[j][k]` |
| draughts thicken into one denser draught | the scalar product `A[i][k]*B[k][j]` |
| all wells of a fleet feed a shared standpipe, settle by weight, add without touching | `n` partial products reduced by hardware add, not one serial running accumulator |
| owl waits until the stomach is full, then calls | the whole `k`-reduction for `(i,j)` finishes before any write |
| owl pours into the third table | `C[i][j] = sum` (single store) |

Silent assumption broken: **#8** ("one product is one problem; many products are many problems") — every `(i,k,j)` triple gets its own independent well, i.e. the algorithm is written as n³ independent multiply-problems merged afterward, not a single fused accumulate stream. It also nicks **#3** ("a matrix is a 2‑D grid living in one memory") — the upper floor is explicitly *re-cut* into a second physical layout (B is transposed into new memory) rather than read in its native layout.

**SEED 2 — "Oxygen draughts forced together down one well thicken into their product."**

| World object | Problem object |
|---|---|
| oxygen draught | an operand double |
| forcing two draughts down a narrow shaft | the multiply |
| "they do not mix evenly, they thicken" | result is a genuinely different, denser quantity (product, not blend) |
| "that is oxygen's nature here" | multiply is a hardware-given primitive |

Honestly, this seed mostly *confirms* assumption **#5** rather than breaking it — I won't pretend otherwise. Its only real payload is that the thickening happens *independently in each well*, which is the same independence claim as SEED 1, not a new mechanism.

**SEED 3 — "An owl calling over a settled stomach seals a fleet's sum, leaving tea that is never cleared away."**

| World object | Problem object |
|---|---|
| stomach beneath the floor | accumulator for one `(i,j)` |
| owl, one per fleet | task responsible for that cell |
| owl doesn't stir until every draught settled | don't write `C[i][j]` until all k-contributions are summed |
| tea stain that will not lift | a persistent completion marker, checked before redoing work |

Breaks **#1** ("output produced one cell at a time, row by row") — fleets/owls are independent and can finish in any order, driven by data readiness, not row-major sequence. But its most novel content — the checkpoint/never-redo mark — doesn't actually translate into a speed technique for *this* contract: a single one-shot call with no prior state and `C` freely overwritable has nothing to memoize against. I'm flagging that honestly rather than inventing a use for it.

# CHOSEN SEED

**SEED 1.** It is the most literal (it names an explicit physical action — re-cutting the second floor "crosswise" — that translates directly into a concrete code decision: transpose B) and it is structurally the most different from the known way: OpenBLAS packs panels and streams in `ikj` order for reuse; SEED 1 mandates a `(i,j)`-dot-product order with an explicit second, transposed copy of B, which is the opposite loop nest from panel-packed GEMM.

# ASSUMPTION BROKEN

**#8** (products treated as many independent problems, not folded into one accumulate stream) and **#3** (B is deliberately re-laid-out into a second memory image, "crosswise," rather than read in its given layout).

# ARTIFACT

```c
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HAVE_AVX2_FMA 1
#endif

/* C = A*B, row-major n x n. Literal translation of the bowler:
   - "upper floor cut crosswise": B is repacked transposed into Bt so a
     well's two draughts (A row, Bt row) are both contiguous along k.
   - "one well per step of the shared index, one fleet per (i,j)": the
     inner loop sinks n independent product-wells for each output cell.
   - "several standpipes settle into one stomach without my touching
     them": four independent FMA accumulators (not one serial running
     sum) absorb the products, then are reduced once.
   - "the owl doesn't stir until the stomach is full, then calls": the
     full k-reduction for (i,j) completes before the single store to C.
*/
void kernel(int n, const double *A, const double *B, double *C) {
    double *Bt = (double *)malloc((size_t)n * n * sizeof(double));
    if (!Bt) {
        memset(C, 0, (size_t)n * n * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                double a = A[(size_t)i * n + k];
                for (int j = 0; j < n; j++) C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
            }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            Bt[(size_t)j * n + k] = B[(size_t)k * n + j];

    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n; i++) {
        const double *arow = A + (size_t)i * n;
        double *crow = C + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double *brow = Bt + (size_t)j * n;
            double sum;
#ifdef HAVE_AVX2_FMA
            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            __m256d acc2 = _mm256_setzero_pd();
            __m256d acc3 = _mm256_setzero_pd();
            int k = 0;
            for (; k + 16 <= n; k += 16) {
                acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k),      _mm256_loadu_pd(brow + k),      acc0);
                acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k + 4),  _mm256_loadu_pd(brow + k + 4),  acc1);
                acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k + 8),  _mm256_loadu_pd(brow + k + 8),  acc2);
                acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k + 12), _mm256_loadu_pd(brow + k + 12), acc3);
            }
            __m256d acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
            for (; k + 4 <= n; k += 4)
                acc = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k), _mm256_loadu_pd(brow + k), acc);
            double buf[4];
            _mm256_storeu_pd(buf, acc);
            sum = buf[0] + buf[1] + buf[2] + buf[3];
            for (; k < n; k++) sum += arow[k] * brow[k];
#else
            sum = 0.0;
            for (int k = 0; k < n; k++) sum += arow[k] * brow[k];
#endif
            crow[j] = sum;
        }
    }
    free(Bt);
}
```

Design path (four steps, as allowed):
1. Naive `ijk` dot product, B read column-wise (stride n) — rejected mentally, this is exactly the stride penalty cache-blocking exists to avoid.
2. Add the transpose `Bt` ("cut crosswise") so both operand rows are contiguous — the single biggest expected win.
3. Vectorize with one AVX2 FMA accumulator — correct but the serial FMA dependency chain (≈4–5 cycle latency) throttles throughput.
4. Split into four independent accumulators ("several standpipes settling into one stomach without touching them") to hide FMA latency, plus `omp parallel for` with dynamic scheduling across `i` ("one owl per fleet," independent completions) — final artifact above.

PREDICTION: speedup_vs_blocked = 2.0

# MEASUREMENT

No tools were available in this session (`kernel_bench` etc. could not actually be invoked here, per the session note), so I have **not** run the benchmark myself — I'm reporting that plainly rather than fabricating a number. The prediction above is a pre-registered estimate for the pipeline to check against a real cache-blocked triple-loop baseline; I have not seen or influenced that measurement.

# VERDICT

Pending: expected to beat a cache-blocked triple loop (contiguous SIMD dot products + 4-way ILP + multi-core parallelism over independent fleets should comfortably clear a modestly-tiled reference), but **not** expected to beat OpenBLAS — this kernel has no register/L1/L2/L3-aware panel packing or hand-tuned microkernel, which is where Goto-style GEMM gets most of its edge over any single-well, dot-product formulation. If measurement contradicts the 2.0x prediction, that should be reported as plainly as if it confirmed it, not adjusted after the fact.