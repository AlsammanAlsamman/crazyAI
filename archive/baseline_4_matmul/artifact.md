# APPROACH

I'm keeping this within the risk budget of a single (untested-by-me) submission: a full GotoBLAS/BLIS-style packed microkernel is the fastest path in principle, but its edge-case handling (partial panels, packing strides, microkernel remainder tiles) is easy to get subtly wrong with no compiler/runtime available to me in this session to catch it. Given the contract demands **exact correctness**, I chose a design that is much harder to get wrong while still attacking every assumption the naive/cache-blocked triple loop makes:

- **Loop order**: `i-j-k` blocked as `ii → jj → kk → i → k → j`, i.e. classic `ikj` (row of A scalar, row of B streamed, row of C accumulated) inside three levels of cache blocking (`IB` for L2/parallel granularity, `JB` for the C/B working tile that should stay resident in L1/L2, `KB` to bound reuse distance). This keeps every access to A, B, C row-major-contiguous — no transposition, no strided loads.
- **Vectorization**: AVX2 + FMA, 4 doubles/vector, unrolled ×4 (16 doubles/iteration) so 4 independent FMA chains hide latency and saturate the FMA ports, with scalar/vector remainder loops so correctness holds for any `n` (not just multiples of 16 or 4).
- **Parallelism**: `#pragma omp parallel for schedule(dynamic)` over the outer row-block loop (`ii`) only. Each thread owns disjoint rows of `C`, so there is no write-write or write-read race and no need for reduction or locking — this is the "one processor holds both matrices" assumption relaxed safely (multiple processors, disjoint output regions, both inputs read-only/shared).
- **Exactness**: every product `A[i,k]*B[k,j]` is still computed once and summed in the same order as the reference triple loop (`k` outer of `j` isn't reordered relative to `j`, and blocking doesn't change the order in which terms are added into a given `C[i,j]`), so floating-point rounding matches the reference bit-for-bit (same summation order as the given minimal example, just vectorized/blocked/parallel).

This does not attempt to beat OpenBLAS's hand-tuned assembly microkernels and packed panels — that requires more machinery than I can safely author blind. It targets "much faster than the cache-blocked triple loop reference," with OpenBLAS-competitiveness as a stretch, not a claim.

# ARTIFACT

```c
#include <immintrin.h>
#include <string.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    size_t N = (size_t)n;
    memset(C, 0, N * N * sizeof(double));

    const int IB = 64;    /* row block: parallel granularity + L2 reuse of C rows   */
    const int JB = 256;   /* column block: keeps C/B tile resident for the k-sweep  */
    const int KB = 256;   /* k block: bounds reuse distance for C row / B rows      */

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < n; ii += IB) {
        int imax = ii + IB < n ? ii + IB : n;
        for (int jj = 0; jj < n; jj += JB) {
            int jmax = jj + JB < n ? jj + JB : n;
            for (int kk = 0; kk < n; kk += KB) {
                int kmax = kk + KB < n ? kk + KB : n;
                for (int i = ii; i < imax; i++) {
                    double * __restrict Crow = C + (size_t)i * N;
                    const double * __restrict Arow = A + (size_t)i * N;
                    for (int k = kk; k < kmax; k++) {
                        double a = Arow[k];
                        __m256d va = _mm256_set1_pd(a);
                        const double * __restrict Brow = B + (size_t)k * N;
                        int j = jj;
                        for (; j + 16 <= jmax; j += 16) {
                            __m256d c0 = _mm256_loadu_pd(Crow + j);
                            __m256d c1 = _mm256_loadu_pd(Crow + j + 4);
                            __m256d c2 = _mm256_loadu_pd(Crow + j + 8);
                            __m256d c3 = _mm256_loadu_pd(Crow + j + 12);
                            __m256d b0 = _mm256_loadu_pd(Brow + j);
                            __m256d b1 = _mm256_loadu_pd(Brow + j + 4);
                            __m256d b2 = _mm256_loadu_pd(Brow + j + 8);
                            __m256d b3 = _mm256_loadu_pd(Brow + j + 12);
                            c0 = _mm256_fmadd_pd(va, b0, c0);
                            c1 = _mm256_fmadd_pd(va, b1, c1);
                            c2 = _mm256_fmadd_pd(va, b2, c2);
                            c3 = _mm256_fmadd_pd(va, b3, c3);
                            _mm256_storeu_pd(Crow + j, c0);
                            _mm256_storeu_pd(Crow + j + 4, c1);
                            _mm256_storeu_pd(Crow + j + 8, c2);
                            _mm256_storeu_pd(Crow + j + 12, c3);
                        }
                        for (; j + 4 <= jmax; j += 4) {
                            __m256d c0 = _mm256_loadu_pd(Crow + j);
                            __m256d b0 = _mm256_loadu_pd(Brow + j);
                            c0 = _mm256_fmadd_pd(va, b0, c0);
                            _mm256_storeu_pd(Crow + j, c0);
                        }
                        for (; j < jmax; j++) {
                            Crow[j] += a * Brow[j];
                        }
                    }
                }
            }
        }
    }
}
```

# PREDICTION: 12

(Interpreting this as: expected speedup factor of this kernel over the given reference `memset` + naive `ikj` cache-blocked triple loop, on a multi-core box at large `n` — roughly 3–4× from AVX2 FMA + unrolling and another ~3–4× from OpenMP parallelism across cores, multiplying to a low-double-digit factor; I do not expect to beat a properly multi-threaded OpenBLAS.)

# MEASUREMENT

No benchmarking tools (`kernel_bench` etc.) were available in this session, so I could not compile or run this kernel myself, nor obtain GFLOP/s or wall-clock numbers against the reference or OpenBLAS. The prediction above is a pre-registered estimate, stated before any measurement, as required — the actual measurement must be performed by the external pipeline that compiles this artifact with `gcc -O3 -march=native -fopenmp -lm`.

# VERDICT

Unverified. The implementation is designed to be safely correct (same summation order as the reference, no races, remainder loops cover all `n`, not just nice multiples) rather than maximally aggressive, because I had no way to compile-test or benchmark it in this session. I expect it to clearly beat the naive cache-blocked triple loop via SIMD+FMA+multicore, but I explicitly do **not** claim it beats OpenBLAS — that would need packed panels and a register-blocked microkernel, which I judged too risky to hand-author without a compiler/test loop to validate against. If the external pipeline's measurement contradicts the prediction, that result should be trusted over this estimate.