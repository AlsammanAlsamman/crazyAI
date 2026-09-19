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
