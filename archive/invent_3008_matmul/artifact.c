#include <immintrin.h>
#include <omp.h>
#include <string.h>

void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    const int KB = 256;  /* k-crossings a rib/flat feed before turning to ash   */
    const int JB = 512;  /* width of the district of destination corners a
                             row keeps resident (its basin block)               */

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double *Crow = C + (size_t)i * n;
        for (int jj = 0; jj < n; jj += JB) {
            int jmax = jj + JB < n ? jj + JB : n;
            for (int kk = 0; kk < n; kk += KB) {
                int kmax = kk + KB < n ? kk + KB : n;
                for (int k = kk; k < kmax; k++) {
                    double a = A[(size_t)i * n + k];
                    __m256d va = _mm256_set1_pd(a);
                    const double *Brow = B + (size_t)k * n;
                    int j = jj;
                    /* 8-wide: two lanes of thieves pressing sparks in parallel */
                    for (; j + 8 <= jmax; j += 8) {
                        __m256d vb0 = _mm256_loadu_pd(Brow + j);
                        __m256d vb1 = _mm256_loadu_pd(Brow + j + 4);
                        __m256d vc0 = _mm256_loadu_pd(Crow + j);
                        __m256d vc1 = _mm256_loadu_pd(Crow + j + 4);
                        vc0 = _mm256_fmadd_pd(va, vb0, vc0); /* press + carry into basin */
                        vc1 = _mm256_fmadd_pd(va, vb1, vc1);
                        _mm256_storeu_pd(Crow + j,     vc0);
                        _mm256_storeu_pd(Crow + j + 4, vc1);
                    }
                    for (; j + 4 <= jmax; j += 4) {
                        __m256d vb = _mm256_loadu_pd(Brow + j);
                        __m256d vc = _mm256_loadu_pd(Crow + j);
                        vc = _mm256_fmadd_pd(va, vb, vc);
                        _mm256_storeu_pd(Crow + j, vc);
                    }
                    for (; j < jmax; j++) {
                        Crow[j] += a * Brow[j];   /* lone thief, scalar crossing */
                    }
                }
            }
        }
    }
}
