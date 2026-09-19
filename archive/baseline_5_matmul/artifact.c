#include <immintrin.h>
#include <string.h>
#include <omp.h>

#define BI 64
#define BK 128
#define BJ 256

void kernel(int n, const double *A, const double *B, double *C) {
    size_t N = (size_t)n;
    memset(C, 0, N * N * sizeof(double));

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < n; ii += BI) {
        int i_max = (ii + BI < n) ? ii + BI : n;
        for (int kk = 0; kk < n; kk += BK) {
            int k_max = (kk + BK < n) ? kk + BK : n;
            for (int jj = 0; jj < n; jj += BJ) {
                int j_max = (jj + BJ < n) ? jj + BJ : n;
                for (int i = ii; i < i_max; i++) {
                    double *Crow = C + (size_t)i * N;
                    const double *Arow = A + (size_t)i * N;
                    for (int k = kk; k < k_max; k++) {
                        double a = Arow[k];
                        __m256d va = _mm256_set1_pd(a);
                        const double *Brow = B + (size_t)k * N;
                        int j = jj;
                        for (; j + 4 <= j_max; j += 4) {
                            __m256d vb = _mm256_loadu_pd(Brow + j);
                            __m256d vc = _mm256_loadu_pd(Crow + j);
                            vc = _mm256_fmadd_pd(va, vb, vc);
                            _mm256_storeu_pd(Crow + j, vc);
                        }
                        for (; j < j_max; j++) {
                            Crow[j] += a * Brow[j];
                        }
                    }
                }
            }
        }
    }
}
