#include <immintrin.h>
#include <string.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    const int BK = 256;  /* span of the current pooled before the bank moves on */
    const int BJ = 512;  /* how many coats (columns) stay resident while pooling */
    const int IB = 4;    /* does burning abreast: one ring pull lights 4 antler-fires */

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < n; ii += IB) {
        int i_max = (ii + IB <= n) ? ii + IB : n;
        int ib = i_max - ii;

        for (int kk = 0; kk < n; kk += BK) {
            int k_max = (kk + BK <= n) ? kk + BK : n;

            for (int jj = 0; jj < n; jj += BJ) {
                int j_max = (jj + BJ <= n) ? jj + BJ : n;

                for (int k = kk; k < k_max; k++) {
                    const double *Bk = B + (size_t)k * n;
                    double a[4];
                    for (int t = 0; t < ib; t++)
                        a[t] = A[(size_t)(ii + t) * n + k];

                    int j = jj;
                    for (; j + 4 <= j_max; j += 4) {
                        __m256d vb = _mm256_loadu_pd(Bk + j);
                        for (int t = 0; t < ib; t++) {
                            double *Ci = C + (size_t)(ii + t) * n;
                            __m256d va = _mm256_set1_pd(a[t]);
                            __m256d vc = _mm256_loadu_pd(Ci + j);
                            vc = _mm256_fmadd_pd(va, vb, vc);
                            _mm256_storeu_pd(Ci + j, vc);
                        }
                    }
                    for (; j < j_max; j++) {
                        double bvj = Bk[j];
                        for (int t = 0; t < ib; t++)
                            C[(size_t)(ii + t) * n + j] += a[t] * bvj;
                    }
                }
            }
        }
    }
}
