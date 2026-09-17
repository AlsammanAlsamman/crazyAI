#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    /* Room across the hall: B laid crosswise so that shelf j, rung k
       holds Bt[j][k] = B[k][j] -- column meets row at the doorway. */
    double *Bt = (double *)malloc((size_t)n * n * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        const double *Brow = B + (size_t)k * n;
        for (int j = 0; j < n; j++) {
            Bt[(size_t)j * n + k] = Brow[j];
        }
    }

    const int IB = 4; /* four doorways open onto the same rung together */

    #pragma omp parallel for schedule(dynamic, 8)
    for (int i0 = 0; i0 < n; i0 += IB) {
        int ib = (i0 + IB <= n) ? IB : (n - i0);

        for (int j = 0; j < n; j++) {
            const double *Brow = Bt + (size_t)j * n;
            __m256d vacc[4];
            double acc[4] = {0.0, 0.0, 0.0, 0.0};
            for (int t = 0; t < ib; t++) vacc[t] = _mm256_setzero_pd();

            int k = 0;
            /* the book that never closes: the nest lives in vacc[t]
               for the whole rung, never spilled to memory mid-way */
            for (; k + 4 <= n; k += 4) {
                __m256d vb = _mm256_loadu_pd(Brow + k);
                for (int t = 0; t < ib; t++) {
                    const double *Arow = A + (size_t)(i0 + t) * n;
                    __m256d va = _mm256_loadu_pd(Arow + k);
                    vacc[t] = _mm256_fmadd_pd(va, vb, vacc[t]);
                }
            }

            /* the mirror's face is thrown away the instant its heap is
               named: fold the register to a scalar right away, once */
            for (int t = 0; t < ib; t++) {
                double tmp[4];
                _mm256_storeu_pd(tmp, vacc[t]);
                acc[t] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }

            /* strays outside their own rung count for nothing */
            for (; k < n; k++) {
                double b = Brow[k];
                for (int t = 0; t < ib; t++) {
                    acc[t] += A[(size_t)(i0 + t) * n + k] * b;
                }
            }

            for (int t = 0; t < ib; t++) {
                C[(size_t)(i0 + t) * n + j] = acc[t];
            }
        }
    }

    free(Bt);
}
