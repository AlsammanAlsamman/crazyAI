#include <immintrin.h>
#include <string.h>
#include <stdlib.h>

void kernel(int n, const double *A, const double *B, double *C) {
    size_t N = (size_t)n;
    if (N == 0) return;

    /* the wall's second face: B re-laid so a "column" becomes contiguous,
       so a grow can draw sap from both faces without crossing strides */
    double *Bt = (double *)malloc(N * N * sizeof(double));
    if (!Bt) {
        /* fallback: plain correct triple loop, never leave C wrong */
        memset(C, 0, N * N * sizeof(double));
        for (size_t i = 0; i < N; i++)
            for (size_t k = 0; k < N; k++) {
                double a = A[i * N + k];
                for (size_t j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
            }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (long k = 0; k < (long)N; k++)
        for (size_t j = 0; j < N; j++)
            Bt[j * N + (size_t)k] = B[(size_t)k * N + j];

    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < (long)N; ii++) {
        size_t i = (size_t)ii;
        const double *arow = A + i * N;
        double *crow = C + i * N;

        for (size_t j = 0; j < N; j++) {
            const double *brow = Bt + j * N;   /* the paired column, now a face of its own */

            /* eight independent baskets-in-progress: many grows fruit concurrently,
               none is summed into another until the whole basket is weighed */
            __m256d acc0 = _mm256_setzero_pd(), acc1 = _mm256_setzero_pd();
            __m256d acc2 = _mm256_setzero_pd(), acc3 = _mm256_setzero_pd();
            __m256d acc4 = _mm256_setzero_pd(), acc5 = _mm256_setzero_pd();
            __m256d acc6 = _mm256_setzero_pd(), acc7 = _mm256_setzero_pd();

            size_t k = 0;
            for (; k + 32 <= N; k += 32) {
                acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k),    _mm256_loadu_pd(brow+k),    acc0);
                acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+4),  _mm256_loadu_pd(brow+k+4),  acc1);
                acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+8),  _mm256_loadu_pd(brow+k+8),  acc2);
                acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+12), _mm256_loadu_pd(brow+k+12), acc3);
                acc4 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+16), _mm256_loadu_pd(brow+k+16), acc4);
                acc5 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+20), _mm256_loadu_pd(brow+k+20), acc5);
                acc6 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+24), _mm256_loadu_pd(brow+k+24), acc6);
                acc7 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+28), _mm256_loadu_pd(brow+k+28), acc7);
            }
            for (; k + 4 <= N; k += 4)
                acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k), _mm256_loadu_pd(brow+k), acc0);

            /* weigh the whole basket at once, in one reduction, not incrementally */
            __m256d s01 = _mm256_add_pd(acc0, acc1);
            __m256d s23 = _mm256_add_pd(acc2, acc3);
            __m256d s45 = _mm256_add_pd(acc4, acc5);
            __m256d s67 = _mm256_add_pd(acc6, acc7);
            __m256d total = _mm256_add_pd(_mm256_add_pd(s01, s23), _mm256_add_pd(s45, s67));

            double buf[4];
            _mm256_storeu_pd(buf, total);
            double s = (buf[0] + buf[1]) + (buf[2] + buf[3]);

            for (; k < N; k++) s += arow[k] * brow[k];  /* leftover grows, planted last */

            crow[j] = s;   /* dropped once, whole, onto the third neutral wall */
        }
    }

    free(Bt);
}
