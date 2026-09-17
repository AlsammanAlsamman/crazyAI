#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;
    size_t N = (size_t)n;

    /* "every cell has already been cut into a pipe": pre-cut the column-reeds
       of B into contiguous rows (Bt) so the sheaf (row-reed + column-reed)
       can be carried to the desk without straying through strided memory. */
    double *Bt = (double *)malloc(N * N * sizeof(double));
    if (!Bt) {
        memset(C, 0, N * N * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                double a = A[(size_t)i * N + k];
                for (int j = 0; j < n; j++) C[(size_t)i * N + j] += a * B[(size_t)k * N + j];
            }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        const double *Brow = B + (size_t)k * N;
        for (int j = 0; j < n; j++) Bt[(size_t)j * N + k] = Brow[j];
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *Arow = A + (size_t)i * N;
        double *Crow = C + (size_t)i * N;

        for (int j = 0; j < n; j++) {
            const double *Bcol = Bt + (size_t)j * N;

            /* the flock: four independent accumulators, no bow-stroke
               waits on the one before it */
            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            __m256d acc2 = _mm256_setzero_pd();
            __m256d acc3 = _mm256_setzero_pd();

            int k = 0;
            for (; k + 16 <= n; k += 16) {
                acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k),      _mm256_loadu_pd(Bcol + k),      acc0);
                acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 4),  _mm256_loadu_pd(Bcol + k + 4),  acc1);
                acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 8),  _mm256_loadu_pd(Bcol + k + 8),  acc2);
                acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 12), _mm256_loadu_pd(Bcol + k + 12), acc3);
            }

            /* the flock converges to one voice */
            __m256d acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));

            for (; k + 4 <= n; k += 4)
                acc = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k), _mm256_loadu_pd(Bcol + k), acc);

            __m128d lo   = _mm256_castpd256_pd128(acc);
            __m128d hi   = _mm256_extractf128_pd(acc, 1);
            __m128d sum2 = _mm_add_pd(lo, hi);
            __m128d sum1 = _mm_hadd_pd(sum2, sum2);
            double cry = _mm_cvtsd_f64(sum1);

            for (; k < n; k++) cry += Arow[k] * Bcol[k];   /* stragglers of the flock */

            Crow[j] = cry;   /* write into the matching cell before the ink dries */
        }
    }

    free(Bt);   /* release the flock; the prism stays clear for the next sheaf */
}
