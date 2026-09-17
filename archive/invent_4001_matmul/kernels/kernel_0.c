#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static void kernel_fallback(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i * n + k];
            const double *Brow = B + (size_t)k * n;
            double *Crow = C + (size_t)i * n;
            for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
        }
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;

    /* Plant the second avenue: replant B's columns as contiguous rows
       (chalk/scratch memory, not stone) so a "south-house" is a
       contiguous run of doubles, just like an "east-house" row of A. */
    double *Bt = (double *)malloc((size_t)n * n * sizeof(double));
    if (!Bt) { kernel_fallback(n, A, B, C); return; }

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        const double *Brow = B + (size_t)k * n;
        for (int j = 0; j < n; j++) Bt[(size_t)j * n + k] = Brow[j];
    }

    /* Every (i,j) crossing is an independent reflection: let all cores
       take crossings in whatever order they reach them ("never one
       drawn before another"). Within one crossing, chalk each
       neighbouring-suns pull (a*b) into a private, instantly-forgettable
       scratch accumulator (SIMD registers), sum orbit by orbit over the
       shared index k, then cut the finished sum permanently into the
       new house of C exactly once. */
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n; i++) {
        const double *Arow = A + (size_t)i * n;
        double *Crow = C + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double *Bcol = Bt + (size_t)j * n;
            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            int k = 0;
            int limit = n - (n % 8);
            for (; k < limit; k += 8) {
                __m256d a0 = _mm256_loadu_pd(Arow + k);
                __m256d b0 = _mm256_loadu_pd(Bcol + k);
                acc0 = _mm256_fmadd_pd(a0, b0, acc0);
                __m256d a1 = _mm256_loadu_pd(Arow + k + 4);
                __m256d b1 = _mm256_loadu_pd(Bcol + k + 4);
                acc1 = _mm256_fmadd_pd(a1, b1, acc1);
            }
            __m256d acc = _mm256_add_pd(acc0, acc1);
            double tmp[4];
            _mm256_storeu_pd(tmp, acc);
            double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            for (; k < n; k++) sum += Arow[k] * Bcol[k];   /* wipe the tablet: */
            Crow[j] = sum;                                  /* stone, cut once */
        }
    }

    free(Bt);
}
