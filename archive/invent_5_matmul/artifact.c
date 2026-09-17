#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    /* "Hang the column-frame crosswise": transpose B once so each
       column becomes a contiguous row-frame running top to bottom.
       This is the one-time cost of re-laying B into index-contiguous
       memory; everything after this never touches B's original layout. */
    double *Bt = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!Bt) {
        /* fallback: plain correct triple loop, never leave C wrong */
        memset(C, 0, (size_t)n * n * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                double a = A[(size_t)i * n + k];
                for (int j = 0; j < n; j++)
                    C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
            }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            Bt[(size_t)j * n + k] = B[(size_t)k * n + j];

    const int JR = 4; /* number of easels (columns) sharing one fixed row-frame at a time */

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *arow = A + (size_t)i * n;   /* fixed row-frame: never moves for this i */
        int j = 0;
        for (; j + JR <= n; j += JR) {
            const double *b0 = Bt + (size_t)(j + 0) * n;
            const double *b1 = Bt + (size_t)(j + 1) * n;
            const double *b2 = Bt + (size_t)(j + 2) * n;
            const double *b3 = Bt + (size_t)(j + 3) * n;

            __m256d strip0 = _mm256_setzero_pd(); /* blank glass strips */
            __m256d strip1 = _mm256_setzero_pd();
            __m256d strip2 = _mm256_setzero_pd();
            __m256d strip3 = _mm256_setzero_pd();

            int k = 0;
            for (; k + 4 <= n; k += 4) {
                __m256d a = _mm256_loadu_pd(arow + k); /* index walks the corridor */
                strip0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(b0 + k), strip0);
                strip1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(b1 + k), strip1);
                strip2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(b2 + k), strip2);
                strip3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(b3 + k), strip3);
            }

            double sum0, sum1, sum2, sum3;
            {
                __m128d lo, hi, s;
                lo = _mm256_castpd256_pd128(strip0); hi = _mm256_extractf128_pd(strip0, 1);
                s = _mm_add_pd(lo, hi); s = _mm_hadd_pd(s, s); sum0 = _mm_cvtsd_f64(s);
                lo = _mm256_castpd256_pd128(strip1); hi = _mm256_extractf128_pd(strip1, 1);
                s = _mm_add_pd(lo, hi); s = _mm_hadd_pd(s, s); sum1 = _mm_cvtsd_f64(s);
                lo = _mm256_castpd256_pd128(strip2); hi = _mm256_extractf128_pd(strip2, 1);
                s = _mm_add_pd(lo, hi); s = _mm_hadd_pd(s, s); sum2 = _mm_cvtsd_f64(s);
                lo = _mm256_castpd256_pd128(strip3); hi = _mm256_extractf128_pd(strip3, 1);
                s = _mm_add_pd(lo, hi); s = _mm_hadd_pd(s, s); sum3 = _mm_cvtsd_f64(s);
            }
            for (; k < n; k++) { /* remainder of the corridor walk */
                double a = arow[k];
                sum0 += a * b0[k]; sum1 += a * b1[k];
                sum2 += a * b2[k]; sum3 += a * b3[k];
            }
            /* finished frame set onto its stand */
            C[(size_t)i * n + j + 0] = sum0;
            C[(size_t)i * n + j + 1] = sum1;
            C[(size_t)i * n + j + 2] = sum2;
            C[(size_t)i * n + j + 3] = sum3;
        }
        for (; j < n; j++) { /* leftover columns, one easel at a time */
            const double *b = Bt + (size_t)j * n;
            __m256d strip = _mm256_setzero_pd();
            int k = 0;
            for (; k + 4 <= n; k += 4)
                strip = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k), _mm256_loadu_pd(b + k), strip);
            __m128d lo = _mm256_castpd256_pd128(strip), hi = _mm256_extractf128_pd(strip, 1);
            __m128d s = _mm_add_pd(lo, hi); s = _mm_hadd_pd(s, s);
            double sum = _mm_cvtsd_f64(s);
            for (; k < n; k++) sum += arow[k] * b[k];
            C[(size_t)i * n + j] = sum;
        }
    }
    free(Bt);
}
