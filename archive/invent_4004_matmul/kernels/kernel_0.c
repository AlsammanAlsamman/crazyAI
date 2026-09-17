#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#ifndef IB_ROWS
#define IB_ROWS 64
#endif

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;

#if defined(__AVX2__) && defined(__FMA__)
    const int n4 = n - (n % 4); /* aligned j-bound for 4-wide vector lanes */

    #pragma omp parallel for schedule(dynamic)
    for (int ib = 0; ib < n; ib += IB_ROWS) {
        int i_end = ib + IB_ROWS < n ? ib + IB_ROWS : n;

        /* B column-panel B[*, j..j+3] ("the column of bone circles") is
           reused across every row i in this block ("a thousand chests
           tick at once, side by side") before we move to the next
           column-panel. Nothing is packed/copied -- read straight from
           the original table. */
        for (int j = 0; j < n4; j += 4) {
            for (int i = ib; i < i_end; i++) {
                const double *Ai = A + (size_t)i * n;
                __m256d acc0 = _mm256_setzero_pd(); /* two independent   */
                __m256d acc1 = _mm256_setzero_pd(); /* "candles" -> ILP  */
                int k = 0;
                int k2 = n - (n % 2);
                for (; k < k2; k += 2) {
                    __m256d a0 = _mm256_set1_pd(Ai[k]);
                    __m256d a1 = _mm256_set1_pd(Ai[k + 1]);
                    __m256d b0 = _mm256_loadu_pd(B + (size_t)k * n + j);
                    __m256d b1 = _mm256_loadu_pd(B + (size_t)(k + 1) * n + j);
                    acc0 = _mm256_fmadd_pd(a0, b0, acc0); /* burn candle */
                    acc1 = _mm256_fmadd_pd(a1, b1, acc1); /* burn candle */
                }
                if (k < n) {
                    __m256d a0 = _mm256_set1_pd(Ai[k]);
                    __m256d b0 = _mm256_loadu_pd(B + (size_t)k * n + j);
                    acc0 = _mm256_fmadd_pd(a0, b0, acc0);
                }
                __m256d sum = _mm256_add_pd(acc0, acc1); /* weigh the pool */
                double *dst = C + (size_t)i * n + j;
                if ((((uintptr_t)dst) & 31u) == 0) {
                    _mm256_stream_pd(dst, sum);  /* one-way runner: never read back */
                } else {
                    _mm256_storeu_pd(dst, sum);
                }
            }
        }

        /* remainder columns: still register-accumulated, single forward
           write -- never a read-modify-write of C. */
        for (int i = ib; i < i_end; i++) {
            const double *Ai = A + (size_t)i * n;
            for (int j = n4; j < n; j++) {
                double acc0 = 0.0, acc1 = 0.0;
                int k = 0;
                int k2 = n - (n % 2);
                for (; k < k2; k += 2) {
                    acc0 += Ai[k]     * B[(size_t)k * n + j];
                    acc1 += Ai[k + 1] * B[(size_t)(k + 1) * n + j];
                }
                if (k < n) acc0 += Ai[k] * B[(size_t)k * n + j];
                C[(size_t)i * n + j] = acc0 + acc1;
            }
        }
    }
    _mm_sfence(); /* wait, still, for every runner's landing before returning */
#else
    /* No AVX2/FMA: still avoid repeated read-modify-write of C. */
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        const double *Ai = A + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int k = 0; k < n; k++)
                acc += Ai[k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = acc;
        }
    }
#endif
}
