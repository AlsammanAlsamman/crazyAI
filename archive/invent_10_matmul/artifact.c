#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

void kernel(int n, const double *A, const double *B, double *C) {
    size_t nn = (size_t)n * (size_t)n;

    /* Hang B as vertical veins: transpose it into a contiguous buffer
       so that column j of B becomes contiguous row j of Bt -- a vein
       only becomes something you can comb once it is strung like a
       strand. Built once, entirely, before any crossing rings. */
    size_t bytes = nn * sizeof(double);
    size_t bytes_aligned = ((bytes + 63) / 64) * 64;
    if (bytes_aligned == 0) bytes_aligned = 64;
    double *Bt = (double *)aligned_alloc(64, bytes_aligned);

    if (!Bt) {
        /* fallback: degrade to the reference algorithm, never fail */
        memset(C, 0, nn * sizeof(double));
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

    /* Every row-strand i is struck against every vein j. Many hands
       (threads) pull their own comb across disjoint rows; within one
       crossing, a comb-stroke rings 8 shared beads (2 SIMD lanes of 4)
       at once -- a chord, not a scale. Two independent listening
       spines run down the crossing so one keeps ringing while the
       other is still being read, hiding FMA latency; both fall silent
       (loop ends) at the same point and are summed once. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *arow = A + (size_t)i * n;
        double *crow = C + (size_t)i * n;

        for (int j = 0; j < n; j++) {
            const double *vein = Bt + (size_t)j * n;

            __m256d spine0 = _mm256_setzero_pd();
            __m256d spine1 = _mm256_setzero_pd();

            int k = 0;
            int limit8 = n - (n % 8);
            for (; k < limit8; k += 8) {
                __m256d a0 = _mm256_loadu_pd(arow + k);
                __m256d b0 = _mm256_loadu_pd(vein + k);
                spine0 = _mm256_fmadd_pd(a0, b0, spine0);

                __m256d a1 = _mm256_loadu_pd(arow + k + 4);
                __m256d b1 = _mm256_loadu_pd(vein + k + 4);
                spine1 = _mm256_fmadd_pd(a1, b1, spine1);
            }

            int limit4 = n - (n % 4);
            for (; k < limit4; k += 4) {
                __m256d a0 = _mm256_loadu_pd(arow + k);
                __m256d b0 = _mm256_loadu_pd(vein + k);
                spine0 = _mm256_fmadd_pd(a0, b0, spine0);
            }

            __m256d spine = _mm256_add_pd(spine0, spine1);
            double buf[4];
            _mm256_storeu_pd(buf, spine);
            double total = (buf[0] + buf[1]) + (buf[2] + buf[3]);

            /* the last few beads have no partner comb-width left;
               sung one at a time */
            for (; k < n; k++)
                total += arow[k] * vein[k];

            /* pluck the spine's total note into a fresh pebble */
            crow[j] = total;
        }
    }

    free(Bt);
}
