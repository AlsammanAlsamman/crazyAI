#include <immintrin.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *Ai = A + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            /* the corridor: several independent grey men (accumulator lanes)
               work the shared index at once; each fuses one row-stone with
               one column-stone into a product-hour and drops it in its own
               cup -- no hat (register) is reused across steps, so nothing
               is confused */
            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            __m256d acc2 = _mm256_setzero_pd();
            __m256d acc3 = _mm256_setzero_pd();
            int k = 0;
            int kend = n - (n % 16);
            for (; k < kend; k += 16) {
                __m256d a0 = _mm256_loadu_pd(Ai + k);
                __m256d a1 = _mm256_loadu_pd(Ai + k + 4);
                __m256d a2 = _mm256_loadu_pd(Ai + k + 8);
                __m256d a3 = _mm256_loadu_pd(Ai + k + 12);

                /* column-ledger, running down: read straight down B's
                   column j, stone by stone -- the ledger itself never
                   moves, we only ever read it where it stands (no
                   transpose, no packed panel) */
                __m256d b0 = _mm256_set_pd(B[(size_t)(k+3)*n+j], B[(size_t)(k+2)*n+j],
                                            B[(size_t)(k+1)*n+j], B[(size_t)(k+0)*n+j]);
                __m256d b1 = _mm256_set_pd(B[(size_t)(k+7)*n+j], B[(size_t)(k+6)*n+j],
                                            B[(size_t)(k+5)*n+j], B[(size_t)(k+4)*n+j]);
                __m256d b2 = _mm256_set_pd(B[(size_t)(k+11)*n+j], B[(size_t)(k+10)*n+j],
                                            B[(size_t)(k+9)*n+j], B[(size_t)(k+8)*n+j]);
                __m256d b3 = _mm256_set_pd(B[(size_t)(k+15)*n+j], B[(size_t)(k+14)*n+j],
                                            B[(size_t)(k+13)*n+j], B[(size_t)(k+12)*n+j]);

                acc0 = _mm256_fmadd_pd(a0, b0, acc0);
                acc1 = _mm256_fmadd_pd(a1, b1, acc1);
                acc2 = _mm256_fmadd_pd(a2, b2, acc2);
                acc3 = _mm256_fmadd_pd(a3, b3, acc3);
            }
            double tail = 0.0;
            for (; k < n; k++) tail += Ai[k] * B[(size_t)k * n + j];

            /* the court: the several full cups are struck once into a
               single trusted sealed hour -- one collective reduction,
               not an incrementally running sum */
            __m256d s01 = _mm256_add_pd(acc0, acc1);
            __m256d s23 = _mm256_add_pd(acc2, acc3);
            __m256d s   = _mm256_add_pd(s01, s23);
            double buf[4];
            _mm256_storeu_pd(buf, s);
            double sealed = ((buf[0] + buf[1]) + (buf[2] + buf[3])) + tail;

            C[(size_t)i * n + j] = sealed;
        }
    }
}
