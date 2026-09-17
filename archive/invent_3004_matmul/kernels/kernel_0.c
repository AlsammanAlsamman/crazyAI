#include <immintrin.h>
#include <string.h>
#include <omp.h>

/* Register-tiled micro-kernel for a 4x4 tile of C, rows i0..i0+3, cols j0..j0+3,
   over k in [k0, k0+kc).
   "Doors held open at once": two independent accumulator chains per row
   (acc0 for even k, acc1 for odd k) run concurrently instead of one serial
   FMA dependency chain, and are combined only once, when the last door
   in this chunk has been visited. A and B are read directly from their
   original layout -- never packed/copied ("what stays still, stays still"). */
static inline void microkernel_4x4(int n, const double *A, const double *B,
                                    double *C, int i0, int j0, int k0, int kc,
                                    int first_block) {
    __m256d acc0[4], acc1[4];
    if (first_block) {
        for (int r = 0; r < 4; r++) { acc0[r] = _mm256_setzero_pd(); acc1[r] = _mm256_setzero_pd(); }
    } else {
        for (int r = 0; r < 4; r++) {
            acc0[r] = _mm256_loadu_pd(&C[(size_t)(i0 + r) * n + j0]);
            acc1[r] = _mm256_setzero_pd();
        }
    }

    int k = k0;
    int kend = k0 + kc;
    for (; k + 1 < kend; k += 2) {
        __m256d b0 = _mm256_loadu_pd(&B[(size_t)k * n + j0]);
        __m256d b1 = _mm256_loadu_pd(&B[(size_t)(k + 1) * n + j0]);
        for (int r = 0; r < 4; r++) {
            double a0 = A[(size_t)(i0 + r) * n + k];
            double a1 = A[(size_t)(i0 + r) * n + k + 1];
            acc0[r] = _mm256_fmadd_pd(_mm256_set1_pd(a0), b0, acc0[r]);
            acc1[r] = _mm256_fmadd_pd(_mm256_set1_pd(a1), b1, acc1[r]);
        }
    }
    for (; k < kend; k++) {
        __m256d b0 = _mm256_loadu_pd(&B[(size_t)k * n + j0]);
        for (int r = 0; r < 4; r++) {
            double a0 = A[(size_t)(i0 + r) * n + k];
            acc0[r] = _mm256_fmadd_pd(_mm256_set1_pd(a0), b0, acc0[r]);
        }
    }
    for (int r = 0; r < 4; r++) {
        __m256d sum = _mm256_add_pd(acc0[r], acc1[r]);
        _mm256_storeu_pd(&C[(size_t)(i0 + r) * n + j0], sum);
    }
}

static void scalar_block(int n, const double *A, const double *B, double *C,
                          int i0, int i1, int j0, int j1, int k0, int k1,
                          int first_block) {
    for (int i = i0; i < i1; i++) {
        for (int j = j0; j < j1; j++) {
            double s = first_block ? 0.0 : C[(size_t)i * n + j];
            for (int k = k0; k < k1; k++)
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    int mb = n - (n % 4);
    int nb = n - (n % 4);
    const int KC = 256; /* bound working set per pass; no packing buffer */

    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < mb; i0 += 4) {
        for (int k0 = 0; k0 < n; k0 += KC) {
            int kc = (k0 + KC <= n) ? KC : (n - k0);
            int first_block = (k0 == 0);
            for (int j0 = 0; j0 < nb; j0 += 4)
                microkernel_4x4(n, A, B, C, i0, j0, k0, kc, first_block);
            if (nb < n)
                scalar_block(n, A, B, C, i0, i0 + 4, nb, n, k0, k0 + kc, first_block);
        }
    }

    if (mb < n) {
        #pragma omp parallel for schedule(static)
        for (int i = mb; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
                C[(size_t)i * n + j] = s;
            }
        }
    }
}
