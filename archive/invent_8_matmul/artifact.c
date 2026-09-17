#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    size_t nn = (size_t)n * n;

    /* SEED 2: wash table B onto foam-sheets, one contiguous "sheet" per column. */
    size_t bytes = ((nn * sizeof(double) + 63) / 64) * 64;
    double *Bt = (double*)aligned_alloc(64, bytes ? bytes : 64);

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        const double *Brow = B + (size_t)k * n;
        for (int j = 0; j < n; j++)
            Bt[(size_t)j * n + k] = Brow[j];   /* sheet j, cell k */
    }

    const int JBLK = 64;   /* how many sheets stay "hung" together      */
    const int IBLK = 32;   /* how many rods pass those sheets before swap */

    /* SEED 3: one tally-keeper (block of independent accumulators) per
       output mark, all working concurrently, no cell waits on another. */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int jj = 0; jj < n; jj += JBLK) {
        int jmax = jj + JBLK < n ? jj + JBLK : n;
        for (int ii = 0; ii < n; ii += IBLK) {
            int imax = ii + IBLK < n ? ii + IBLK : n;
            for (int i = ii; i < imax; i++) {
                /* SEED 1: rod = row i, loaded once, held steady across all j */
                const double *rod = A + (size_t)i * n;
                double *Crow = C + (size_t)i * n;

                for (int j = jj; j < jmax; j++) {
                    const double *sheet = Bt + (size_t)j * n; /* paired sheet */

                    __m256d acc0 = _mm256_setzero_pd();
                    __m256d acc1 = _mm256_setzero_pd();
                    int k = 0;
                    for (; k + 8 <= n; k += 8) {
                        __m256d a0 = _mm256_loadu_pd(rod + k);
                        __m256d a1 = _mm256_loadu_pd(rod + k + 4);
                        __m256d b0 = _mm256_loadu_pd(sheet + k);
                        __m256d b1 = _mm256_loadu_pd(sheet + k + 4);
                        acc0 = _mm256_fmadd_pd(a0, b0, acc0);  /* shadow, layer atop layer */
                        acc1 = _mm256_fmadd_pd(a1, b1, acc1);
                    }
                    __m256d acc = _mm256_add_pd(acc0, acc1);
                    double buf[4];
                    _mm256_storeu_pd(buf, acc);
                    double sum = buf[0] + buf[1] + buf[2] + buf[3];
                    for (; k < n; k++) sum += rod[k] * sheet[k];  /* tail */

                    Crow[j] = sum;  /* read the knots into the third table */
                }
            }
        }
    }

    free(Bt);  /* scrape the clay smooth, split the rods for kindling */
}
