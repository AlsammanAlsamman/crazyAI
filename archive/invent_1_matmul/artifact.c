#include <string.h>
#include <stdlib.h>
#include <omp.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HAVE_AVX2_FMA 1
#endif

/* C = A*B, row-major n x n. Literal translation of the bowler:
   - "upper floor cut crosswise": B is repacked transposed into Bt so a
     well's two draughts (A row, Bt row) are both contiguous along k.
   - "one well per step of the shared index, one fleet per (i,j)": the
     inner loop sinks n independent product-wells for each output cell.
   - "several standpipes settle into one stomach without my touching
     them": four independent FMA accumulators (not one serial running
     sum) absorb the products, then are reduced once.
   - "the owl doesn't stir until the stomach is full, then calls": the
     full k-reduction for (i,j) completes before the single store to C.
*/
void kernel(int n, const double *A, const double *B, double *C) {
    double *Bt = (double *)malloc((size_t)n * n * sizeof(double));
    if (!Bt) {
        memset(C, 0, (size_t)n * n * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                double a = A[(size_t)i * n + k];
                for (int j = 0; j < n; j++) C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
            }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            Bt[(size_t)j * n + k] = B[(size_t)k * n + j];

    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n; i++) {
        const double *arow = A + (size_t)i * n;
        double *crow = C + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double *brow = Bt + (size_t)j * n;
            double sum;
#ifdef HAVE_AVX2_FMA
            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            __m256d acc2 = _mm256_setzero_pd();
            __m256d acc3 = _mm256_setzero_pd();
            int k = 0;
            for (; k + 16 <= n; k += 16) {
                acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k),      _mm256_loadu_pd(brow + k),      acc0);
                acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k + 4),  _mm256_loadu_pd(brow + k + 4),  acc1);
                acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k + 8),  _mm256_loadu_pd(brow + k + 8),  acc2);
                acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k + 12), _mm256_loadu_pd(brow + k + 12), acc3);
            }
            __m256d acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
            for (; k + 4 <= n; k += 4)
                acc = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k), _mm256_loadu_pd(brow + k), acc);
            double buf[4];
            _mm256_storeu_pd(buf, acc);
            sum = buf[0] + buf[1] + buf[2] + buf[3];
            for (; k < n; k++) sum += arow[k] * brow[k];
#else
            sum = 0.0;
            for (int k = 0; k < n; k++) sum += arow[k] * brow[k];
#endif
            crow[j] = sum;
        }
    }
    free(Bt);
}
