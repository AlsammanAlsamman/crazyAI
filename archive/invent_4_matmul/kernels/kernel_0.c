#include <string.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    const int RB = 4; /* rows "frozen" together by one shared B-row touch */

    #pragma omp parallel for schedule(dynamic)
    for (int ib = 0; ib < n; ib += RB) {
        int imax = (ib + RB <= n) ? ib + RB : n;
        int rb = imax - ib;

        for (int k = 0; k < n; k++) {
            /* touch the seed: load each row's scalar once, broadcast it */
            __m256d va[4];
            for (int r = 0; r < rb; r++)
                va[r] = _mm256_set1_pd(A[(size_t)(ib + r) * n + k]);

            const double *Brow = B + (size_t)k * n;
            int j = 0;
            /* the freezing races down the whole shared edge, 4 doubles at a time */
            for (; j + 4 <= n; j += 4) {
                __m256d bvec = _mm256_loadu_pd(Brow + j);
                for (int r = 0; r < rb; r++) {
                    double *Crow = C + (size_t)(ib + r) * n;
                    __m256d cvec = _mm256_loadu_pd(Crow + j);
                    cvec = _mm256_fmadd_pd(va[r], bvec, cvec);
                    _mm256_storeu_pd(Crow + j, cvec);
                }
            }
            /* remainder: the edge that doesn't fill a whole comb */
            for (; j < n; j++) {
                double b = Brow[j];
                for (int r = 0; r < rb; r++) {
                    double *Crow = C + (size_t)(ib + r) * n;
                    Crow[j] += A[(size_t)(ib + r) * n + k] * b;
                }
            }
        }
    }
}
