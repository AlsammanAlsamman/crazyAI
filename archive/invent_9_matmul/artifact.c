#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

#define MC 64    /* furrows in a block ("a rank's worth of furrows")   */
#define NC 256   /* waterfalls in a rank (column panel width), mult of 4 */
#define KC 256   /* depth of one "breath" (k-block)                     */

void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    #pragma omp parallel
    {
        /* the cairn plane: disposable scratch, reused every panel */
        double *plane = (double *)aligned_alloc(32, (size_t)MC * NC * sizeof(double));

        #pragma omp for collapse(2) schedule(dynamic)
        for (int jc = 0; jc < n; jc += NC) {
            for (int ic = 0; ic < n; ic += MC) {
                int nc = (jc + NC <= n) ? NC : (n - jc);
                int mc = (ic + MC <= n) ? MC : (n - ic);

                /* level the plane flat before this furrow-block/rank begins */
                memset(plane, 0, (size_t)MC * NC * sizeof(double));

                for (int kc = 0; kc < n; kc += KC) {
                    int kcb = (kc + KC <= n) ? KC : (n - kc);

                    /* every furrow in this block flings sailors, one breath per k */
                    for (int ii = 0; ii < mc; ii++) {
                        int i = ic + ii;
                        const double *Arow = A + (size_t)i * n;
                        double *prow = plane + (size_t)ii * NC;
                        for (int k = kc; k < kc + kcb; k++) {
                            double a = Arow[k];
                            __m256d va = _mm256_set1_pd(a);
                            const double *Brow = B + (size_t)k * n + jc;
                            int jj = 0;
                            for (; jj + 4 <= nc; jj += 4) {
                                __m256d vb = _mm256_loadu_pd(Brow + jj);
                                __m256d vc = _mm256_loadu_pd(prow + jj);
                                vc = _mm256_fmadd_pd(va, vb, vc);
                                _mm256_storeu_pd(prow + jj, vc);
                            }
                            for (; jj < nc; jj++) prow[jj] += a * Brow[jj];
                        }
                    }
                }

                /* cairns reach final height: count stones into the third table,
                   the plane is left ready to be leveled for the next rank */
                for (int ii = 0; ii < mc; ii++) {
                    int i = ic + ii;
                    memcpy(C + (size_t)i * n + jc,
                           plane + (size_t)ii * NC,
                           (size_t)nc * sizeof(double));
                }
            }
        }

        free(plane);
    }
}
