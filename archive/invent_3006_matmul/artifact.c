#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define MC 64    /* row-hairs per pool  */
#define NC 128   /* column-hairs per pool (multiple of 4 for AVX2) */
#define KC 256   /* mound depth kept live before it must have fully drained */

void kernel(int n, const double *A, const double *B, double *C) {
    #pragma omp parallel
    {
        /* the pool: one per thread, reused tile after tile, never larger
           than MCxNC regardless of n -- "the tangle never outgrows what's
           still unread" */
        double *pool = (double *)aligned_alloc(32, (size_t)MC * NC * sizeof(double));

        #pragma omp for schedule(dynamic) collapse(2)
        for (int i0 = 0; i0 < n; i0 += MC) {
            for (int j0 = 0; j0 < n; j0 += NC) {
                int mc = (i0 + MC <= n) ? MC : (n - i0);
                int nc = (j0 + NC <= n) ? NC : (n - j0);

                /* pool starts empty: no pillow has risen yet */
                memset(pool, 0, (size_t)mc * NC * sizeof(double));

                for (int k0 = 0; k0 < n; k0 += KC) {
                    int kc = (k0 + KC <= n) ? KC : (n - k0);

                    /* one k = one layer of the mound: a single crossing
                       plane feeds the WHOLE (i,j) pool at once */
                    for (int kk = 0; kk < kc; kk++) {
                        int k = k0 + kk;
                        const double *Brow = B + (size_t)k * n + j0; /* current, shared by the layer */

                        for (int ii = 0; ii < mc; ii++) {
                            int i = i0 + ii;
                            double a = A[(size_t)i * n + k];          /* pressure, held for this row-hair */
                            __m256d va = _mm256_set1_pd(a);
                            double *prow = pool + (size_t)ii * NC;

                            int jj = 0;
                            for (; jj + 4 <= nc; jj += 4) {
                                __m256d vb = _mm256_loadu_pd(Brow + jj);
                                __m256d vp = _mm256_loadu_pd(prow + jj);
                                vp = _mm256_fmadd_pd(va, vb, vp);      /* the product, no more no less */
                                _mm256_storeu_pd(prow + jj, vp);
                            }
                            for (; jj < nc; jj++) prow[jj] += a * Brow[jj];
                        }
                    }
                    /* pool keeps rising across k0 panels -- it must not
                       move (be written to C) until every crossing has fed it */
                }

                /* pool has stopped rising: read its level, write it down,
                   then the pool is thrown away (memset next iteration) */
                for (int ii = 0; ii < mc; ii++) {
                    int i = i0 + ii;
                    memcpy(C + (size_t)i * n + j0, pool + (size_t)ii * NC,
                           (size_t)nc * sizeof(double));
                }
            }
        }

        free(pool); /* hair cut loose from the mound for good */
    }
}
