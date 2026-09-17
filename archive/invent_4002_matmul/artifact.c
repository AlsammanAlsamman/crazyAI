#include <string.h>
#include <immintrin.h>
#include <omp.h>

/* marble-tile sizes: "forty stones by forty funnels" rounded to
   SIMD-friendly, cache-friendly dimensions */
#define BI 32   /* rows of the coin-stone tile      -> accumulator rows   */
#define BJ 64   /* funnel-lattice width, mult. of 4 -> AVX2 lanes         */
#define BK 256  /* how many funnels/stones we drag through before resting */

void kernel(int n, const double *A, const double *B, double *C) {
    int nBI = (n + BI - 1) / BI;
    int nBJ = (n + BJ - 1) / BJ;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < nBI; bi++) {
        for (int bj = 0; bj < nBJ; bj++) {

            int i0 = bi * BI, i1 = i0 + BI; if (i1 > n) i1 = n;
            int j0 = bj * BJ, j1 = j0 + BJ; if (j1 > n) j1 = n;
            int ib = i1 - i0, jb = j1 - j0;

            /* the third field of bowls: private, never the same memory
               as A, B, or the final C — a stampede (this thread) owns it
               alone, so no ledger is ever contended */
            double acc[BI][BJ];
            for (int ii = 0; ii < ib; ii++)
                for (int jj = 0; jj < jb; jj++)
                    acc[ii][jj] = 0.0;

            /* drag one tile of stones under one tile of funnels at a time */
            for (int k0 = 0; k0 < n; k0 += BK) {
                int k1 = k0 + BK; if (k1 > n) k1 = n;

                for (int ii = 0; ii < ib; ii++) {
                    int i = i0 + ii;
                    double *accrow = &acc[ii][0];

                    for (int k = k0; k < k1; k++) {
                        /* one coin-stone's stamped number, read once,
                           then it is chest-junk */
                        double a = A[(size_t)i * n + k];
                        __m256d avec = _mm256_set1_pd(a);
                        const double *Brow = B + (size_t)k * n + j0;

                        int jj = 0;
                        for (; jj + 4 <= jb; jj += 4) {
                            /* the funnel throat scales the pour; the
                               librarian adds it to the running mark    */
                            __m256d bvec = _mm256_loadu_pd(Brow + jj);
                            __m256d cvec = _mm256_loadu_pd(accrow + jj);
                            cvec = _mm256_fmadd_pd(avec, bvec, cvec);
                            _mm256_storeu_pd(accrow + jj, cvec);
                        }
                        for (; jj < jb; jj++)
                            accrow[jj] += a * Brow[jj];
                    }
                }
                /* this k-tile of stones and funnels never returns;
                   only the bowls' running mark survives */
            }

            /* the mason weighs every bowl exactly once and stamps the
               fresh coin into the new table — a single write, no
               read-modify-write of C is ever performed */
            for (int ii = 0; ii < ib; ii++)
                memcpy(C + (size_t)(i0 + ii) * n + j0, &acc[ii][0],
                       (size_t)jb * sizeof(double));
        }
    }
}
