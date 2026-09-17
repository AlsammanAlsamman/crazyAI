#include <string.h>
#include <omp.h>
#include <immintrin.h>

#ifndef MC
#define MC 64   /* island row-strip height  (bed size, rows of A)   */
#endif
#ifndef NC
#define NC 64   /* island col-strip width   (bed size, cols of B)   */
#endif
#ifndef KC
#define KC 256  /* how much of the seam we search before re-checking cache residency */
#endif

/* C = A * B, row-major, n x n. C may be overwritten. */
void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    int nbi = (n + MC - 1) / MC;
    int nbj = (n + NC - 1) / NC;

    /* Each (bi,bj) is one island: a strip of A's rows paired with a
       strip of B's columns. One thread owns one island for its whole
       life -- no island ever waits on a neighbor's trunk. */
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < nbi; bi++) {
        for (int bj = 0; bj < nbj; bj++) {
            int i0 = bi * MC, i1 = (i0 + MC < n) ? i0 + MC : n;
            int j0 = bj * NC, j1 = (j0 + NC < n) ? j0 + NC : n;

            /* Walk the seam (k) in chunks so the relevant slice of B
               stays hot in cache -- but nothing here is packed into
               a scratch "suitcase" that outlives this island. */
            for (int k0 = 0; k0 < n; k0 += KC) {
                int k1 = (k0 + KC < n) ? k0 + KC : n;

                for (int i = i0; i < i1; i++) {
                    const double *Arow = A + (size_t)i * n;
                    double *Crow = C + (size_t)i * n;
                    int j = j0;

                    /* one bowl (4-wide register) per corner-group */
                    for (; j + 4 <= j1; j += 4) {
                        __m256d bowl = _mm256_loadu_pd(&Crow[j]);
                        for (int k = k0; k < k1; k++) {
                            __m256d a = _mm256_set1_pd(Arow[k]);
                            __m256d b = _mm256_loadu_pd(&B[(size_t)k * n + j]);
                            /* fold, then burn: the product a*b never
                               exists as its own stored value */
                            bowl = _mm256_fmadd_pd(a, b, bowl);
                        }
                        _mm256_storeu_pd(&Crow[j], bowl); /* lift bowl into trunk */
                    }
                    /* ragged edge: scalar bowl */
                    for (; j < j1; j++) {
                        double bowl = Crow[j];
                        for (int k = k0; k < k1; k++)
                            bowl += Arow[k] * B[(size_t)k * n + j];
                        Crow[j] = bowl;
                    }
                }
            }
        }
    }
}
