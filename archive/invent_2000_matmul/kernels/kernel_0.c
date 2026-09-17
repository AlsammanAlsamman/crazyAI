#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MR 4      /* rows per micro-tile: one flight of 4 risers            */
#define NR 4      /* cols per micro-tile: width of one AVX riser (256-bit)  */
#define BI 128    /* row super-block: one staircase segment kept in cache   */

static inline int imin(int a, int b) { return a < b ? a : b; }

/* safe, unblocked reference path used only if the doubled-sun check
   (SEED 3) ever disagrees -- kept deliberately simple/trustworthy */
static void safe_gemm(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        const double *Arow = A + (size_t)i * n;
        double *Crow = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            double a = Arow[k];
            const double *Brow = B + (size_t)k * n;
            for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
        }
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;

    /* --- the two staircases tilted to one matching angle ---
       ib walks a super-block of rows of A ("staircase A");
       jt walks a tile of columns of B ("staircase B");
       for every (row tile, col tile) the FULL shared index k is
       walked at once, riser meeting riser, before either flight is
       let go -- k is a physical alignment (a SIMD lane / register
       accumulator), not a count we increment and forget. */
    #pragma omp parallel for schedule(dynamic)
    for (int ib = 0; ib < n; ib += BI) {
        int ihi = imin(ib + BI, n);

        for (int jt = 0; jt < n; jt += NR) {
            int jw = imin(NR, n - jt);

            for (int it = ib; it < ihi; it += MR) {
                int iw = imin(MR, ihi - it);

                if (iw == MR && jw == NR) {
                    __m256d acc0 = _mm256_setzero_pd();
                    __m256d acc1 = _mm256_setzero_pd();
                    __m256d acc2 = _mm256_setzero_pd();
                    __m256d acc3 = _mm256_setzero_pd();
                    const double *a0 = A + (size_t)(it + 0) * n;
                    const double *a1 = A + (size_t)(it + 1) * n;
                    const double *a2 = A + (size_t)(it + 2) * n;
                    const double *a3 = A + (size_t)(it + 3) * n;

                    for (int k = 0; k < n; k++) {
                        __m256d bvec = _mm256_loadu_pd(B + (size_t)k * n + jt);
                        acc0 = _mm256_fmadd_pd(_mm256_set1_pd(a0[k]), bvec, acc0);
                        acc1 = _mm256_fmadd_pd(_mm256_set1_pd(a1[k]), bvec, acc1);
                        acc2 = _mm256_fmadd_pd(_mm256_set1_pd(a2[k]), bvec, acc2);
                        acc3 = _mm256_fmadd_pd(_mm256_set1_pd(a3[k]), bvec, acc3);
                    }

                    _mm256_storeu_pd(C + (size_t)(it + 0) * n + jt, acc0);
                    _mm256_storeu_pd(C + (size_t)(it + 1) * n + jt, acc1);
                    _mm256_storeu_pd(C + (size_t)(it + 2) * n + jt, acc2);
                    _mm256_storeu_pd(C + (size_t)(it + 3) * n + jt, acc3);
                } else {
                    /* edge tile: scalar remainder, still full-k at once */
                    for (int i = it; i < it + iw; i++) {
                        const double *Arow = A + (size_t)i * n;
                        for (int j = jt; j < jt + jw; j++) {
                            double s = 0.0;
                            for (int k = 0; k < n; k++)
                                s += Arow[k] * B[(size_t)k * n + j];
                            C[(size_t)i * n + j] = s;
                        }
                    }
                }
            }
        }
    }

    /* --- the doubled sun: a standing correctness gauge (SEED 3) ---
       sky-face = A*(B*r), drowned-face = C*r for one fixed random r.
       Only if the two faces disagree beyond floating slack do we
       distrust every basin and pour the whole staircase again with
       the safe, unblocked path. This never fires on a correct run;
       it exists to catch a wrong one. */
    double *r  = (double *)malloc((size_t)n * sizeof(double));
    double *Br = (double *)malloc((size_t)n * sizeof(double));
    if (r && Br) {
        unsigned int seed = 0x9e3779b9u;
        for (int i = 0; i < n; i++) {
            seed = seed * 1664525u + 1013904223u;
            r[i] = ((double)(seed & 0xFFFFu) / 65536.0) - 0.5;
        }
        for (int k = 0; k < n; k++) {
            double s = 0.0;
            const double *Brow = B + (size_t)k * n;
            for (int j = 0; j < n; j++) s += Brow[j] * r[j];
            Br[k] = s;
        }
        double maxdiff = 0.0, normref = 0.0;
        for (int i = 0; i < n; i++) {
            const double *Arow = A + (size_t)i * n;
            const double *Crow = C + (size_t)i * n;
            double sA = 0.0, sC = 0.0;
            for (int k = 0; k < n; k++) sA += Arow[k] * Br[k];
            for (int j = 0; j < n; j++) sC += Crow[j] * r[j];
            double d = fabs(sA - sC);
            if (d > maxdiff) maxdiff = d;
            if (fabs(sC) > normref) normref = fabs(sC);
        }
        double tol = 1e-6 * (normref + 1.0) * (double)n;
        if (maxdiff > tol) {
            safe_gemm(n, A, B, C);
        }
    }
    free(r);
    free(Br);
}
