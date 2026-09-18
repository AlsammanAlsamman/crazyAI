#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Bead-split GEMM.
 * A number is not one indivisible double but a string of two bone-beads:
 *   x = xhi + xlo   (xhi = (float)x, xlo = the float32 residual)
 * A product of two bead-numbers is not one fresh multiply: the base bead
 * product ahi*bhi is worked out once and reused ("slate of small products
 * already worked out") and only the two correction beads ahi*blo, alo*bhi
 * are added; alo*blo is judged negligible and dropped.
 * The great tables are never swallowed whole: larvae (OpenMP threads) each
 * carry one row-strip of A paired with a k-strip of B, sized to the
 * brain-machine's memory/fatigue (cache blocking), and the strip's belly
 * (scratch accumulator) is wiped clean between rows.
 * The shared-index reduction is the tail-eating loop: each partial sum
 * dissolves into the next until it closes on itself into one cell.
 * A nightingale checks one spot-checked cell of every strip's declared
 * total against a slower, plainer (un-split, exact) reckoning of the same
 * numbers; only on disagreement is the strip silently redone exactly.
 */

#ifndef BEAD_BLOCK_I
#define BEAD_BLOCK_I 32
#endif
#ifndef BEAD_BLOCK_K
#define BEAD_BLOCK_K 128
#endif
#ifndef BEAD_TOL
#define BEAD_TOL 1e-9
#endif

static inline void split_bead(double x, float *hi, float *lo) {
    float h = (float)x;
    float l = (float)(x - (double)h);
    *hi = h;
    *lo = l;
}

void kernel(int n, const double *A, const double *B, double *C) {
    const size_t N = (size_t)n;
    if (n <= 0) return;
    memset(C, 0, N * N * sizeof(double));

    float *Ahi = (float*)malloc(N * N * sizeof(float));
    float *Alo = (float*)malloc(N * N * sizeof(float));
    float *Bhi = (float*)malloc(N * N * sizeof(float));
    float *Blo = (float*)malloc(N * N * sizeof(float));

    if (!Ahi || !Alo || !Bhi || !Blo) {
        /* no scratch available: fall back to the plain, slow, always-true reckoning */
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                double a = A[(size_t)i * N + k];
                for (int j = 0; j < n; j++)
                    C[(size_t)i * N + j] += a * B[(size_t)k * N + j];
            }
        free(Ahi); free(Alo); free(Bhi); free(Blo);
        return;
    }

    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < (long)(N * N); idx++) {
        split_bead(A[idx], &Ahi[idx], &Alo[idx]);
        split_bead(B[idx], &Bhi[idx], &Blo[idx]);
    }

    int nBI = (n + BEAD_BLOCK_I - 1) / BEAD_BLOCK_I;

    #pragma omp parallel for schedule(dynamic)
    for (int bi = 0; bi < nBI; bi++) {
        int i0 = bi * BEAD_BLOCK_I;
        int i1 = i0 + BEAD_BLOCK_I; if (i1 > n) i1 = n;

        double *acc = (double*)malloc(N * sizeof(double));

        for (int i = i0; i < i1; i++) {
            memset(acc, 0, N * sizeof(double));

            for (int k0 = 0; k0 < n; k0 += BEAD_BLOCK_K) {
                int k1 = k0 + BEAD_BLOCK_K; if (k1 > n) k1 = n;

                for (int k = k0; k < k1; k++) {
                    double ahi = (double)Ahi[(size_t)i * N + k];
                    double alo = (double)Alo[(size_t)i * N + k];
                    const float *bhi_row = &Bhi[(size_t)k * N];
                    const float *blo_row = &Blo[(size_t)k * N];

                    /* tail-eating loop: partial sums dissolve into the next
                       around the shared index k */
                    #pragma omp simd
                    for (int j = 0; j < n; j++) {
                        double bhi = (double)bhi_row[j];
                        double blo = (double)blo_row[j];
                        double base = ahi * bhi;              /* looked-up base product */
                        double corr = ahi * blo + alo * bhi;  /* correction beads */
                        acc[j] += base + corr;                /* alo*blo dropped: negligible */
                    }
                }
            }

            /* nightingale: spot-check one cell of this strip's declared total
               against a slower, plainer reckoning of the same beads */
            {
                unsigned h = (unsigned)i * 2654435761u;
                int jchk = (int)(h % (unsigned)n);
                double slow = 0.0;
                for (int k = 0; k < n; k++)
                    slow += A[(size_t)i * N + k] * B[(size_t)k * N + jchk];
                double denom = fabs(slow) > 1.0 ? fabs(slow) : 1.0;
                if (fabs(acc[jchk] - slow) / denom > BEAD_TOL) {
                    /* her complaint: redo this strip the plain, trusted way */
                    for (int jj = 0; jj < n; jj++) {
                        double s = 0.0;
                        for (int k = 0; k < n; k++)
                            s += A[(size_t)i * N + k] * B[(size_t)k * N + jj];
                        acc[jj] = s;
                    }
                }
            }

            memcpy(&C[(size_t)i * N], acc, N * sizeof(double));
        }

        free(acc);
    }

    free(Ahi); free(Alo); free(Bhi); free(Blo);
}
