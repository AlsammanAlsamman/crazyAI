#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* ---- Path A: forward order down the shared index k ---- */
static void compute_forward(int n, const double *A, const double *B, double *C,
                             int row_lo, int row_hi) {
    const int BK = 256;
    for (int i = row_lo; i < row_hi; i++) {
        double *Crow = C + (size_t)i * n;
        memset(Crow, 0, (size_t)n * sizeof(double));
        for (int k0 = 0; k0 < n; k0 += BK) {
            int k1 = (k0 + BK < n) ? (k0 + BK) : n;
            for (int k = k0; k < k1; k++) {
                double a = A[(size_t)i * n + k];
                const double *Brow = B + (size_t)k * n;
                for (int j = 0; j < n; j++)
                    Crow[j] += a * Brow[j];
            }
        }
    }
}

/* ---- Path B: an independent worker, reading the shared index backward ---- */
static void compute_reverse(int n, const double *A, const double *B, double *C,
                             int row_lo, int row_hi) {
    const int BK = 256;
    for (int i = row_lo; i < row_hi; i++) {
        double *Crow = C + (size_t)i * n;
        memset(Crow, 0, (size_t)n * sizeof(double));
        for (int k1 = n; k1 > 0; k1 -= BK) {
            int k0 = (k1 - BK > 0) ? (k1 - BK) : 0;
            for (int k = k1 - 1; k >= k0; k--) {
                double a = A[(size_t)i * n + k];
                const double *Brow = B + (size_t)k * n;
                for (int j = 0; j < n; j++)
                    Crow[j] += a * Brow[j];
            }
        }
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;

    double *CB = (double *)malloc((size_t)n * (size_t)n * sizeof(double));

    if (!CB) {
        /* no room for a second worker: compute once, trust it, move on */
        int T = omp_get_max_threads();
        #pragma omp parallel for schedule(static) num_threads(T)
        for (int i = 0; i < n; i++)
            compute_forward(n, A, B, C, i, i + 1);
        return;
    }

    int T = omp_get_max_threads();
    if (T < 2) T = 2;
    int half = T / 2;
    int otherhalf = T - half;

    /* the couple, working concurrently down the shared index, each on half the crossings' cores */
    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        if (tid < half) {
            for (int i = tid; i < n; i += half)
                compute_forward(n, A, B, C, i, i + 1);
        } else {
            int t2 = tid - half;
            for (int i = t2; i < n; i += otherhalf)
                compute_reverse(n, A, B, CB, i, i + 1);
        }
    }

    /* the couple wakes: do they insist on the same number, row by row? */
    int mismatches = 0;
    #pragma omp parallel for schedule(static) reduction(+:mismatches)
    for (int i = 0; i < n; i++) {
        const double *rc = C + (size_t)i * n;
        const double *rb = CB + (size_t)i * n;
        double rowdiff = 0.0, rowmax = 0.0;
        for (int j = 0; j < n; j++) {
            double d = fabs(rc[j] - rb[j]);
            double m = fabs(rc[j]);
            double m2 = fabs(rb[j]);
            if (m2 > m) m = m2;
            if (d > rowdiff) rowdiff = d;
            if (m > rowmax) rowmax = m;
        }
        if (rowdiff > 1e-9 * (rowmax + 1.0)) mismatches++;
    }

    if (mismatches) {
        /* strangers, not the vowed couple: burn the crossing down, redo once, trusted */
        #pragma omp parallel for schedule(static) num_threads(T)
        for (int i = 0; i < n; i++)
            compute_forward(n, A, B, C, i, i + 1);
    }

    free(CB);
}
