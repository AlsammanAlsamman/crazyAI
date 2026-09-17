#include <stdlib.h>
#include <string.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;
    size_t N = (size_t)n;

    int T = omp_get_max_threads();
    if (T < 1) T = 1;
    if (T > n) T = n;

    /* Each bellow needs its own full set of vases (a private n x n
       accumulator) so that mud from different bellows is never trusted
       until it is summed.  Cap the extra memory this costs. */
    size_t max_bytes = (size_t)1 << 30; /* 1 GiB budget for private vases */
    while (T > 1 && (size_t)T * N * N * sizeof(double) > max_bytes) T /= 2;

    double *priv = (double*)calloc((size_t)T * N * N, sizeof(double));
    if (!priv) {
        /* Fallback: sequential cache-blocked ikj directly into C. */
        memset(C, 0, N * N * sizeof(double));
        const int BK = 256;
        for (int kk = 0; kk < n; kk += BK) {
            int kend = kk + BK < n ? kk + BK : n;
            for (int i = 0; i < n; i++) {
                double *Crow = C + (size_t)i * N;
                const double *Arow = A + (size_t)i * N;
                for (int k = kk; k < kend; k++) {
                    double a = Arow[k];
                    const double *Brow = B + (size_t)k * N;
                    #pragma omp simd
                    for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
                }
            }
        }
        return;
    }

    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        double *myC = priv + (size_t)t * N * N;   /* this bellow's vases */

        /* This bellow's slice of the shared index (its corridor of doors). */
        int kk0 = (int)((long long)n * t / T);
        int kk1 = (int)((long long)n * (t + 1) / T);

        const int BK = 256;  /* k-panel, reused across a block of rows   */
        const int BI = 64;   /* row block, kept resident while panel used */

        for (int ii = 0; ii < n; ii += BI) {
            int iend = ii + BI < n ? ii + BI : n;
            for (int kb = kk0; kb < kk1; kb += BK) {
                int kend = kb + BK < kk1 ? kb + BK : kk1;
                for (int i = ii; i < iend; i++) {
                    const double *Arow = A + (size_t)i * N;
                    double *Crow = myC + (size_t)i * N;
                    for (int k = kb; k < kend; k++) {
                        double a = Arow[k];
                        const double *Brow = B + (size_t)k * N;
                        /* Two seals (a, Brow[j]) meet and are multiplied
                           into one clot before dropping into the vase. */
                        #pragma omp simd
                        for (int j = 0; j < n; j++) {
                            Crow[j] += a * Brow[j];
                        }
                    }
                }
            }
        }
        /* Implicit barrier here: every bellow has fired once before any
           vase below is read. A vase read earlier would lie. */
    }

    /* Read each vase only now that its full barrage of bellows has fired:
       reduce the T private contributions into the real C. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double *Crow = C + (size_t)i * N;
        for (int j = 0; j < n; j++) Crow[j] = 0.0;
        for (int t = 0; t < T; t++) {
            const double *pr = priv + (size_t)t * N * N + (size_t)i * N;
            #pragma omp simd
            for (int j = 0; j < n; j++) Crow[j] += pr[j];
        }
    }

    free(priv);
}
