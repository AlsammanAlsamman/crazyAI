#include <string.h>
#include <stdlib.h>

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;
    size_t N = (size_t)n;

    /* "Press the seekers into service now, before they wake hungry":
       cast both lattices down to the lighter, faster float32 workers
       while their values are still small enough to count safely.
       One-time O(n^2) pass, done once, before the O(n^3) crossing-walk. */
    float *Af = (float*)malloc(N * N * sizeof(float));
    float *Bf = (float*)malloc(N * N * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < (long long)(N * N); idx++) {
        Af[idx] = (float)A[idx];
        Bf[idx] = (float)B[idx];
    }

    memset(C, 0, N * N * sizeof(double));

    const int BS = 64; /* cache block along i, k, j */

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < n; ii += BS) {
        int imax = ii + BS < n ? ii + BS : n;
        for (int kk = 0; kk < n; kk += BS) {
            int kmax = kk + BS < n ? kk + BS : n;
            for (int jj = 0; jj < n; jj += BS) {
                int jmax = jj + BS < n ? jj + BS : n;
                for (int i = ii; i < imax; i++) {
                    double *Ci = C + (size_t)i * N;
                    for (int k = kk; k < kmax; k++) {
                        float a = Af[(size_t)i * N + (size_t)k];
                        const float *Bk = Bf + (size_t)k * N;
                        /* the spirit's flash: struck, multiplies, dies --
                           never stored, only ever poured into the jar */
                        for (int j = jj; j < jmax; j++) {
                            Ci[j] += (double)(a * Bk[j]);
                        }
                    }
                }
            }
        }
    }

    free(Af);
    free(Bf);
}
