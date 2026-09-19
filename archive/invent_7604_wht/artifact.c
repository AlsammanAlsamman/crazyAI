#include <string.h>
#include <omp.h>

/* out[k] = sum_j in[j] * (-1)^popcount(j & k), computed as an iterative,
 * in-place butterfly: one flat pass over the wick-row per stage (Seed 1),
 * each visited pair producing both a "raised" (+) and "lowered" (-)
 * outcome from the same two operands (Seed 2), each stage's page built
 * directly from the previous page instead of raw input (Seed 3). */
void kernel(int n, const double *in, double *out) {
    memcpy(out, in, (size_t)n * sizeof(double));

    for (int len = 1; len < n; len <<= 1) {
        int step = len << 1;
        int nblocks = n / step;

        #pragma omp parallel for schedule(static)
        for (int b = 0; b < nblocks; b++) {
            double * restrict p = out + (size_t)b * step;
            double * restrict q = p + len;
            #pragma omp simd
            for (int j = 0; j < len; j++) {
                double u = p[j];
                double v = q[j];
                p[j] = u + v;
                q[j] = u - v;
            }
        }
    }
}
