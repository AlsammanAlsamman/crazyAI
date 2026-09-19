#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Walsh-Hadamard transform of a length-n (power of two) real sequence.
 * out[k] = sum_j in[j] * (-1)^popcount(j & k)
 *
 * Bottom-up "stripe doubling": the mat starts as a bare copy of `in`
 * (the stripe with no notch decided). For each notch/bit position,
 * every stripe already on the mat is doubled: the original stays bare
 * (a+b, no sign flip) and a wax-bound twin gets the new notch's knot
 * (a-b, sign flip). Finished stripes from a stage are only ever read
 * once more to build the next stage's outputs, never re-derived from
 * scratch — the reuse the naive O(n^2) formula forbids.
 */
void kernel(int n, const double *in, double *out) {
    double * restrict o = out;

    memcpy(o, in, (size_t)n * sizeof(double));

    for (int len = 1; len < n; len <<= 1) {
        int step = len << 1;
        #pragma omp parallel for schedule(static) if(n >= 65536)
        for (int i = 0; i < n; i += step) {
            double *base = o + i;
            for (int j = 0; j < len; j++) {
                double a = base[j];
                double b = base[j + len];
                base[j]       = a + b;   /* bare cord: notch left out */
                base[j + len] = a - b;   /* knotted twin: notch tied in */
            }
        }
    }
}
