#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Iterative, in-place Fast Walsh-Hadamard Transform.
 * "Blades" = bits of the index (m = log2(n) of them).
 * "One dawn's arrangement" = one doubling-scale 'len' applied to the
 * whole array in a single sweep ("breath -> fog -> one stripe pattern").
 * "Freeze & reset" = overwrite in place, move to the next (never-repeated)
 * scale, until all log2(n) scales have been visited exactly once.
 */
void kernel(int n, const double *in, double *out) {
    memcpy(out, in, (size_t)n * sizeof(double));

    for (int len = 1; len < n; len <<= 1) {
        int step = len << 1;
        int nblocks = n / step;

        #pragma omp parallel for schedule(static) if(nblocks > 1 && (long)n > 4096)
        for (int b = 0; b < nblocks; b++) {
            int i = b * step;
            double * restrict pa = out + i;
            double * restrict pb = out + i + len;
            #pragma omp simd
            for (int j = 0; j < len; j++) {
                double a = pa[j];
                double c = pb[j];
                pa[j] = a + c;   /* blade left standing: "+" contribution   */
                pb[j] = a - c;   /* blade pressed flat: "-" contribution    */
            }
        }
    }
}
