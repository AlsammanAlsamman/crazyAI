#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void kernel(int n, const double *in, double *out) {
    if (n <= 0) return;
    memcpy(out, in, (size_t)n * sizeof(double));
    double * restrict a = out;

    for (int len = 1; len < n; len <<= 1) {
        int step = len << 1;
        if (n >= (1 << 16)) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i += step) {
                double * restrict p = a + i;
                double * restrict q = a + i + len;
                for (int j = 0; j < len; j++) {
                    double u = p[j];
                    double v = q[j];
                    p[j] = u + v;
                    q[j] = u - v;
                }
            }
        } else {
            for (int i = 0; i < n; i += step) {
                double * restrict p = a + i;
                double * restrict q = a + i + len;
                for (int j = 0; j < len; j++) {
                    double u = p[j];
                    double v = q[j];
                    p[j] = u + v;
                    q[j] = u - v;
                }
            }
        }
    }
}
