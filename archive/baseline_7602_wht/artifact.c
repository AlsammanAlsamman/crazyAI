#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void kernel(int n, const double *in, double *out) {
    memcpy(out, in, (size_t)n * sizeof(double));
    if (n <= 1) return;

    for (int len = 1; len < n; len <<= 1) {
        int step = len << 1;
#ifdef _OPENMP
        if (n >= 65536) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < n; i += step) {
                for (int j = 0; j < len; j++) {
                    double *a = out + i + j;
                    double *b = a + len;
                    double u = *a, v = *b;
                    *a = u + v;
                    *b = u - v;
                }
            }
            continue;
        }
#endif
        for (int i = 0; i < n; i += step) {
            double *a = out + i;
            double *b = a + len;
            for (int j = 0; j < len; j++) {
                double u = a[j], v = b[j];
                a[j] = u + v;
                b[j] = u - v;
            }
        }
    }
}
