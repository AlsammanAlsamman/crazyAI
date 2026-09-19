#include <string.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

void kernel(int n, const double *in, double *out) {
    if (out != in) memcpy(out, in, (size_t)n * sizeof(double));

    for (int len = 1; len < n; len <<= 1) {
        int step = len << 1;

        #pragma omp parallel for schedule(static) if(n >= 65536)
        for (int i = 0; i < n; i += step) {
            double *a = out + i;
            double *b = out + i + len;
            #pragma omp simd
            for (int j = 0; j < len; j++) {
                double u = a[j];
                double v = b[j];
                a[j] = u + v;
                b[j] = u - v;
            }
        }
    }
}
