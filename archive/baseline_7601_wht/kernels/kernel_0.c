#include <string.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void kernel(int n, const double *in, double *out) {
    if (n <= 0) return;
    memcpy(out, in, (size_t)n * sizeof(double));
    if (n == 1) return;

    int start;
    if (n >= 4) {
        /* Fuse the len=1 and len=2 butterfly stages into a single
           4-point WHT computed entirely in registers per block. */
        for (int i = 0; i < n; i += 4) {
            double a = out[i], b = out[i+1], c = out[i+2], d = out[i+3];
            double ab = a + b, amb = a - b;
            double cd = c + d, cmd = c - d;
            out[i]   = ab + cd;
            out[i+1] = amb + cmd;
            out[i+2] = ab - cd;
            out[i+3] = amb - cmd;
        }
        start = 4;
    } else { /* n == 2 */
        double a = out[0], b = out[1];
        out[0] = a + b;
        out[1] = a - b;
        return;
    }

    for (int len = start; len < n; len <<= 1) {
        int step = len << 1;
        int nblocks = n / step;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(n >= (1 << 16))
        #endif
        for (int b = 0; b < nblocks; b++) {
            int i = b * step;
            int j = i;
            for (; j + 4 <= i + len; j += 4) {
                __m256d u = _mm256_loadu_pd(&out[j]);
                __m256d v = _mm256_loadu_pd(&out[j + len]);
                _mm256_storeu_pd(&out[j],       _mm256_add_pd(u, v));
                _mm256_storeu_pd(&out[j + len], _mm256_sub_pd(u, v));
            }
            for (; j < i + len; j++) {
                double u = out[j];
                double v = out[j + len];
                out[j]       = u + v;
                out[j + len] = u - v;
            }
        }
    }
}
