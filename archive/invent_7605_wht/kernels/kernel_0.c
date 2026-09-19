#include <immintrin.h>
#include <omp.h>

void kernel(int n, const double *in, double *out) {
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        __m256d vsum = _mm256_setzero_pd();
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            /* one token per nail, sun-side-up (+1) or iron-blade-down (-1),
               read directly off the hardware -- never a bit-by-bit count */
            int p0 = __builtin_parity((unsigned)((j + 0) & k));
            int p1 = __builtin_parity((unsigned)((j + 1) & k));
            int p2 = __builtin_parity((unsigned)((j + 2) & k));
            int p3 = __builtin_parity((unsigned)((j + 3) & k));
            double s0 = p0 ? -1.0 : 1.0;
            double s1 = p1 ? -1.0 : 1.0;
            double s2 = p2 ? -1.0 : 1.0;
            double s3 = p3 ? -1.0 : 1.0;

            __m256d vin   = _mm256_loadu_pd(&in[j]);
            __m256d vsign = _mm256_set_pd(s3, s2, s1, s0);
            /* let the flood settle: multiply-accumulate, not a running
               chain of separate additions */
            vsum = _mm256_add_pd(vsum, _mm256_mul_pd(vsign, vin));
        }

        double buf[4];
        _mm256_storeu_pd(buf, vsum);
        double sum = buf[0] + buf[1] + buf[2] + buf[3];

        for (; j < n; j++) {
            int p = __builtin_parity((unsigned)(j & k));
            sum += (p ? -1.0 : 1.0) * in[j];
        }

        /* score the single number onto the room, once, and never
           revisit it -- no state carried to the next k */
        out[k] = sum;
    }
}
