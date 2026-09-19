#include <string.h>
#include <omp.h>

#define WHT_BASE        2048      /* block size that stays cache-resident */
#define WHT_PAR_THRESH  (1 << 16) /* only spawn OpenMP tasks above this size */

/* bottom-up butterfly on a self-contained, cache-resident block */
static void fwht_iter(double *a, long long n) {
    for (long long len = 1; len < n; len <<= 1) {
        long long step = len << 1;
        for (long long i = 0; i < n; i += step) {
            double *p = a + i;
            double *q = p + len;
            for (long long j = 0; j < len; j++) {
                double u = p[j];
                double v = q[j];
                p[j] = u + v;
                q[j] = u - v;
            }
        }
    }
}

/* top-down, cache-oblivious: largest-stride butterfly first, then recurse
   into two independent contiguous halves (stage order is irrelevant to
   the final WHT result, only the disjoint-bit-position structure matters) */
static void fwht_rec(double *a, long long n) {
    if (n <= WHT_BASE) {
        fwht_iter(a, n);
        return;
    }
    long long h = n >> 1;
    for (long long i = 0; i < h; i++) {
        double u = a[i] + a[i + h];
        double v = a[i] - a[i + h];
        a[i]     = u;
        a[i + h] = v;
    }
    if (n > WHT_PAR_THRESH) {
        #pragma omp task
        fwht_rec(a, h);
        #pragma omp task
        fwht_rec(a + h, h);
        #pragma omp taskwait
    } else {
        fwht_rec(a, h);
        fwht_rec(a + h, h);
    }
}

void kernel(int n, const double *in, double *out) {
    if (n <= 0) return;
    memcpy(out, in, (size_t)n * sizeof(double));
    if (n <= 1) return;

    long long nn = (long long)n;
    if (nn > WHT_PAR_THRESH) {
        #pragma omp parallel
        {
            #pragma omp single
            fwht_rec(out, nn);
        }
    } else {
        fwht_rec(out, nn);
    }
}
