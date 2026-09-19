# APPROACH

The reference computes each `out[k]` as an independent length-n dot product against a row of the Hadamard matrix — O(n²) work, and it recomputes the same partial sums for every k. The Walsh-Hadamard transform is exactly a Kronecker product of n/log₂n copies of the 2×2 Hadamard matrix, so it factors into log₂n "butterfly" stages of pure add/subtract, O(n log n) total, with **no multiplication at all**.

Key structural point I use beyond the plain textbook FWHT: the order in which the log₂n butterfly stages are applied is irrelevant to the final result (unlike FFT, there's no twiddle factor and no bit-reversal requirement — every stage is a ±1 combination on disjoint bit positions, so the stages commute). That lets me pick a **cache-oblivious, top-down recursive order** (apply the largest-stride butterfly first, then recurse independently into the two contiguous halves) instead of the usual bottom-up iterative order that repeatedly strides across the whole array with large strides at *every* stage-boundary crossing. Once a recursive sub-block's size drops below a cache-resident threshold, all further stages for that block are done with zero further cache misses, switching to an unrolled bottom-up loop (cheaper than recursing all the way to n=1). For large n the two independent recursive halves are also embarrassingly parallel, so I hand them to OpenMP tasks above a size threshold (avoiding thread/task overhead for small inputs).

I verified by hand (n=4, symbolic butterfly expansion) that both the top-down recursive order and the bottom-up iterative order reproduce `out[k] = Σ_j in[j]·(-1)^popcount(j&k)` exactly, confirming correctness of reordering the stages.

# ARTIFACT

```c
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
```

# PREDICTION: 1.4

(Predicted speedup of this implementation over a plain, non-blocked, single-threaded O(n log n) iterative FWHT ("the known way") at large n — driven by the cache-oblivious blocking reducing memory traffic/misses at large-stride stages, plus OpenMP parallelism kicking in above 65536 elements. Against the O(n²) naive reference the speedup is expected to be many orders of magnitude larger for any non-trivial n, so 1.4 is the honest, conservative number for the comparison that actually stresses this design — improved algorithm vs. improved algorithm, not vs. the quadratic baseline.)

# MEASUREMENT

No benchmarking tools (`wht_bench` or any other) were available in this session — the tool list was declared present in the prompt, but the environment explicitly stated "No tools are available in this session" and instructed me to reason, predict, and hand off the artifact for the external pipeline to compile and measure. I did not fabricate numbers. The pipeline referenced in the task ("the pipeline will compile and measure it") is expected to fill in actual timings against both the O(n²) reference and the plain FWHT baseline; I have no measured data to report here.

# VERDICT

Correctness: verified by hand-expansion for n=4 that both the top-down recursive butterfly order and the standard bottom-up order reproduce the exact contract `out[k] = Σ_j in[j]·(-1)^popcount(j&k)`; the general argument (stages act on disjoint bit positions and commute, unlike FFT which needs bit-reversal) extends this to all power-of-two n. The base case (`fwht_iter`) is the textbook correct bottom-up FWHT restricted to a contiguous power-of-two sub-block, which is self-contained and correct in isolation.

Performance: unmeasured in this session due to tool unavailability — the prediction above (1.4×) is an analytical estimate, not a measured result, and should be treated with the same skepticism as any unverified claim until the external pipeline reports actual numbers. If the measured speedup comes back near 1.0 or negative for small/medium n (where OpenMP and recursion overhead could dominate over the simple iterative loop the compiler already autovectorizes well), that would be a legitimate negative outcome worth reporting plainly rather than explained away.