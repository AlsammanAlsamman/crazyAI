## APPROACH

The reference computes an O(n²) direct sum. The known fast solution is the Fast Walsh-Hadamard Transform (FWHT): a butterfly of log₂(n) stages, each doing n/2 independent (add, sub) pairs — O(n log n) total, no multiplications, no data reordering needed (this iterative form already produces natural-order output matching `out[k] = Σ in[j]·(-1)^popcount(j&k)`).

To beat a plain scalar FWHT baseline I apply three standard, low-risk optimizations on top of the same algorithm (same assumptions exploited, no new mathematical trick):

1. **Stage fusion for the smallest butterfly distances (len=1,2 → one fused 4-point kernel).** This halves the number of full sweeps over memory for the first two stages and keeps everything in registers, avoiding two separate memory-bound passes.
2. **AVX2 vectorization** of the inner butterfly loop for `len ≥ 4`: `u=load(out+j)`, `v=load(out+j+len)`, store `u+v` and `u-v` — both operands are contiguous per stage, so this vectorizes trivially with 4 doubles/instruction.
3. **OpenMP parallelization** over independent blocks at each stage, gated behind a size threshold (`n ≥ 2^16`) so small transforms don't pay thread-launch overhead.

This does not change asymptotic complexity but reduces the constant factor via SIMD width and multicore, and cuts memory passes slightly via fusion — a legitimate response to the "each output is independent" / "no reuse between outputs" assumptions the naive contract relies on, which the butterfly already exploits, plus the "combining always means ordinary addition" assumption (still true here — I don't change the operation, only its vector width and scheduling).

## ARTIFACT

```c
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
```

Correctness sanity-checked by hand for n=2,4 against the direct-sum reference (verified out[0..3] for the fused 4-point block match `Σ in[j]·(-1)^popcount(j&k)` exactly for all k=0..3); the remaining stages are the textbook in-place FWHT recurrence, which is well established to produce natural-order output.

PREDICTION: 2.5

## MEASUREMENT

No `wht_bench`/tool access was available in this session (tool list was declared but unusable here — explicitly noted: "No tools are available in this session"). I could not execute an actual timing run against the scalar FWHT baseline or the O(n²) reference myself; the prediction above (~2.5× over a plain scalar FWHT, for large n where AVX2 + OpenMP kick in) is an estimate based on: 4-wide AVX2 giving up to ~4× on the vectorizable stages, offset by the fact that early stages (len<4) and thread-launch/sync overhead don't benefit, and the transform is memory-bandwidth-bound at large n which caps realistic speedup well below the raw SIMD width. For small n (below the OpenMP threshold, n<4), essentially only the SIMD fusion applies, so speedup there should be modest (~1.2–1.5×). The external pipeline is expected to compile and measure this artifact against the reference/baseline to confirm or refute this number.

## VERDICT

Implementation is complete, contract-conformant (`void kernel(int n, const double *in, double *out)`, no signature changes), and algorithmically correct (O(n log n) FWHT verified by hand-tracing n=2 and n=4 against the direct-sum definition). Performance claim is a **prediction only** — I was not able to run `wht_bench` or any measurement tool in this session to validate it, so the MEASUREMENT section above is honestly reported as unmeasured/estimated rather than fabricated, and the actual speedup (or possible regression from OpenMP overhead at the threshold boundary, or from AVX2 not being enabled if `-march=native` doesn't include AVX2 on the target) should be confirmed by the pipeline's own compile-and-measure step.