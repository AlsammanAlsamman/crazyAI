## APPROACH

The naive kernel computes each of the n outputs as an independent O(n) dot product against a row of the Hadamard matrix — O(n²) total, and it never reuses partial sums between outputs even though adjacent rows of a Hadamard matrix are built from the same half-size Hadamard matrix (H_{2m} = [[H_m, H_m],[H_m, -H_m]]).

The fast Walsh–Hadamard transform exploits exactly that recursive block structure: instead of computing sums independently, do log₂(n) "butterfly" stages, each pass over the array combining pairs of elements separated by a fixed stride with only an add and a subtract. This is algorithmically identical to an FFT but with the twiddle factors reduced to ±1 (XOR-domain instead of frequency-domain), giving O(n log n) work with no multiplications at all — pure adds/subtracts, which also vectorizes and parallelizes trivially since within a stage all butterfly pairs are independent.

Implementation choices:
- In-place, iterative (not recursive) butterfly directly on `out`, after copying `in` → `out` once. No extra scratch buffer needed (satisfies "out may be uninitialised on entry").
- Inner loop over contiguous blocks of `len` elements offset by `len` — this is exactly the pattern `-O3 -march=native` auto-vectorizes well with AVX since both `a[j..j+len)` and `a[j+len..j+2*len)` are contiguous.
- OpenMP over the outer block index only for large n (avoids parallel-region overhead dominating on small inputs, and each stage's blocks are fully independent so no synchronization races).
- restrict-qualified pointer to help the vectorizer.

Complexity: O(n log n) vs. reference O(n²) — the gap should be enormous for n in the thousands-to-millions range typically used in such benchmarks.

## ARTIFACT

```c
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
```

## PREDICTION: 2000

## MEASUREMENT

No measurement tools (`wht_bench`, etc.) were exposed in this session, so I could not actually compile/run/profile this kernel against the reference. The prediction above (~2000×) is a reasoned estimate, not an observed number: for n in the ~10^5–10^6 range the naive kernel does O(n²) ≈ 10^10–10^12 scalar multiply-adds while the FWHT does O(n log n) ≈ n·20 flops of pure add/sub — a raw operation-count ratio of 10^4–10^5×, discounted heavily here because the naive reference's inner loop is simple and auto-vectorizes/parallelizes reasonably well too, and because real wall-clock speedup is capped by memory bandwidth on the butterfly's O(log n) full-array passes rather than by op count alone.

## VERDICT

Implementation is complete and, by construction, algorithmically correct (standard in-place iterative FWHT, verified in the docstring-equivalent form countless times: it reduces to the same recursive H_{2m} block decomposition as the reference's closed-form popcount sum). However, since no benchmarking tool was available in this session, I cannot confirm the predicted ~2000× speedup was actually achieved — that measurement is deferred to the external pipeline mentioned in the instructions. Reporting this limitation plainly rather than fabricating a measured number.