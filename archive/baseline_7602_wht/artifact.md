# APPROACH

The reference computes each of the n outputs as an independent O(n) dot product against a row of the Hadamard matrix → O(n²) total. The known fast algorithm replaces this with the standard FWHT butterfly: at each of log₂(n) stages, pair up elements `n/2` apart (in bit-position `len`) and replace `(a,b)` with `(a+b, a−b)`. This reuses partial sums across stages instead of recomputing full dot products, giving O(n log n) work using only additions/subtractions — exactly matching the reference's XOR/popcount matrix by induction on the bit-decomposition of `k`.

I implement it **iteratively** (not recursively) in-place on `out`:
1. `memcpy(in → out)` once (O(n)), since `out` may be uninitialized and `in` must stay untouched.
2. For `len = 1, 2, 4, ... < n`: process disjoint blocks of size `2*len`, butterflying `out[i+j]` with `out[i+j+len]`.
3. Inner loop is unit-stride on two contiguous double arrays → trivially auto-vectorized by GCC with `-O3 -march=native` (AVX2/AVX-512 add+sub, no gather/scatter needed).
4. For large `n`, parallelize with OpenMP using `collapse(2)` so that every stage — including the last one where there are only 1–2 blocks but each block is huge — gets full, even load-balancing across threads (a plain `parallel for` on the outer block loop alone would starve threads on the last few stages).
5. For small `n`, skip OpenMP entirely (thread/parallel-region overhead would dominate the trivial work).

This avoids: recursion/call overhead of the "textbook" recursive butterfly, and the O(n²) redundant work of the naive reference.

# ARTIFACT

```c
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
```

# PREDICTION: 1.8

# MEASUREMENT

No tools were available in this session (the tool list shown is not actually invokable here), so I could not run `wht_bench`/`wht_contract` myself. Per the session instructions, reasoning and prediction are recorded now, before any measurement; the pipeline is expected to compile with `gcc -O3 -march=native -fopenmp -lm` and benchmark this kernel against the naive O(n²) reference and the recursive-butterfly baseline. I did not fabricate numbers here.

Expected qualitative outcome, reasoning from the algorithm's structure:
- vs. the O(n²) naive reference: enormous speedup (grows as n/log₂n), e.g. ~50,000× at n=2²⁰.
- vs. the "known" recursive FWHT baseline: smaller but real speedup, from (a) eliminating recursive call/stack overhead, (b) fully unit-stride, auto-vectorizable inner loops, and (c) OpenMP parallelism that stays load-balanced even in the last stages via `collapse(2)`. For small n the serial path should be at least as fast as a naive recursive version; for large n the OpenMP path should scale with core count.

# VERDICT

Correctness: the iterative in-place butterfly is the standard textbook transform (equivalent by induction to `out[k] = Σ_j in[j]·(−1)^popcount(j&k)`), copy-then-in-place handles the "out may be uninitialized"/"in is const" constraints correctly, and `n=1` is handled as a no-op copy. I'm confident this is correct and asymptotically optimal in shape (O(n log n), additions/subtractions only, matching the "known way"); the only open question — since I couldn't execute the benchmark tools in this session — is the exact realized speedup factor over the recursive baseline, which the pipeline's measurement will determine.