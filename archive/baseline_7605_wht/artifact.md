# APPROACH

The naive reference is O(n²) (direct matrix-vector product with ±1 entries). The known fast solution is the iterative Fast Walsh–Hadamard Transform: log₂(n) butterfly stages, each doing n/2 add/sub pairs, giving O(n log n) work with no divisions or multiplications — just adds and subtracts, which is friendly to both SIMD and multithreading.

To beat a "plain" FWHT implementation further, I:
1. Use the standard in-place iterative butterfly (avoids recursion overhead and enables contiguous-block operations at each stage).
2. Structure the inner loop so the two operands (`a[j]`, `b[j]`) come from two disjoint contiguous ranges — this is exactly the pattern `-O3 -march=native` auto-vectorizes into AVX add/sub pairs, and I reinforce it with `#pragma omp simd`.
3. Parallelize the outer block loop with OpenMP, gated by an `if(n >= 65536)` clause so small transforms (where thread spin-up cost would dominate the O(n log n) work) stay single-threaded, while large transforms get multi-core scaling.
4. Use `memcpy` once up front rather than touching `in`/`out` aliasing issues, so the transform is safely in-place afterward and works whether or not `in == out`.

This keeps the algorithmic shape identical to the "known way" (butterfly, XOR-structured) but adds vectorization + threading on top, which the naive recursive/scalar reference typically lacks.

# ARTIFACT

```c
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
```

PREDICTION: 2.5

# MEASUREMENT

No `wht_bench` tool call is available in this session (tool list is provided but disabled here per the environment note), so I cannot produce actual timing numbers myself. The implementation above is handed to the external pipeline, which will compile it with `gcc -O3 -march=native -fopenmp -lm` and benchmark it against the reference. I could not run the four allowed measure/improve iterations locally, so no empirical data can be reported at this time.

# VERDICT

Unverified locally. Algorithmically this is a correct O(n log n) FWHT (in-place iterative butterfly, XOR-additive structure matching `out[k] = Σ in[j]·(-1)^popcount(j&k)`), with vectorization-friendly disjoint-range inner loops and threshold-gated OpenMP parallelism added on top of the "known way." Expected outcome: parity or modest speedup (~2–3x, per prediction) over a plain scalar/recursive FWHT reference on large n from SIMD+threads, with negligible/no regression on small n due to the OpenMP threshold guard — but this is a prediction, not a measured result, since the benchmarking tool was unavailable in this session.