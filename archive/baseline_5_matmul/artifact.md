# APPROACH

The naive/reference `kernel` (`ikj` order, no blocking) is already reasonably cache-friendly on the `j` axis (row-major `B[k,:]` and `C[i,:]` are scanned contiguously), but it has three problems the standard solution silently accepts:

1. **No explicit SIMD** — it relies on autovectorization of a scalar `+=` loop, which gcc will vectorize but without FMA-aware register blocking or multiple independent accumulators, so it under-uses the ALUs.
2. **No cache blocking on `k`/`j`** — for large `n`, each pass over a `B` row-block and `C` row-block evicts and re-streams from DRAM `n` times (once per `i`), since neither block fits L2.
3. **Single-threaded** — "one processor holds both matrices" is assumed; nothing stops splitting disjoint row-ranges of `C` across cores with zero synchronization.

I attack all three: block the iteration space into `(ii, kk, jj)` tiles sized to fit L2 (`128×256` doubles ≈ 256 KiB for the active `B`/`C` sub-panels, reused across the whole `ii` row-range and `k`-range respectively), vectorize the innermost `j` loop with AVX2 `_mm256_fmadd_pd` (4 doubles/instruction, load-C/load-B/FMA/store-C), and parallelize over the outer `ii` blocks with OpenMP `schedule(dynamic)` — each thread owns disjoint `C` rows for the whole computation, so there's no need for atomics, reductions, or a separate accumulate-then-merge phase.

I deliberately do **not** attempt full A/B panel packing + a register-blocked micro-kernel (the actual OpenBLAS technique) in one shot: without the ability to compile/test in this session, a packing microkernel is high-risk for silent index bugs, and the contract demands *correctness first*. This version keeps memory access patterns simple enough to reason about exactly, while still removing the three biggest inefficiencies of the reference loop.

Known limitation I'm accepting up front: the inner loop does 1 FMA per ~2 loads + 1 store (B read, C read, C write), so it is memory-bandwidth-bound rather than compute-bound — it will *not* match OpenBLAS's packed, register-tiled microkernel (which gets ~1 FMA per load via 2-4 reused accumulators and L1-resident packed panels). It should, however, clearly beat the plain cache-blocked triple loop given in the example.

# ARTIFACT

```c
#include <immintrin.h>
#include <string.h>
#include <omp.h>

#define BI 64
#define BK 128
#define BJ 256

void kernel(int n, const double *A, const double *B, double *C) {
    size_t N = (size_t)n;
    memset(C, 0, N * N * sizeof(double));

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < n; ii += BI) {
        int i_max = (ii + BI < n) ? ii + BI : n;
        for (int kk = 0; kk < n; kk += BK) {
            int k_max = (kk + BK < n) ? kk + BK : n;
            for (int jj = 0; jj < n; jj += BJ) {
                int j_max = (jj + BJ < n) ? jj + BJ : n;
                for (int i = ii; i < i_max; i++) {
                    double *Crow = C + (size_t)i * N;
                    const double *Arow = A + (size_t)i * N;
                    for (int k = kk; k < k_max; k++) {
                        double a = Arow[k];
                        __m256d va = _mm256_set1_pd(a);
                        const double *Brow = B + (size_t)k * N;
                        int j = jj;
                        for (; j + 4 <= j_max; j += 4) {
                            __m256d vb = _mm256_loadu_pd(Brow + j);
                            __m256d vc = _mm256_loadu_pd(Crow + j);
                            vc = _mm256_fmadd_pd(va, vb, vc);
                            _mm256_storeu_pd(Crow + j, vc);
                        }
                        for (; j < j_max; j++) {
                            Crow[j] += a * Brow[j];
                        }
                    }
                }
            }
        }
    }
}
```

Correctness notes: every `C` element is written by exactly one thread (the thread owning its `ii`-block), `A`/`B` are read-only and shared safely, the scalar tail loop handles `n` not a multiple of 4, and all three block-loop upper bounds are clamped to `n` so it's correct for any `n ≥ 0` (including `n=0`, where all loops simply don't execute after the trivially-safe `memset` of 0 bytes).

# PREDICTION: 7

(Predicted speedup factor of this kernel's execution time relative to the reference cache-blocked triple-loop baseline, on the harness's default `n`. I do **not** expect this to beat OpenBLAS — the inner loop is bandwidth-bound, not compute-bound, so I'd expect it to land well under OpenBLAS's throughput, plausibly 3-8x slower than OpenBLAS depending on core count and `n`.)

# MEASUREMENT

No `kernel_bench` (or any other) tool was available in this session — the tool list attached to this conversation is for an unrelated environment (Docs/Gmail/Slack/Drive), and I was explicitly instructed that no benchmarking tools are available here and that "the pipeline will compile and measure it" outside this turn. I have not run the code and have no timing numbers to report; the PREDICTION above is a reasoned estimate only, not a measured value.

# VERDICT

Unverified in this session. Based on reasoning alone: the artifact should be **correct** (single-writer-per-row parallelism, clamped tile bounds, scalar remainder handling) and should **outperform the naive/reference cache-blocked triple loop** by a solid margin from SIMD + blocking + multithreading, but is **not expected to beat OpenBLAS**, since it lacks panel packing and a register-blocked microkernel and is therefore memory-bandwidth-bound rather than compute-bound. Actual verdict is pending the pipeline's own measurement, which I cannot perform here.