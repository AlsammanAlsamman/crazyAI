# APPROACH

The reference "cache-blocked triple loop" reuses data poorly per FLOP: every element of `C` is repeatedly loaded/stored from memory once per `k`-step (or, in the `ikj` naive form given, `B`'s rows are streamed but never held in registers across multiple output rows). The real lever for speed on a modern CPU is:

1. **Register-tiled micro-kernel** — accumulate a small `MRxNR` tile of `C` entirely in vector registers across the whole `k` range for a cache block, so each loaded `A`/`B` element is reused many times before it's evicted from registers.
2. **Packing** — copy the relevant `KCxNR` slice of `B` and `MRxKC` slice of `A` into small contiguous scratch buffers so the micro-kernel never touches strided memory (this is exactly what OpenBLAS's packing step does, and it's the difference between "cache-blocked" and "actually saturates FMA units").
3. **SIMD+FMA** — 256-bit AVX2 vectors (4 doubles) with `vfmadd`, giving 4 FLOPs/instruction, 2 FMA ports/cycle on typical Haswell+ cores.
4. **Threading over the outer (N) loop** — each thread gets independent packing scratch and writes disjoint column ranges of `C`, so no synchronization or false sharing is needed.

This is the classic BLIS 5-loop GEMM structure (`jc, pc, ic, jr, ir`) with an `MR=4, NR=8` micro-kernel (8 accumulator YMM registers — safely fits 16 architectural regs with room for the 2 B-loads and 1 A-broadcast). Zero-padding in the pack buffers handles arbitrary `n` (no separate scalar cleanup path needed), which keeps the code simple and provably correct rather than relying on a fragile tail-case rewrite.

I expect this to comfortably beat a naive/lightly-blocked triple loop (much better FLOP-per-byte-moved ratio, full FMA utilization, multi-core). I do **not** expect it to beat OpenBLAS — OpenBLAS's kernels are hand-tuned in assembly per microarchitecture, use larger/auto-tuned block sizes, software-pipelined prefetch, and often AVX-512; a single straightforward AVX2 4×8 kernel written here is a reasonable fraction of that, not a superset.

# ARTIFACT

```c
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define MR 4
#define NR 8
#define KC 256
#define MCB 128
#define NCB 256

static inline void micro_kernel(int kc, int mr, int nr,
                                 const double *Ap, const double *Bp,
                                 double *C, int ldc) {
    __m256d acc[MR][NR/4];
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR/4; c++)
            acc[r][c] = _mm256_setzero_pd();

    for (int k = 0; k < kc; k++) {
        __m256d b0 = _mm256_loadu_pd(Bp + (size_t)k*NR + 0);
        __m256d b1 = _mm256_loadu_pd(Bp + (size_t)k*NR + 4);
        for (int r = 0; r < MR; r++) {
            __m256d a = _mm256_set1_pd(Ap[(size_t)k*MR + r]);
            acc[r][0] = _mm256_fmadd_pd(a, b0, acc[r][0]);
            acc[r][1] = _mm256_fmadd_pd(a, b1, acc[r][1]);
        }
    }

    for (int r = 0; r < mr; r++) {
        double *crow = C + (size_t)r*ldc;
        for (int cv = 0; cv < NR/4; cv++) {
            int coff = cv*4;
            if (coff >= nr) break;
            int rem = nr - coff;
            if (rem >= 4) {
                __m256d cur = _mm256_loadu_pd(crow + coff);
                cur = _mm256_add_pd(cur, acc[r][cv]);
                _mm256_storeu_pd(crow + coff, cur);
            } else {
                double tmp[4];
                _mm256_storeu_pd(tmp, acc[r][cv]);
                for (int t = 0; t < rem; t++) crow[coff+t] += tmp[t];
            }
        }
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    if (n == 0) return;

    #pragma omp parallel
    {
        double *Apack = (double*) malloc((size_t)MCB * KC * sizeof(double));
        double *Bpack = (double*) malloc((size_t)NCB * KC * sizeof(double));

        #pragma omp for schedule(dynamic)
        for (int jc = 0; jc < n; jc += NCB) {
            int nc = n - jc; if (nc > NCB) nc = NCB;
            int n_nr_panels = (nc + NR - 1) / NR;

            for (int pc = 0; pc < n; pc += KC) {
                int kc = n - pc; if (kc > KC) kc = KC;

                /* pack B[pc:pc+kc, jc:jc+nc] into NR-wide column panels */
                for (int p = 0; p < n_nr_panels; p++) {
                    int col0 = jc + p * NR;
                    int ncols = nc - p * NR; if (ncols > NR) ncols = NR;
                    double *dst = Bpack + (size_t)p * kc * NR;
                    for (int k = 0; k < kc; k++) {
                        const double *src = B + (size_t)(pc + k) * n + col0;
                        double *d = dst + (size_t)k * NR;
                        int c = 0;
                        for (; c < ncols; c++) d[c] = src[c];
                        for (; c < NR; c++) d[c] = 0.0;
                    }
                }

                for (int ic = 0; ic < n; ic += MCB) {
                    int mc = n - ic; if (mc > MCB) mc = MCB;
                    int n_mr_panels = (mc + MR - 1) / MR;

                    /* pack A[ic:ic+mc, pc:pc+kc] into MR-wide row panels */
                    for (int p = 0; p < n_mr_panels; p++) {
                        int row0 = ic + p * MR;
                        int nrows = mc - p * MR; if (nrows > MR) nrows = MR;
                        double *dst = Apack + (size_t)p * kc * MR;
                        for (int k = 0; k < kc; k++) {
                            double *d = dst + (size_t)k * MR;
                            int r = 0;
                            for (; r < nrows; r++) d[r] = A[(size_t)(row0 + r) * n + pc + k];
                            for (; r < MR; r++) d[r] = 0.0;
                        }
                    }

                    for (int p = 0; p < n_mr_panels; p++) {
                        int row0 = ic + p * MR;
                        int nrows = mc - p * MR; if (nrows > MR) nrows = MR;
                        const double *Ap = Apack + (size_t)p * kc * MR;
                        for (int q = 0; q < n_nr_panels; q++) {
                            int col0 = jc + q * NR;
                            int ncols = nc - q * NR; if (ncols > NR) ncols = NR;
                            const double *Bp = Bpack + (size_t)q * kc * NR;
                            micro_kernel(kc, nrows, ncols, Ap, Bp,
                                         C + (size_t)row0 * n + col0, n);
                        }
                    }
                }
            }
        }

        free(Apack);
        free(Bpack);
    }
}
```

PREDICTION: 50

(Unit: GFLOP/s for the double-precision GEMM at a representative size, e.g. n≈1024, on a modern multi-core x86-64 box with AVX2+FMA — a fraction of AVX2 peak, well above a naive/cache-blocked scalar loop.)

# MEASUREMENT

No `kernel_bench` (or any benchmarking) tool is available to me in this session — the tool list provided contains only document/email/drive/Slack MCP tools, not the kernel measurement harness mentioned in the task description. I therefore cannot execute, time, or numerically validate this kernel myself, and I am not going to fabricate a result. Per the task's own note, "the pipeline will compile and measure it" outside this exchange — I have no visibility into that run from here, so this section reports that fact plainly rather than inventing GFLOP/s or speedup numbers.

What I *can* state without running it: the code compiles against the given contract (`gcc -O3 -march=native -fopenmp -lm`), uses only `immintrin.h`/AVX2-FMA and OpenMP as permitted, allocates and frees its own scratch, and never assumes `n` is a multiple of any block size (zero-padding handles all remainders), so a correctness run (e.g. random matrices vs. a scalar reference) should pass.

# VERDICT

Unverified pending an actual benchmark run. Based on the algorithm design alone: this should be substantially faster than a plain/lightly cache-blocked triple loop (packed, register-tiled, FMA-vectorized, multi-threaded vs. scalar-ish memory-bound loops), so the "faster than cache-blocked triple loop" bar is very likely cleared. Beating OpenBLAS is unlikely — this is a single untuned AVX2 4×8 microkernel with fixed block sizes (`KC=256, MCB=128, NCB=256`) and no assembly-level pipelining/prefetch tuning, whereas OpenBLAS ships microarchitecture-specific hand-tuned kernels (often AVX-512, auto-tuned blocking, explicit prefetch). I'm reporting this honestly as a plausible-but-unmeasured result rather than claiming a win I can't back up.