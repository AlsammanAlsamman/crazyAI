# APPROACH

The naive/cache-blocked triple loop the contract shows is already `ikj`-ordered (good for row-major streaming), but it still: (1) touches C once per (i,k) pair instead of keeping a small tile of C in registers across the whole K reduction, (2) reads B in full-row strides directly from the input array (no reuse across cache levels — for large n, each row of B is evicted from L2 before it's reused by the next `i`), and (3) is single-threaded.

I replace it with a BLIS-style blocked GEMM:

- **Three-level blocking**: `jc` over N in chunks of `NC=512`, `pc` over K in chunks of `KC=256` (sequential — it's the reduction/accumulation dimension), `ic` over M in chunks of `MC=256` (parallelized with OpenMP — disjoint output rows per thread, so no race even though C is accumulated across the sequential `pc` loop).
- **Packing**: for each `(jc,pc)` pair, pack a `KC × NC` panel of B into a contiguous, NR(=8)-padded buffer once, reused by every `ic` block (and every thread). For each `(ic,pc)` pair, each thread packs its own `MC × KC` panel of A into a contiguous, MR(=4)-padded thread-local buffer. This turns strided row-major sub-block access into unit-stride streaming, cutting cache-line waste and TLB pressure.
- **Register-tiled microkernel**: 4×8 output tile held entirely in 8 YMM (AVX2/FMA) registers for the whole `kc` reduction — C is written back only once per tile per `(pc)` block, not once per multiply-add.
- **Threading**: `#pragma omp parallel for` over `ic` blocks with `dynamic` scheduling; each thread owns its own A-pack buffer to avoid false sharing/races.

This is exactly the structure OpenBLAS/BLIS use, just far less tuned (no assembly microkernel, no prefetch scheduling, fixed block sizes instead of cache-size-derived ones, no AVX-512). So I expect it to comfortably beat the cache-blocked triple loop, but not to beat OpenBLAS, which additionally exploits AVX-512/AMX where available, hand-scheduled prefetch, and auto-tuned block sizes per microarchitecture.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#define MC 256
#define KC 256
#define NC 512
#define MR 4
#define NR 8

static void pack_A(const double *A, int n, int ic, int mc, int pc, int kc, double *Apack) {
    int mc_pad = ((mc + MR - 1) / MR) * MR;
    for (int i = 0; i < mc_pad; i++) {
        double *dst = Apack + (size_t)i * kc;
        if (i < mc) {
            memcpy(dst, A + (size_t)(ic + i) * n + pc, (size_t)kc * sizeof(double));
        } else {
            memset(dst, 0, (size_t)kc * sizeof(double));
        }
    }
}

static void pack_B(const double *B, int n, int pc, int kc, int jc, int nc, double *Bpack) {
    int nc_pad = ((nc + NR - 1) / NR) * NR;
    for (int k = 0; k < kc; k++) {
        double *dst = Bpack + (size_t)k * nc_pad;
        memcpy(dst, B + (size_t)(pc + k) * n + jc, (size_t)nc * sizeof(double));
        if (nc_pad > nc) memset(dst + nc, 0, (size_t)(nc_pad - nc) * sizeof(double));
    }
}

static inline void micro_kernel(int kc,
                                 const double *Apack, int lda,
                                 const double *Bpack, int ldb,
                                 double *C, int ldc,
                                 int mr_valid, int nr_valid) {
    __m256d c0 = _mm256_setzero_pd(), c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd(), c3 = _mm256_setzero_pd();
    __m256d c4 = _mm256_setzero_pd(), c5 = _mm256_setzero_pd();
    __m256d c6 = _mm256_setzero_pd(), c7 = _mm256_setzero_pd();

    for (int k = 0; k < kc; k++) {
        __m256d b0 = _mm256_loadu_pd(Bpack + (size_t)k * ldb);
        __m256d b1 = _mm256_loadu_pd(Bpack + (size_t)k * ldb + 4);

        __m256d a0 = _mm256_set1_pd(Apack[0 * lda + k]);
        __m256d a1 = _mm256_set1_pd(Apack[1 * lda + k]);
        __m256d a2 = _mm256_set1_pd(Apack[2 * lda + k]);
        __m256d a3 = _mm256_set1_pd(Apack[3 * lda + k]);

        c0 = _mm256_fmadd_pd(a0, b0, c0); c1 = _mm256_fmadd_pd(a0, b1, c1);
        c2 = _mm256_fmadd_pd(a1, b0, c2); c3 = _mm256_fmadd_pd(a1, b1, c3);
        c4 = _mm256_fmadd_pd(a2, b0, c4); c5 = _mm256_fmadd_pd(a2, b1, c5);
        c6 = _mm256_fmadd_pd(a3, b0, c6); c7 = _mm256_fmadd_pd(a3, b1, c7);
    }

    double tmp[4][8];
    _mm256_storeu_pd(&tmp[0][0], c0); _mm256_storeu_pd(&tmp[0][4], c1);
    _mm256_storeu_pd(&tmp[1][0], c2); _mm256_storeu_pd(&tmp[1][4], c3);
    _mm256_storeu_pd(&tmp[2][0], c4); _mm256_storeu_pd(&tmp[2][4], c5);
    _mm256_storeu_pd(&tmp[3][0], c6); _mm256_storeu_pd(&tmp[3][4], c7);

    for (int r = 0; r < mr_valid; r++)
        for (int c = 0; c < nr_valid; c++)
            C[(size_t)r * ldc + c] += tmp[r][c];
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;
    memset(C, 0, (size_t)n * n * sizeof(double));

    int nthreads = omp_get_max_threads();
    int mc_pad_max = ((MC + MR - 1) / MR) * MR;
    double **Apacks = (double **)malloc(sizeof(double *) * (size_t)nthreads);
    for (int t = 0; t < nthreads; t++)
        Apacks[t] = (double *)aligned_alloc(64, (size_t)mc_pad_max * KC * sizeof(double));

    int nc_pad_max = ((NC + NR - 1) / NR) * NR;
    double *Bpack = (double *)aligned_alloc(64, (size_t)KC * nc_pad_max * sizeof(double));

    for (int jc = 0; jc < n; jc += NC) {
        int nc = (n - jc < NC) ? (n - jc) : NC;
        int nc_pad = ((nc + NR - 1) / NR) * NR;

        for (int pc = 0; pc < n; pc += KC) {
            int kc = (n - pc < KC) ? (n - pc) : KC;
            pack_B(B, n, pc, kc, jc, nc, Bpack);

            #pragma omp parallel for schedule(dynamic)
            for (int ic = 0; ic < n; ic += MC) {
                int tid = omp_get_thread_num();
                double *Apack = Apacks[tid];
                int mc = (n - ic < MC) ? (n - ic) : MC;
                pack_A(A, n, ic, mc, pc, kc, Apack);

                for (int ir = 0; ir < mc; ir += MR) {
                    int mr_valid = (mc - ir < MR) ? (mc - ir) : MR;
                    for (int jr = 0; jr < nc; jr += NR) {
                        int nr_valid = (nc - jr < NR) ? (nc - jr) : NR;
                        micro_kernel(kc,
                                     Apack + (size_t)ir * kc, kc,
                                     Bpack + jr, nc_pad,
                                     C + (size_t)(ic + ir) * n + (jc + jr), n,
                                     mr_valid, nr_valid);
                    }
                }
            }
        }
    }

    for (int t = 0; t < nthreads; t++) free(Apacks[t]);
    free(Apacks);
    free(Bpack);
}
```

# PREDICTION: 60

(Estimated attained double-precision GFLOP/s for a large n, e.g. n=1024–2048, on a typical multi-core AVX2 desktop/server with ~8 cores — expected to land well above the cache-blocked scalar/naive-vectorized triple loop, but below a tuned OpenBLAS run on the same hardware, which I'd expect to reach 2–4× higher.)

# MEASUREMENT

No `kernel_bench` (or any performance-measurement) tool was exposed in this session — the available toolset here was limited to Docs/Gmail/Drive/Slack, not the `kernel_bench`/`symbolic_*`/`unconventional_*` tools listed in the task description. I could not compile, run, or iterate against real hardware, so no empirical GFLOP/s, wall-clock time, or comparison to the cache-blocked baseline / OpenBLAS was obtained in this session. The number above is an analytic estimate from operation counts and typical AVX2 FMA throughput, not a measurement, and should be treated as such until the pipeline actually compiles and benchmarks the artifact.

# VERDICT

Unverified. The implementation is correct by construction (standard BLIS-style packed, register-tiled, multithreaded GEMM; zero-padding on remainder tiles is masked out before the write-back, so it cannot corrupt results), and should be substantially faster than the naive cache-blocked triple loop due to register-resident C tiles, contiguous packed access, and OpenMP parallelism. It is unlikely to beat OpenBLAS outright, since it lacks hand-tuned microkernel assembly, AVX-512/wider vectorization, prefetch scheduling, and auto-tuned block sizes — but no actual measurement was possible in this session to confirm either claim.