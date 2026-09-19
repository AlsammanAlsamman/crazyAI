## APPROACH

The naive/cache-blocked triple loop leaves three big things on the table: (1) no data packing, so both A and B are walked with non-unit or large strides most of the time, defeating prefetch and TLB reach; (2) no register blocking, so the inner loop only ever keeps one C accumulator live instead of a whole tile, wasting FMA throughput; (3) single-threaded.

I'll write a small BLIS-style GEMM:

- Partition the M×N output into independent (MC×NC) rectangles — one per OpenMP task. Since rectangles are disjoint, no locking/atomics are needed on C.
- For each rectangle, loop over K in chunks of KC, packing the relevant A sub-block into MR-row panels and the relevant B sub-block into NR-column panels (contiguous, zero-padded at the edges) into small thread-local scratch buffers.
- A 6×8 register microkernel (MR=6, NR=8) computes a full tile using AVX2 FMA: 12 accumulator YMM registers hold the C tile while broadcasting one A scalar at a time across two 4-wide B vectors — the classic Haswell-class double-precision microkernel shape.
- Edge tiles (M or N not a multiple of MR/NR) are computed into a small stack buffer and copied back with a scalar loop so no out-of-bounds/partition-violating writes occur.
- Threads get their own pack buffers (`_mm_malloc`, one pair per thread), so there's no shared mutable state besides the disjoint C rectangles.

This should beat the given cache-blocked-triple-loop baseline by a large margin (packing + register blocking + threads), but is very unlikely to beat a mature, hand-tuned, multi-microarchitecture OpenBLAS (which additionally uses AVX-512, nested threading, and far more exhaustively tuned block sizes per cache level).

## ARTIFACT

```c
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define MR 6
#define NR 8
#define MC 240
#define KC 256
#define NC 512

static void pack_A(const double *A, int n, int i0, int k0, int mc, int kc, double *Apack) {
    int mc_pad = ((mc + MR - 1) / MR) * MR;
    for (int p = 0; p < mc_pad; p += MR) {
        double *dst = Apack + (size_t)(p / MR) * kc * MR;
        for (int kk = 0; kk < kc; kk++) {
            for (int r = 0; r < MR; r++) {
                int row = p + r;
                dst[kk * MR + r] = (row < mc) ? A[(size_t)(i0 + row) * n + (k0 + kk)] : 0.0;
            }
        }
    }
}

static void pack_B(const double *B, int n, int k0, int j0, int kc, int nc, double *Bpack) {
    int nc_pad = ((nc + NR - 1) / NR) * NR;
    for (int q = 0; q < nc_pad; q += NR) {
        double *dst = Bpack + (size_t)(q / NR) * kc * NR;
        for (int kk = 0; kk < kc; kk++) {
            for (int c = 0; c < NR; c++) {
                int col = q + c;
                dst[kk * NR + c] = (col < nc) ? B[(size_t)(k0 + kk) * n + (j0 + col)] : 0.0;
            }
        }
    }
}

static inline void microkernel(const double *Apanel, const double *Bpanel, int kc,
                                double *C, int ldc, int mr_valid, int nr_valid) {
    __m256d acc[MR][2];
    for (int r = 0; r < MR; r++) { acc[r][0] = _mm256_setzero_pd(); acc[r][1] = _mm256_setzero_pd(); }
    for (int kk = 0; kk < kc; kk++) {
        __m256d b0 = _mm256_loadu_pd(Bpanel + (size_t)kk * NR);
        __m256d b1 = _mm256_loadu_pd(Bpanel + (size_t)kk * NR + 4);
        const double *arow = Apanel + (size_t)kk * MR;
        for (int r = 0; r < MR; r++) {
            __m256d av = _mm256_set1_pd(arow[r]);
            acc[r][0] = _mm256_fmadd_pd(av, b0, acc[r][0]);
            acc[r][1] = _mm256_fmadd_pd(av, b1, acc[r][1]);
        }
    }
    double tile[MR][NR];
    for (int r = 0; r < MR; r++) {
        _mm256_storeu_pd(&tile[r][0], acc[r][0]);
        _mm256_storeu_pd(&tile[r][4], acc[r][1]);
    }
    for (int r = 0; r < mr_valid; r++) {
        double *crow = C + (size_t)r * ldc;
        for (int c = 0; c < nr_valid; c++) crow[c] += tile[r][c];
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    if (n <= 0) return;
    int M = n, N = n, K = n;

    int n_ic = (M + MC - 1) / MC;
    int n_jc = (N + NC - 1) / NC;
    int ntasks = n_ic * n_jc;
    if (ntasks == 0) return;

    int max_threads = omp_get_max_threads();
    double **Apacks = (double **)malloc(sizeof(double *) * max_threads);
    double **Bpacks = (double **)malloc(sizeof(double *) * max_threads);
    size_t apack_sz = (size_t)MC * KC;
    size_t bpack_sz = (size_t)KC * NC;
    for (int t = 0; t < max_threads; t++) {
        Apacks[t] = (double *)_mm_malloc(apack_sz * sizeof(double), 64);
        Bpacks[t] = (double *)_mm_malloc(bpack_sz * sizeof(double), 64);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int task = 0; task < ntasks; task++) {
        int tid = omp_get_thread_num();
        double *Apack = Apacks[tid];
        double *Bpack = Bpacks[tid];

        int ic_idx = task / n_jc;
        int jc_idx = task % n_jc;
        int ic0 = ic_idx * MC;
        int jc0 = jc_idx * NC;
        int mc = (ic0 + MC <= M) ? MC : (M - ic0);
        int nc = (jc0 + NC <= N) ? NC : (N - jc0);

        for (int k0 = 0; k0 < K; k0 += KC) {
            int kc = (k0 + KC <= K) ? KC : (K - k0);
            pack_A(A, n, ic0, k0, mc, kc, Apack);
            pack_B(B, n, k0, jc0, kc, nc, Bpack);

            int mc_pad = ((mc + MR - 1) / MR) * MR;
            int nc_pad = ((nc + NR - 1) / NR) * NR;

            for (int q = 0; q < nc_pad; q += NR) {
                int nr_valid = (q + NR <= nc) ? NR : (nc - q);
                const double *Bpanel = Bpack + (size_t)(q / NR) * kc * NR;
                for (int p = 0; p < mc_pad; p += MR) {
                    int mr_valid = (p + MR <= mc) ? MR : (mc - p);
                    const double *Apanel = Apack + (size_t)(p / MR) * kc * MR;
                    double *Cptr = C + (size_t)(ic0 + p) * n + (jc0 + q);
                    microkernel(Apanel, Bpanel, kc, Cptr, n, mr_valid, nr_valid);
                }
            }
        }
    }

    for (int t = 0; t < max_threads; t++) {
        _mm_free(Apacks[t]);
        _mm_free(Bpacks[t]);
    }
    free(Apacks);
    free(Bpacks);
}
```

PREDICTION: 12

## MEASUREMENT

No benchmarking tool was available in this session (the environment note states no tools are available here and that the external pipeline will compile and run `kernel_bench`), so I cannot report an actually-measured number myself. Based on reasoning only:

- vs. the given cache-blocked triple loop: packing + a 6×8 FMA register tile + OpenMP across cores should plausibly give roughly an order of magnitude or more speedup (hence PREDICTION: 12) on a multi-core AVX2 box for moderate-to-large n; for very small n the fixed packing overhead may erode much of that gain.
- vs. OpenBLAS: expect to land below it, likely in the 0.3×–0.7× range of OpenBLAS's GFLOPS, since OpenBLAS additionally exploits AVX-512 where available, more refined block-size auto-tuning per cache size, and more sophisticated multi-level threading — none of which this kernel attempts.

I am flagging this explicitly as an untested prediction rather than a measured result.

## VERDICT

Unverified in this session. The implementation is a correct, reasonably well-structured packed/register-tiled/threaded GEMM that should clearly outperform the naive cache-blocked triple loop, but it was not run, so I cannot claim it beats OpenBLAS — only that it was designed with that goal in mind and is likely to fall short of it given OpenBLAS's much deeper tuning. If actual measurement (via the pipeline) shows otherwise, that measured number should override this prediction.