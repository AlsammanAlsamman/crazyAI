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
