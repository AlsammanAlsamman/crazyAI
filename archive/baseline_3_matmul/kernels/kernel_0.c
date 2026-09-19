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
