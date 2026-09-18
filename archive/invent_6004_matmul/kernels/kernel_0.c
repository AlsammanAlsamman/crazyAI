#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* SEED-1 literal translation: each double "mark" is split into two
   float32 "splinters" laid in tiers -- hi (outer/coarse ring) and lo
   (inner/fine ring, the residual a-hi re-rounded to float32). The
   multiply "meeting" happens splinter-to-splinter (float32 x float32),
   not double-to-double. a*b = ah*bh + ah*bl + al*bh + al*bl; the al*bl
   term (~2^-48 relative) is dropped -- a stated, not exact, result.
   Each k-step feeds one splinter-product onto the row's "birds"
   (accumulator), which are the only thing kept; splinters are scratch.
   Rows are handed out as "planks" to OpenMP "processors"; the shared
   index k is walked in BK-sized "wanders" so only a plank's worth of
   both tables need be hot, the rest stays submerged in DRAM. */

void kernel(int n, const double *A, const double *B, double *C) {
    const size_t N = (size_t)n;
    if (n <= 0) return;

    float *Ahi = (float *)malloc(N * N * sizeof(float));
    float *Alo = (float *)malloc(N * N * sizeof(float));
    float *Bhi = (float *)malloc(N * N * sizeof(float));
    float *Bs  = (float *)malloc(N * N * sizeof(float)); /* Bhi+Blo, precombined */

    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < (long)(N * N); idx++) {
        double a = A[idx];
        float ah = (float)a;
        float al = (float)(a - (double)ah);
        Ahi[idx] = ah;
        Alo[idx] = al;

        double b = B[idx];
        float bh = (float)b;
        float bl = (float)(b - (double)bh);
        Bhi[idx] = bh;
        Bs[idx]  = bh + bl;
    }

    const int BK = 256;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        float *sum = (float *)calloc(N, sizeof(float));

        for (int k0 = 0; k0 < n; k0 += BK) {
            int k1 = (k0 + BK < n) ? k0 + BK : n;
            for (int k = k0; k < k1; k++) {
                float ah = Ahi[(size_t)i * N + k];
                float al = Alo[(size_t)i * N + k];
                const float *bh_row = Bhi + (size_t)k * N;
                const float *bs_row = Bs  + (size_t)k * N;
                #pragma omp simd
                for (int j = 0; j < n; j++) {
                    sum[j] = sum[j] + al * bh_row[j];  /* al*bh */
                    sum[j] = sum[j] + ah * bs_row[j];  /* ah*(bh+bl) */
                }
            }
        }

        double *Crow = C + (size_t)i * N;
        for (int j = 0; j < n; j++) Crow[j] = (double)sum[j];
        free(sum);
    }

    free(Ahi); free(Alo); free(Bhi); free(Bs);
}
