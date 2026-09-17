#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/* "Atom -> electrons": split each double into a float32 high part and a
   float32 low-order residual so that x ~= hi + lo. A later pounce only
   ever touches one electron (hi or lo) at a time. */
static void split_hilo(const double *src, float *hi, float *lo, size_t n2) {
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < (long)n2; t++) {
        double x = src[t];
        float h = (float)x;
        hi[t] = h;
        lo[t] = (float)(x - (double)h);
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    const size_t N  = (size_t)n;
    const size_t N2 = N * N;

    float *Ahi = (float*)malloc(N2 * sizeof(float));
    float *Alo = (float*)malloc(N2 * sizeof(float));
    float *Bhi = (float*)malloc(N2 * sizeof(float));
    float *Blo = (float*)malloc(N2 * sizeof(float));

    split_hilo(A, Ahi, Alo, N2);
    split_hilo(B, Bhi, Blo, N2);

    memset(C, 0, N2 * sizeof(double));

    const int BI = 64;   /* stake-pairs handled by one goose-walk chunk   */
    const int BK = 128;  /* season chunk                                  */
    const int BJ = 256;  /* pot width kept resident while season is walked */

    #pragma omp parallel for schedule(dynamic)
    for (int i0 = 0; i0 < n; i0 += BI) {
        int imax = (i0 + BI < n) ? i0 + BI : n;
        for (int k0 = 0; k0 < n; k0 += BK) {
            int kmax = (k0 + BK < n) ? k0 + BK : n;
            for (int j0 = 0; j0 < n; j0 += BJ) {
                int jmax = (j0 + BJ < n) ? j0 + BJ : n;
                for (int i = i0; i < imax; i++) {
                    const float *Ahi_row = Ahi + (size_t)i * N;
                    const float *Alo_row = Alo + (size_t)i * N;
                    double *Crow = C + (size_t)i * N;
                    for (int k = k0; k < kmax; k++) {
                        /* the goose's foot lands on stake k */
                        float ah = Ahi_row[k];
                        float al = Alo_row[k];
                        const float *Bhi_row = Bhi + (size_t)k * N;
                        const float *Blo_row = Blo + (size_t)k * N;
                        int j = j0;
#if defined(__AVX2__) && defined(__FMA__)
                        __m256 vah = _mm256_set1_ps(ah);
                        __m256 val = _mm256_set1_ps(al);
                        for (; j + 8 <= jmax; j += 8) {
                            /* the cat pounces: three electron-pairs, not one atom-pair */
                            __m256 bh = _mm256_loadu_ps(Bhi_row + j);
                            __m256 bl = _mm256_loadu_ps(Blo_row + j);
                            __m256 prod = _mm256_mul_ps(vah, bh);        /* hi*hi  */
                            prod = _mm256_fmadd_ps(vah, bl, prod);       /* +hi*lo */
                            prod = _mm256_fmadd_ps(val, bh, prod);       /* +lo*hi */
                            __m128 plo = _mm256_castps256_ps128(prod);
                            __m128 phi = _mm256_extractf128_ps(prod, 1);
                            __m256d dlo = _mm256_cvtps_pd(plo);
                            __m256d dhi = _mm256_cvtps_pd(phi);
                            __m256d clo = _mm256_loadu_pd(Crow + j);
                            __m256d chi = _mm256_loadu_pd(Crow + j + 4);
                            clo = _mm256_add_pd(clo, dlo);   /* coins into the pot */
                            chi = _mm256_add_pd(chi, dhi);
                            _mm256_storeu_pd(Crow + j, clo);
                            _mm256_storeu_pd(Crow + j + 4, chi);
                        }
#endif
                        for (; j < jmax; j++) {
                            float bh = Bhi_row[j];
                            float bl = Blo_row[j];
                            double p = (double)(ah * bh) + (double)(ah * bl) + (double)(al * bh);
                            Crow[j] += p;
                        }
                    }
                }
            }
        }
    }

    free(Ahi); free(Alo); free(Bhi); free(Blo);
}
