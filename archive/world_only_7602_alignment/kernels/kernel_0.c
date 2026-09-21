#include <stdlib.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int *prev = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *cand = (int*)malloc((size_t)(n + 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    for (int i = 1; i <= n; i++) {
        cur[0] = i * GAP;
        char ai = a[i - 1];
        int j = 1;

#if defined(__AVX2__)
        {
            __m256i vmatch    = _mm256_set1_epi32(MATCH);
            __m256i vmismatch = _mm256_set1_epi32(MISMATCH);
            __m256i vgap      = _mm256_set1_epi32(GAP);
            __m128i vai8      = _mm_set1_epi8(ai);

            for (; j + 7 <= n; j += 8) {
                __m128i bbytes   = _mm_loadl_epi64((const __m128i*)(b + j - 1));
                __m128i eqmask8  = _mm_cmpeq_epi8(bbytes, vai8);
                __m256i eqmask32 = _mm256_cvtepi8_epi32(eqmask8);
                __m256i score    = _mm256_blendv_epi8(vmismatch, vmatch, eqmask32);

                __m256i diagv = _mm256_loadu_si256((const __m256i*)(prev + j - 1));
                __m256i upv   = _mm256_loadu_si256((const __m256i*)(prev + j));

                __m256i diagscore = _mm256_add_epi32(diagv, score);
                __m256i upscore   = _mm256_add_epi32(upv, vgap);
                __m256i best      = _mm256_max_epi32(diagscore, upscore);

                _mm256_storeu_si256((__m256i*)(cand + j), best);
            }
        }
#endif
        for (; j <= n; j++) {
            int diag = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int up   = prev[j] + GAP;
            cand[j]  = diag > up ? diag : up;
        }

        int left_prev = cur[0];
        for (int jj = 1; jj <= n; jj++) {
            int leftc = left_prev + GAP;
            int c = cand[jj];
            int val = c > leftc ? c : leftc;
            cur[jj] = val;
            left_prev = val;
        }

        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[n];
    free(prev);
    free(cur);
    free(cand);
    return result;
}
