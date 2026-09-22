#include <stdlib.h>
#include <immintrin.h>

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *curr = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cand = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !curr || !cand) { free(prev); free(curr); free(cand); return 0; }

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    const __m256i match_v    = _mm256_set1_epi32(MATCH);
    const __m256i mismatch_v = _mm256_set1_epi32(MISMATCH);
    const __m256i gap_v      = _mm256_set1_epi32(GAP);

    for (int i = 1; i <= n; i++) {
        curr[0] = i * GAP;

        __m256i a_v = _mm256_set1_epi32((int)(unsigned char)a[i - 1]);

        int j = 1;
        for (; j + 8 <= n + 1; j += 8) {
            __m256i prev_diag = _mm256_loadu_si256((const __m256i *)(prev + j - 1));
            __m256i prev_up   = _mm256_loadu_si256((const __m256i *)(prev + j));

            __m128i b_bytes = _mm_loadl_epi64((const __m128i *)(b + j - 1));
            __m256i b_v     = _mm256_cvtepu8_epi32(b_bytes);

            __m256i eq_mask = _mm256_cmpeq_epi32(a_v, b_v);
            __m256i sc      = _mm256_blendv_epi8(mismatch_v, match_v, eq_mask);

            __m256i diag = _mm256_add_epi32(prev_diag, sc);
            __m256i up   = _mm256_add_epi32(prev_up, gap_v);
            __m256i best = _mm256_max_epi32(diag, up);

            _mm256_storeu_si256((__m256i *)(cand + j), best);
        }
        for (; j <= n; j++) {
            int diag = prev[j - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int up   = prev[j] + GAP;
            cand[j]  = (diag > up) ? diag : up;
        }

        int left = curr[0];
        for (int jj = 1; jj <= n; jj++) {
            int c  = cand[jj];
            int lg = left + GAP;
            int v  = (c > lg) ? c : lg;
            curr[jj] = v;
            left = v;
        }

        int *tmp = prev; prev = curr; curr = tmp;
    }

    int result = prev[n];
    free(prev); free(curr); free(cand);
    return result;
}
