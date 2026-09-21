#include <stdlib.h>
#include <immintrin.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int width = n + 1;
    int *prev = (int *)malloc((size_t)width * sizeof(int));
    int *cur  = (int *)malloc((size_t)width * sizeof(int));

    for (int j = 0; j < width; j++) prev[j] = j * GAP;

    const __m256i gapVec      = _mm256_set1_epi32(GAP);
    const __m256i matchVec    = _mm256_set1_epi32(MATCH);
    const __m256i mismatchVec = _mm256_set1_epi32(MISMATCH);

    for (int i = 1; i <= n; i++) {
        cur[0] = i * GAP;
        char ai = a[i - 1];
        __m128i aiVec = _mm_set1_epi8(ai);

        int j = 1;
        for (; j + 7 <= n; j += 8) {
            /* 8 bytes of b starting at j-1, upper 64 bits of xmm zeroed */
            __m128i bvec8   = _mm_loadl_epi64((const __m128i *)(b + j - 1));
            __m128i eqmask8 = _mm_cmpeq_epi8(bvec8, aiVec); /* 0xFF match / 0x00 mismatch per byte */
            __m256i maskVec = _mm256_cvtepi8_epi32(eqmask8); /* sign-extend low 8 bytes -> -1/0 per lane */

            __m256i scoreVec = _mm256_blendv_epi8(mismatchVec, matchVec, maskVec);

            __m256i diagVec = _mm256_loadu_si256((const __m256i *)(prev + j - 1));
            __m256i upVec   = _mm256_loadu_si256((const __m256i *)(prev + j));

            __m256i diagScore = _mm256_add_epi32(diagVec, scoreVec);
            __m256i upScore   = _mm256_add_epi32(upVec, gapVec);
            __m256i tVec      = _mm256_max_epi32(diagScore, upScore);

            _mm256_storeu_si256((__m256i *)(cur + j), tVec);
        }
        for (; j <= n; j++) {
            int diag = prev[j - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int up   = prev[j] + GAP;
            cur[j] = diag > up ? diag : up;
        }

        /* only remaining true dependency: left-to-right correction */
        for (int jj = 1; jj <= n; jj++) {
            int left = cur[jj - 1] + GAP;
            if (left > cur[jj]) cur[jj] = left;
        }

        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[n];
    free(prev);
    free(cur);
    return result;
}
