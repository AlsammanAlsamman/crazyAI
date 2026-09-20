#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH 1
#define MISMATCH (-1)
#define GAP (-2)
#define NEG (-1000000000)

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int cap = ((n + 1 + 7) / 8) * 8 + 8;
    int32_t *prev2 = (int32_t *)malloc(sizeof(int32_t) * (size_t)cap);
    int32_t *prev1 = (int32_t *)malloc(sizeof(int32_t) * (size_t)cap);
    int32_t *cur   = (int32_t *)malloc(sizeof(int32_t) * (size_t)cap);
    char    *brev  = (char *)malloc((size_t)n);

    for (int t = 0; t < n; t++) brev[t] = b[n - 1 - t];
    for (int i = 0; i < cap; i++) { prev2[i] = NEG; prev1[i] = NEG; cur[i] = NEG; }

    /* k = 0 diagonal: only (0,0) */
    prev2[0] = 0;
    /* k = 1 diagonal: (0,1) and (1,0) */
    prev1[0] = GAP;
    prev1[1] = GAP;

    const char * restrict pa    = a;
    const char * restrict pbrev = brev;

    for (int k = 2; k <= 2 * n; k++) {
        int lo = (k - n > 0) ? (k - n) : 0;
        int hi = (k < n) ? k : n;
        int i = lo;

        if (i == 0) {
            /* row 0: j = k >= 1, only "left" applies */
            cur[0] = prev1[0] + GAP;
            i = 1;
        }

        int vec_hi = (hi < k - 1) ? hi : (k - 1); /* last i with j = k-i >= 1 */

#if defined(__AVX2__)
        const __m256i vGAP      = _mm256_set1_epi32(GAP);
        const __m256i vMATCH    = _mm256_set1_epi32(MATCH);
        const __m256i vMISMATCH = _mm256_set1_epi32(MISMATCH);
        for (; i + 8 <= vec_hi + 1; i += 8) {
            __m256i p2   = _mm256_loadu_si256((const __m256i *)(prev2 + i - 1));
            __m256i p1up = _mm256_loadu_si256((const __m256i *)(prev1 + i - 1));
            __m256i p1lf = _mm256_loadu_si256((const __m256i *)(prev1 + i));

            char abuf[8], bbuf[8];
            memcpy(abuf, pa + (i - 1), 8);
            memcpy(bbuf, pbrev + (n - k + i), 8);
            __m256i va = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)abuf));
            __m256i vb = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)bbuf));
            __m256i eq  = _mm256_cmpeq_epi32(va, vb);
            __m256i sub = _mm256_blendv_epi8(vMISMATCH, vMATCH, eq);

            __m256i diag = _mm256_add_epi32(p2, sub);
            __m256i up   = _mm256_add_epi32(p1up, vGAP);
            __m256i left = _mm256_add_epi32(p1lf, vGAP);

            __m256i best = _mm256_max_epi32(diag, up);
            best = _mm256_max_epi32(best, left);

            _mm256_storeu_si256((__m256i *)(cur + i), best);
        }
#endif
        for (; i <= hi; i++) {
            int j = k - i;
            int diagv = (i >= 1 && j >= 1) ? prev2[i - 1] + ((pa[i - 1] == b[j - 1]) ? MATCH : MISMATCH) : NEG;
            int upv   = (i >= 1) ? prev1[i - 1] + GAP : NEG;
            int leftv = (j >= 1) ? prev1[i] + GAP : NEG;
            int best = diagv;
            if (upv   > best) best = upv;
            if (leftv > best) best = leftv;
            cur[i] = best;
        }

        int32_t *tmp = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur = tmp;
    }

    int result = prev1[n];

    free(prev2); free(prev1); free(cur); free(brev);
    return result;
}
