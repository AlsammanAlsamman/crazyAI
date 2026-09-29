#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

#define NW_MATCH     1
#define NW_MISMATCH -1
#define NW_GAP       2   /* magnitude; penalty is -NW_GAP */

/* Exact fallback: same recurrence, two rows, int precision. */
static int nw_rows(int n, const char *a, const char *b)
{
    int *prev, *cur, r;
    prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = -NW_GAP * j;
    for (int i = 1; i <= n; i++) {
        const char ai = a[i - 1];
        cur[0] = -NW_GAP * i;
        for (int j = 1; j <= n; j++) {
            int dg = prev[j - 1] + ((ai == b[j - 1]) ? NW_MATCH : NW_MISMATCH);
            int up = prev[j]     - NW_GAP;
            int lf = cur[j - 1]  - NW_GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

#if defined(__AVX2__)
    /* int16 is exact while the whole table fits in [-2n, n]. */
    if (n <= 16000) {
        const int stride = n + 1 + 64;            /* slack for SIMD overrun */
        int16_t *base = (int16_t *)malloc((size_t)3 * (size_t)stride * sizeof(int16_t));
        char    *A    = (char *)malloc((size_t)n + 160);
        char    *BR   = (char *)malloc((size_t)n + 160);

        if (base && A && BR) {
            memset(base, 0,    (size_t)3 * (size_t)stride * sizeof(int16_t));
            memset(A,    0x00, (size_t)n + 160);
            memset(BR,   0x7F, (size_t)n + 160);  /* pads can never compare equal */
            memcpy(A, a, (size_t)n);
            for (int k = 0; k < n; k++) BR[k] = b[n - 1 - k];

            int16_t *p2 = base;                   /* diagonal d-2 */
            int16_t *p1 = base + stride;          /* diagonal d-1 */
            int16_t *cu = base + 2 * stride;      /* diagonal d   */

            p2[0] = 0;                            /* d = 0 */
            p1[0] = (int16_t)(-NW_GAP);           /* d = 1 */
            p1[1] = (int16_t)(-NW_GAP);

            for (int d = 2; d <= 2 * n; d++) {
                int ilo = d - n; if (ilo < 1) ilo = 1;
                int ihi = d - 1; if (ihi > n) ihi = n;
                const int boff = n - d;           /* br index = i + boff */

#if defined(__AVX512BW__)
                {
                    const __m512i vneg1 = _mm512_set1_epi16(-1);
                    const __m512i vgap  = _mm512_set1_epi16(NW_GAP);
                    for (int i = ilo; i <= ihi; i += 32) {
                        __m256i ca = _mm256_loadu_si256((const __m256i *)(A  + (i - 1)));
                        __m256i cb = _mm256_loadu_si256((const __m256i *)(BR + (i + boff)));
                        __m512i eq = _mm512_cvtepi8_epi16(_mm256_cmpeq_epi8(ca, cb));
                        __m512i pd = _mm512_loadu_si512((const void *)(p2 + (i - 1)));
                        __m512i pu = _mm512_loadu_si512((const void *)(p1 + (i - 1)));
                        __m512i pl = _mm512_loadu_si512((const void *)(p1 + i));
                        __m512i dg = _mm512_sub_epi16(_mm512_add_epi16(pd, vneg1),
                                                      _mm512_add_epi16(eq, eq));
                        __m512i ul = _mm512_sub_epi16(_mm512_max_epi16(pu, pl), vgap);
                        _mm512_storeu_si512((void *)(cu + i), _mm512_max_epi16(dg, ul));
                    }
                }
#else
                {
                    const __m256i vneg1 = _mm256_set1_epi16(-1);
                    const __m256i vgap  = _mm256_set1_epi16(NW_GAP);
                    for (int i = ilo; i <= ihi; i += 16) {
                        __m128i ca = _mm_loadu_si128((const __m128i *)(A  + (i - 1)));
                        __m128i cb = _mm_loadu_si128((const __m128i *)(BR + (i + boff)));
                        __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                        __m256i pd = _mm256_loadu_si256((const __m256i *)(p2 + (i - 1)));
                        __m256i pu = _mm256_loadu_si256((const __m256i *)(p1 + (i - 1)));
                        __m256i pl = _mm256_loadu_si256((const __m256i *)(p1 + i));
                        __m256i dg = _mm256_sub_epi16(_mm256_add_epi16(pd, vneg1),
                                                      _mm256_add_epi16(eq, eq));
                        __m256i ul = _mm256_sub_epi16(_mm256_max_epi16(pu, pl), vgap);
                        _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi16(dg, ul));
                    }
                }
#endif
                if (d <= n) {                     /* first row / first column cells */
                    int16_t bv = (int16_t)(-NW_GAP * d);
                    cu[0] = bv;
                    cu[d] = bv;
                }
                { int16_t *t = p2; p2 = p1; p1 = cu; cu = t; }
            }

            int res = (int)p1[n];                 /* bottom-right box */
            free(base); free(A); free(BR);
            return res;
        }
        free(base); free(A); free(BR);
    }
#endif
    return nw_rows(n, a, b);
}
