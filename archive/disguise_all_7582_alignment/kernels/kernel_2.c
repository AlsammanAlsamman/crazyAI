#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__) || defined(__AVX512BW__)
#include <immintrin.h>
#endif

#define NW_I16_MAX 10000  /* int16 safe bound incl. scratch-lane growth (|v| <= 3n) */

/* Scalar anti-diagonal sweep, 32-bit: fallback for no-AVX2 builds and huge n. */
static int nw_diag_scalar(int n, const char *a, const char *b)
{
    size_t dl = (size_t)n + 2;
    int *buf = (int *)calloc(3 * dl, sizeof(int));
    if (!buf) return 0;
    int *d2 = buf, *d1 = buf + dl, *d0 = buf + 2 * dl;
    d2[0] = 0;                 /* diagonal 0: dp[0][0] */
    d1[0] = -2; d1[1] = -2;    /* diagonal 1: dp[0][1], dp[1][0] */
    for (int d = 2; d <= 2 * n; d++) {
        int ilo = (d - n > 1) ? d - n : 1;
        int ihi = (d - 1 < n) ? d - 1 : n;
        for (int i = ilo; i <= ihi; i++) {
            int v = d2[i - 1] + ((a[i - 1] == b[d - i - 1]) ? 1 : -1);
            int u = d1[i - 1] - 2, l = d1[i] - 2;
            if (u > v) v = u;
            if (l > v) v = l;
            d0[i] = v;
        }
        if (d <= n) { d0[0] = -2 * d; d0[d] = -2 * d; }
        int *t = d2; d2 = d1; d1 = d0; d0 = t;
    }
    int r = d1[n];
    free(buf);
    return r;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if !(defined(__AVX2__) || defined(__AVX512BW__))
    return nw_diag_scalar(n, a, b);
#else
    if (n > NW_I16_MAX) return nw_diag_scalar(n, a, b);

    const size_t PAD  = 128;
    const size_t blen = (size_t)n + 2 * PAD;              /* padded byte arrays  */
    const size_t dl   = (size_t)n + 2 + 128;              /* padded i16 diagonals */

    unsigned char *raw = (unsigned char *)malloc(2 * blen + 3 * dl * sizeof(int16_t) + 64);
    if (!raw) return nw_diag_scalar(n, a, b);

    unsigned char *ap  = raw;          /* ap[k]  = a[k]                     */
    unsigned char *brp = raw + blen;   /* brp[k] = b[n-1-k]  (reversed b)   */
    memset(raw, 0, 2 * blen);
    memcpy(ap, a, (size_t)n);
    for (int k = 0; k < n; k++) brp[k] = (unsigned char)b[n - 1 - k];

    int16_t *base16 = (int16_t *)(((uintptr_t)(raw + 2 * blen) + 63u) & ~(uintptr_t)63u);
    memset(base16, 0, 3 * dl * sizeof(int16_t));
    int16_t *d2 = base16, *d1 = base16 + dl, *d0 = base16 + 2 * dl;

    d2[0] = 0;                                  /* diagonal 0 */
    d1[0] = -2; d1[1] = -2;                     /* diagonal 1 */

#if defined(__AVX512BW__)
    const __m512i w1 = _mm512_set1_epi16(1);
    const __m512i w2 = _mm512_set1_epi16(2);    /* doubles as the gap penalty */
#else
    const __m256i v1 = _mm256_set1_epi16(1);
    const __m256i v2 = _mm256_set1_epi16(2);
#endif

    const int nn = n, dmax = 2 * n;
    for (int d = 2; d <= dmax; d++) {
        int ilo  = (d - nn > 1) ? d - nn : 1;
        int ihi  = (d - 1 < nn) ? d - 1 : nn;
        int bas  = nn - d;                      /* brp index = bas + i, always >= 0 */
        int i    = ilo;

#if defined(__AVX512BW__)
        for (; i <= ihi; i += 32) {
            __m256i ca = _mm256_loadu_si256((const __m256i *)(ap + (i - 1)));
            __m256i cb = _mm256_loadu_si256((const __m256i *)(brp + (bas + i)));
            __m512i m  = _mm512_cvtepi8_epi16(_mm256_cmpeq_epi8(ca, cb));
            __m512i s  = _mm512_sub_epi16(_mm512_and_si512(m, w2), w1);   /* +1 / -1 */
            __m512i dg = _mm512_add_epi16(_mm512_loadu_si512((const void *)(d2 + i - 1)), s);
            __m512i up = _mm512_loadu_si512((const void *)(d1 + i - 1));
            __m512i lf = _mm512_loadu_si512((const void *)(d1 + i));
            __m512i r  = _mm512_max_epi16(dg,
                             _mm512_sub_epi16(_mm512_max_epi16(up, lf), w2));
            _mm512_storeu_si512((void *)(d0 + i), r);
        }
#else
        for (; i <= ihi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(ap + (i - 1)));
            __m128i cb = _mm_loadu_si128((const __m128i *)(brp + (bas + i)));
            __m256i m  = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i s  = _mm256_sub_epi16(_mm256_and_si256(m, v2), v1);   /* +1 / -1 */
            __m256i dg = _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(d2 + i - 1)), s);
            __m256i up = _mm256_loadu_si256((const __m256i *)(d1 + i - 1));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(d1 + i));
            __m256i r  = _mm256_max_epi16(dg,
                             _mm256_sub_epi16(_mm256_max_epi16(up, lf), v2));
            _mm256_storeu_si256((__m256i *)(d0 + i), r);
        }
#endif
        /* boundary cells written last: the tail vector may have clobbered index d */
        if (d <= nn) { d0[0] = (int16_t)(-2 * d); d0[d] = (int16_t)(-2 * d); }

        int16_t *t = d2; d2 = d1; d1 = d0; d0 = t;
    }

    int res = (int)d1[nn];   /* after the final rotation d1 == diagonal 2n */
    free(raw);
    return res;
#endif
}
