#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX512BW__) || defined(__AVX2__)
#include <immintrin.h>
#endif

/* Needleman-Wunsch score (match +1, mismatch -1, gap -2) computed on
   anti-diagonal wavefronts with a rolling 3-strip window.
   Cells on one anti-diagonal are mutually independent -> SIMD. */
int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    enum { PADB = 64 };
    size_t slen = (size_t)n + 2u * (size_t)PADB;
    unsigned char *sbuf = (unsigned char *)malloc(2u * slen);
    if (!sbuf) return 0;
    unsigned char *A = sbuf + PADB;            /* A[k] = a[k]        */
    unsigned char *B = sbuf + slen + PADB;     /* B[k] = b[n-1-k]    */
    memset(sbuf, 0xF0, PADB);
    memcpy(A, a, (size_t)n);
    memset(A + n, 0xF1, PADB);
    memset(sbuf + slen, 0xF2, PADB);
    {
        const unsigned char *ub = (const unsigned char *)b;
        for (int k = 0; k < n; ++k) B[k] = ub[n - 1 - k];
    }
    memset(B + n, 0xF3, PADB);

    const size_t m = (size_t)n + 2u + 128u;    /* strip length + overrun pad */
    const int dmax = 2 * n;
    int result = 0;

    if (n <= 16000) {
        /* |dp| <= 2n <= 32000 : int16 lanes are safe */
        int16_t *buf = (int16_t *)calloc(3u * m, sizeof(int16_t));
        if (!buf) { free(sbuf); return 0; }
        int16_t *p2 = buf, *p1 = buf + m, *p0 = buf + 2 * m;

        for (int d = 0; d <= dmax; ++d) {
            int ilo, ihi;
            if (d <= n) { ilo = 1;     ihi = d - 1; }
            else        { ilo = d - n; ihi = n;     }
            const int off = n - d;                 /* b-index = i + off */
            int i = ilo;

#if defined(__AVX512BW__)
            {
                const __m512i vm1 = _mm512_set1_epi16(-1);
                const __m512i vg  = _mm512_set1_epi16(2);
                for (; i <= ihi; i += 32) {
                    __m256i va = _mm256_loadu_si256((const __m256i *)(A + i - 1));
                    __m256i vb = _mm256_loadu_si256((const __m256i *)(B + i + off));
                    __m512i eq = _mm512_cvtepi8_epi16(_mm256_cmpeq_epi8(va, vb));
                    __m512i sc = _mm512_sub_epi16(vm1, _mm512_add_epi16(eq, eq));
                    __m512i dg = _mm512_add_epi16(
                                   _mm512_loadu_si512((const void *)(p2 + i - 1)), sc);
                    __m512i up = _mm512_sub_epi16(
                                   _mm512_loadu_si512((const void *)(p1 + i - 1)), vg);
                    __m512i lf = _mm512_sub_epi16(
                                   _mm512_loadu_si512((const void *)(p1 + i)), vg);
                    _mm512_storeu_si512((void *)(p0 + i),
                        _mm512_max_epi16(_mm512_max_epi16(dg, up), lf));
                }
            }
#elif defined(__AVX2__)
            {
                const __m256i vm1 = _mm256_set1_epi16(-1);
                const __m256i vg  = _mm256_set1_epi16(2);
                for (; i <= ihi; i += 16) {
                    __m128i va = _mm_loadu_si128((const __m128i *)(A + i - 1));
                    __m128i vb = _mm_loadu_si128((const __m128i *)(B + i + off));
                    __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(va, vb));
                    __m256i sc = _mm256_sub_epi16(vm1, _mm256_add_epi16(eq, eq));
                    __m256i dg = _mm256_add_epi16(
                                   _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), sc);
                    __m256i up = _mm256_sub_epi16(
                                   _mm256_loadu_si256((const __m256i *)(p1 + i - 1)), vg);
                    __m256i lf = _mm256_sub_epi16(
                                   _mm256_loadu_si256((const __m256i *)(p1 + i)), vg);
                    _mm256_storeu_si256((__m256i *)(p0 + i),
                        _mm256_max_epi16(_mm256_max_epi16(dg, up), lf));
                }
            }
#endif
            {   /* scalar remainder (does all the work when no SIMD is available) */
                const int16_t *restrict q2 = p2;
                const int16_t *restrict q1 = p1;
                int16_t       *restrict q0 = p0;
                const unsigned char *restrict pa = A;
                const unsigned char *restrict pb = B + off;
                for (; i <= ihi; ++i) {
                    int v = q2[i - 1] + (pa[i - 1] == pb[i] ? 1 : -1);
                    int t = q1[i - 1] - 2; if (t > v) v = t;
                    t     = q1[i]     - 2; if (t > v) v = t;
                    q0[i] = (int16_t)v;
                }
            }
            if (d <= n) {                      /* boundary cells, written last */
                int16_t bv = (int16_t)(-2 * d);
                p0[0] = bv; p0[d] = bv;
            }
            { int16_t *t = p2; p2 = p1; p1 = p0; p0 = t; }
        }
        result = (int)p1[n];                   /* diagonal 2n holds only dp[n][n] */
        free(buf);
    } else {
        int32_t *buf = (int32_t *)calloc(3u * m, sizeof(int32_t));
        if (!buf) { free(sbuf); return 0; }
        int32_t *p2 = buf, *p1 = buf + m, *p0 = buf + 2 * m;

        for (int d = 0; d <= dmax; ++d) {
            int ilo, ihi;
            if (d <= n) { ilo = 1;     ihi = d - 1; }
            else        { ilo = d - n; ihi = n;     }
            const int off = n - d;
            int i = ilo;
#if defined(__AVX2__)
            {
                const __m256i vm1 = _mm256_set1_epi32(-1);
                const __m256i vg  = _mm256_set1_epi32(2);
                for (; i <= ihi; i += 8) {
                    __m128i va = _mm_loadu_si128((const __m128i *)(A + i - 1));
                    __m128i vb = _mm_loadu_si128((const __m128i *)(B + i + off));
                    __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(va, vb));
                    __m256i sc = _mm256_sub_epi32(vm1, _mm256_add_epi32(eq, eq));
                    __m256i dg = _mm256_add_epi32(
                                   _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), sc);
                    __m256i up = _mm256_sub_epi32(
                                   _mm256_loadu_si256((const __m256i *)(p1 + i - 1)), vg);
                    __m256i lf = _mm256_sub_epi32(
                                   _mm256_loadu_si256((const __m256i *)(p1 + i)), vg);
                    _mm256_storeu_si256((__m256i *)(p0 + i),
                        _mm256_max_epi32(_mm256_max_epi32(dg, up), lf));
                }
            }
#endif
            {
                const int32_t *restrict q2 = p2;
                const int32_t *restrict q1 = p1;
                int32_t       *restrict q0 = p0;
                const unsigned char *restrict pa = A;
                const unsigned char *restrict pb = B + off;
                for (; i <= ihi; ++i) {
                    int v = q2[i - 1] + (pa[i - 1] == pb[i] ? 1 : -1);
                    int t = q1[i - 1] - 2; if (t > v) v = t;
                    t     = q1[i]     - 2; if (t > v) v = t;
                    q0[i] = v;
                }
            }
            if (d <= n) { int32_t bv = -2 * d; p0[0] = bv; p0[d] = bv; }
            { int32_t *t = p2; p2 = p1; p1 = p0; p0 = t; }
        }
        result = (int)p1[n];
        free(buf);
    }

    free(sbuf);
    return result;
}
