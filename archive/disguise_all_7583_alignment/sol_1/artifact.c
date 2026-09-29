#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <immintrin.h>
#endif

/* ---- 16-bit anti-diagonal wavefront: valid while 2n+2 < 32767 ---- */
static int nw_i16(int n, const char *a, const char *b)
{
    const int PAD = 64;
    size_t blen = (size_t)n + 1 + (size_t)PAD;
    int16_t *mem = (int16_t *)malloc(3 * blen * sizeof(int16_t));
    char *cbuf = (char *)malloc(2 * ((size_t)n + (size_t)PAD));
    if (!mem || !cbuf) { free(mem); free(cbuf); return 0; }

    int16_t *p2 = mem, *p1 = mem + blen, *cu = mem + 2 * blen;
    char *A = cbuf, *B = cbuf + (size_t)n + PAD;

    memcpy(A, a, (size_t)n);
    memset(A + n, 0x00, PAD);
    for (int i = 0; i < n; i++) B[i] = b[n - 1 - i];   /* reversed b */
    memset(B + n, 0x7F, PAD);

    p2[0] = 0;                 /* diagonal 0: D[0][0] */
    p1[0] = -2; p1[1] = -2;    /* diagonal 1: D[0][1], D[1][0] */

    const int dmax = 2 * n;
    for (int d = 2; d <= dmax; d++) {
        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        int off = n - d;
        int i = lo;

#if defined(__AVX512BW__) && defined(__AVX512F__)
        {
            const __m512i m1 = _mm512_set1_epi16(-1);
            const __m512i g2 = _mm512_set1_epi16(2);
            for (; i <= hi; i += 32) {
                __m256i ca = _mm256_loadu_si256((const __m256i *)(A + i - 1));
                __m256i cbv = _mm256_loadu_si256((const __m256i *)(B + off + i));
                __m256i eq = _mm256_cmpeq_epi8(ca, cbv);
                __m512i t = _mm512_cvtepi8_epi16(eq);              /* 0 or -1 */
                __m512i s = _mm512_sub_epi16(m1, _mm512_add_epi16(t, t));
                __m512i dg = _mm512_loadu_si512((const void *)(p2 + i - 1));
                __m512i uu = _mm512_loadu_si512((const void *)(p1 + i - 1));
                __m512i ll = _mm512_loadu_si512((const void *)(p1 + i));
                __m512i best = _mm512_max_epi16(
                        _mm512_add_epi16(dg, s),
                        _mm512_sub_epi16(_mm512_max_epi16(uu, ll), g2));
                _mm512_storeu_si512((void *)(cu + i), best);
            }
        }
#elif defined(__AVX2__)
        {
            const __m256i m1 = _mm256_set1_epi16(-1);
            const __m256i g2 = _mm256_set1_epi16(2);
            for (; i <= hi; i += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(A + i - 1));
                __m128i cbv = _mm_loadu_si128((const __m128i *)(B + off + i));
                __m128i eq = _mm_cmpeq_epi8(ca, cbv);
                __m256i t = _mm256_cvtepi8_epi16(eq);              /* 0 or -1 */
                __m256i s = _mm256_sub_epi16(m1, _mm256_add_epi16(t, t));
                __m256i dg = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i uu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i ll = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m256i best = _mm256_max_epi16(
                        _mm256_add_epi16(dg, s),
                        _mm256_sub_epi16(_mm256_max_epi16(uu, ll), g2));
                _mm256_storeu_si256((__m256i *)(cu + i), best);
            }
        }
#elif defined(__SSE4_1__)
        {
            const __m128i m1 = _mm_set1_epi16(-1);
            const __m128i g2 = _mm_set1_epi16(2);
            for (; i <= hi; i += 8) {
                __m128i ca = _mm_loadl_epi64((const __m128i *)(A + i - 1));
                __m128i cbv = _mm_loadl_epi64((const __m128i *)(B + off + i));
                __m128i eq = _mm_cmpeq_epi8(ca, cbv);
                __m128i t = _mm_cvtepi8_epi16(eq);
                __m128i s = _mm_sub_epi16(m1, _mm_add_epi16(t, t));
                __m128i dg = _mm_loadu_si128((const __m128i *)(p2 + i - 1));
                __m128i uu = _mm_loadu_si128((const __m128i *)(p1 + i - 1));
                __m128i ll = _mm_loadu_si128((const __m128i *)(p1 + i));
                __m128i best = _mm_max_epi16(
                        _mm_add_epi16(dg, s),
                        _mm_sub_epi16(_mm_max_epi16(uu, ll), g2));
                _mm_storeu_si128((__m128i *)(cu + i), best);
            }
        }
#endif
        for (; i <= hi; i++) {                         /* scalar tail */
            int sc = (A[i - 1] == B[off + i]) ? 1 : -1;
            int v = (int)p2[i - 1] + sc;
            int u = (int)p1[i - 1];
            int l = (int)p1[i];
            int m = (u > l ? u : l) - 2;
            cu[i] = (int16_t)(v > m ? v : m);
        }
        if (d <= n) {                                  /* grid boundaries */
            int16_t bv = (int16_t)(-2 * d);
            cu[0] = bv;
            cu[d] = bv;
        }
        { int16_t *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }
    }

    int res = (int)p1[n];
    free(mem); free(cbuf);
    return res;
}

/* ---- 32-bit anti-diagonal wavefront: large n ---- */
static int nw_i32(int n, const char *a, const char *b)
{
    const int PAD = 64;
    size_t blen = (size_t)n + 1 + (size_t)PAD;
    int32_t *mem = (int32_t *)malloc(3 * blen * sizeof(int32_t));
    char *cbuf = (char *)malloc(2 * ((size_t)n + (size_t)PAD));
    if (!mem || !cbuf) { free(mem); free(cbuf); return 0; }

    int32_t *p2 = mem, *p1 = mem + blen, *cu = mem + 2 * blen;
    char *A = cbuf, *B = cbuf + (size_t)n + PAD;

    memcpy(A, a, (size_t)n);
    memset(A + n, 0x00, PAD);
    for (int i = 0; i < n; i++) B[i] = b[n - 1 - i];
    memset(B + n, 0x7F, PAD);

    p2[0] = 0;
    p1[0] = -2; p1[1] = -2;

    const int dmax = 2 * n;
    for (int d = 2; d <= dmax; d++) {
        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        int off = n - d;
        int i = lo;

#if defined(__AVX2__)
        {
            const __m256i m1 = _mm256_set1_epi32(-1);
            const __m256i g2 = _mm256_set1_epi32(2);
            for (; i <= hi; i += 8) {
                __m128i ca = _mm_loadl_epi64((const __m128i *)(A + i - 1));
                __m128i cbv = _mm_loadl_epi64((const __m128i *)(B + off + i));
                __m128i eq = _mm_cmpeq_epi8(ca, cbv);
                __m256i t = _mm256_cvtepi8_epi32(eq);
                __m256i s = _mm256_sub_epi32(m1, _mm256_add_epi32(t, t));
                __m256i dg = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i uu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i ll = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m256i best = _mm256_max_epi32(
                        _mm256_add_epi32(dg, s),
                        _mm256_sub_epi32(_mm256_max_epi32(uu, ll), g2));
                _mm256_storeu_si256((__m256i *)(cu + i), best);
            }
        }
#endif
        for (; i <= hi; i++) {
            int sc = (A[i - 1] == B[off + i]) ? 1 : -1;
            int v = p2[i - 1] + sc;
            int u = p1[i - 1];
            int l = p1[i];
            int m = (u > l ? u : l) - 2;
            cu[i] = (v > m ? v : m);
        }
        if (d <= n) { cu[0] = -2 * d; cu[d] = -2 * d; }
        { int32_t *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }
    }

    int res = p1[n];
    free(mem); free(cbuf);
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n <= 16000) return nw_i16(n, a, b);   /* |D| <= 2n fits int16 */
    return nw_i32(n, a, b);
}
