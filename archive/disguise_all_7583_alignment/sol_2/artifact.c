#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__) || defined(__AVX512BW__)
#include <immintrin.h>
#endif

#define NW_PAD 64

/* ---------- 32-bit lane banded anti-diagonal NW (always available) ---------- */
static int nw_band_i32(int n, const char *A, const char *BR, int d)
{
    const int W = n + 160;
    int *buf = (int *)malloc((size_t)3 * (size_t)W * sizeof(int));
    if (!buf) return 0;
    const int NEG = -(1 << 28);
    { size_t tot = (size_t)3 * (size_t)W, t;
      for (t = 0; t < tot; t++) buf[t] = NEG; }
    int *p2 = buf + 16;          /* diagonal k-2 */
    int *p1 = buf + W + 16;      /* diagonal k-1 */
    int *cu = buf + 2 * W + 16;  /* diagonal k   */
#if defined(__AVX2__)
    const __m256i vone = _mm256_set1_epi32(1);
    const __m256i vtwo = _mm256_set1_epi32(2);
#endif
    const int K = 2 * n;
    for (int k = 0; k <= K; k++) {
        int t  = k - d;
        int lo = (t <= 0) ? 0 : ((t + 1) >> 1);
        if (k - n > lo) lo = k - n;
        int hi = (k + d) >> 1;
        if (hi > k) hi = k;
        if (hi > n) hi = n;
        const int off = n - k;
#if defined(__AVX2__)
        for (int i = lo; i <= hi; i += 8) {
            __m128i ca = _mm_loadl_epi64((const __m128i *)(A + i - 1));
            __m128i cb = _mm_loadl_epi64((const __m128i *)(BR + off + i));
            __m256i m  = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
            __m256i s  = _mm256_sub_epi32(_mm256_and_si256(m, vtwo), vone);
            __m256i vd = _mm256_add_epi32(
                             _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), s);
            __m256i vu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i vl = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i vh = _mm256_sub_epi32(_mm256_max_epi32(vu, vl), vtwo);
            _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi32(vd, vh));
        }
#else
        for (int i = lo; i <= hi; i++) {
            int s = (A[i - 1] == BR[off + i]) ? 1 : -1;
            int v = p2[i - 1] + s;
            int u = p1[i - 1], l = p1[i];
            int h = ((u > l) ? u : l) - 2;
            if (h > v) v = h;
            cu[i] = v;
        }
#endif
        if (k <= d) { cu[0] = -2 * k; cu[k] = -2 * k; }  /* first row / first col */
        cu[lo - 1] = NEG;
        cu[hi + 1] = NEG;
        { int *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }
    }
    int res = p1[n];
    free(buf);
    return res;
}

/* ---------- 16-bit lane banded anti-diagonal NW (AVX2 / AVX-512BW) ---------- */
#if defined(__AVX2__)
static int nw_band_i16(int n, const char *A, const char *BR, int d)
{
    const int W = n + 160;
    short *buf = (short *)malloc((size_t)3 * (size_t)W * sizeof(short));
    if (!buf) return nw_band_i32(n, A, BR, d);
    const short NEG = -32000;
    { size_t tot = (size_t)3 * (size_t)W, t;
      for (t = 0; t < tot; t++) buf[t] = NEG; }
    short *p2 = buf + 16;
    short *p1 = buf + W + 16;
    short *cu = buf + 2 * W + 16;
#if defined(__AVX512BW__)
    const __m512i vone = _mm512_set1_epi16(1);
    const __m512i vtwo = _mm512_set1_epi16(2);
#else
    const __m256i vone = _mm256_set1_epi16(1);
    const __m256i vtwo = _mm256_set1_epi16(2);
#endif
    const int K = 2 * n;
    for (int k = 0; k <= K; k++) {
        int t  = k - d;
        int lo = (t <= 0) ? 0 : ((t + 1) >> 1);
        if (k - n > lo) lo = k - n;
        int hi = (k + d) >> 1;
        if (hi > k) hi = k;
        if (hi > n) hi = n;
        const int off = n - k;
#if defined(__AVX512BW__)
        for (int i = lo; i <= hi; i += 32) {
            __m256i ca = _mm256_loadu_si256((const __m256i *)(A + i - 1));
            __m256i cb = _mm256_loadu_si256((const __m256i *)(BR + off + i));
            __m512i m  = _mm512_cvtepi8_epi16(_mm256_cmpeq_epi8(ca, cb));
            __m512i s  = _mm512_sub_epi16(_mm512_and_si512(m, vtwo), vone);
            __m512i vd = _mm512_adds_epi16(
                             _mm512_loadu_si512((const void *)(p2 + i - 1)), s);
            __m512i vu = _mm512_loadu_si512((const void *)(p1 + i - 1));
            __m512i vl = _mm512_loadu_si512((const void *)(p1 + i));
            __m512i vh = _mm512_subs_epi16(_mm512_max_epi16(vu, vl), vtwo);
            _mm512_storeu_si512((void *)(cu + i), _mm512_max_epi16(vd, vh));
        }
#else
        for (int i = lo; i <= hi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(A + i - 1));
            __m128i cb = _mm_loadu_si128((const __m128i *)(BR + off + i));
            __m256i m  = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i s  = _mm256_sub_epi16(_mm256_and_si256(m, vtwo), vone);
            __m256i vd = _mm256_adds_epi16(
                             _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), s);
            __m256i vu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i vl = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i vh = _mm256_subs_epi16(_mm256_max_epi16(vu, vl), vtwo);
            _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi16(vd, vh));
        }
#endif
        if (k <= d) { cu[0] = (short)(-2 * k); cu[k] = (short)(-2 * k); }
        cu[lo - 1] = NEG;
        cu[hi + 1] = NEG;
        { short *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }
    }
    int res = (int)p1[n];
    free(buf);
    return res;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    /* Lower bound on the optimal score: the pure-diagonal alignment. */
    int D = 0;
    for (int i = 0; i < n; i++) D += (a[i] == b[i]) ? 1 : -1;

    /* Any path of deviation v costs >= 2v gaps => score <= n - 5v.
       d = floor((n-D)/5) => every path leaving the band scores < D <= S*,
       so the banded optimum equals the full Needleman-Wunsch optimum. */
    int d = (n - D) / 5;
    if (d < 16) d = 16;
    if (d > n)  d = n;

    size_t rg = (size_t)n + 2 * NW_PAD;
    char *sb = (char *)malloc(2 * rg);
    if (!sb) return 0;
    memset(sb, 'N', 2 * rg);
    char *A  = sb + NW_PAD;
    char *BR = sb + rg + NW_PAD;
    memcpy(A, a, (size_t)n);
    for (int i = 0; i < n; i++) BR[i] = b[n - 1 - i];   /* reversed b */

    int res;
#if defined(__AVX2__)
    if (n + 2 * d < 31000) res = nw_band_i16(n, A, BR, d);
    else                   res = nw_band_i32(n, A, BR, d);
#else
    res = nw_band_i32(n, A, BR, d);
#endif
    free(sb);
    return res;
}
