#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__AVX2__) || defined(__AVX512BW__)
#include <immintrin.h>
#endif

#define NW_MATCH      1
#define NW_MISMATCH  -1
#define NW_GAP       -2

/* ---- exact reference recurrence, two rolling rows (fallback path) ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int i, j, r;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (j = 0; j <= n; j++) prev[j] = NW_GAP * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = NW_GAP * i;
        for (j = 1; j <= n; j++) {
            int dgl  = prev[j - 1] + (ai == b[j - 1] ? NW_MATCH : NW_MISMATCH);
            int up   = prev[j] + NW_GAP;
            int left = cur[j - 1] + NW_GAP;
            int best = dgl > up ? dgl : up;
            if (left > best) best = left;
            cur[j] = best;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

#if defined(__AVX2__)
/* ---- anti-diagonal wavefront, int16 lanes ---- */
static int nw_wave(int n, const char *a, const char *b)
{
    const int PAD = 64;
    const size_t clen = (size_t)n + 2 * (size_t)PAD;   /* bytes per char buffer  */
    const size_t blen = (size_t)n + 2 * (size_t)PAD;   /* int16 per score buffer */
    unsigned char *cbuf;
    int16_t *ibuf, *p2, *p1, *p0;
    unsigned char *ap, *bp;
    int d, k, result;

    cbuf = (unsigned char *)malloc(2 * clen);
    ibuf = (int16_t *)calloc(3 * blen, sizeof(int16_t));
    if (!cbuf || !ibuf) { free(cbuf); free(ibuf); return nw_rows(n, a, b); }

    memset(cbuf, 0, 2 * clen);
    ap = cbuf + PAD;                 /* a forward                 */
    bp = cbuf + clen + PAD;          /* b reversed: bp[k]=b[n-1-k] */
    memcpy(ap, a, (size_t)n);
    for (k = 0; k < n; k++) bp[k] = (unsigned char)b[n - 1 - k];

    p2 = ibuf + PAD;                 /* diagonal d-2 */
    p1 = ibuf + blen + PAD;          /* diagonal d-1 */
    p0 = ibuf + 2 * blen + PAD;      /* diagonal d   */

    p2[0] = 0;                                   /* d = 0 */
    p1[0] = (int16_t)NW_GAP; p1[1] = (int16_t)NW_GAP;   /* d = 1 */

#if defined(__AVX512BW__) && defined(__AVX512F__)
    {
        const __m512i vgap = _mm512_set1_epi16(NW_GAP);
        const __m512i vone = _mm512_set1_epi16(1);
        const __m512i vtwo = _mm512_set1_epi16(2);
        for (d = 2; d <= 2 * n; d++) {
            int lo = d - n; int hi = (d < n) ? d : n; int bb = n - d; int i;
            if (lo < 0) lo = 0;
            for (i = lo; i <= hi; i += 32) {
                __m256i ca = _mm256_loadu_si256((const __m256i *)(ap + (i - 1)));
                __m256i cb = _mm256_loadu_si256((const __m256i *)(bp + (i + bb)));
                __m512i eq = _mm512_cvtepi8_epi16(_mm256_cmpeq_epi8(ca, cb));
                __m512i s  = _mm512_sub_epi16(_mm512_and_si512(eq, vtwo), vone);
                __m512i dg = _mm512_add_epi16(
                                 _mm512_loadu_si512((const void *)(p2 + (i - 1))), s);
                __m512i u  = _mm512_loadu_si512((const void *)(p1 + (i - 1)));
                __m512i l  = _mm512_loadu_si512((const void *)(p1 + i));
                __m512i g  = _mm512_add_epi16(_mm512_max_epi16(u, l), vgap);
                _mm512_storeu_si512((void *)(p0 + i), _mm512_max_epi16(dg, g));
            }
            if (d <= n) {
                int16_t e = (int16_t)(NW_GAP * d);
                p0[0] = e; p0[d] = e;
            }
            { int16_t *t = p2; p2 = p1; p1 = p0; p0 = t; }
        }
    }
#else
    {
        const __m256i vgap = _mm256_set1_epi16(NW_GAP);
        const __m256i vone = _mm256_set1_epi16(1);
        const __m256i vtwo = _mm256_set1_epi16(2);
        for (d = 2; d <= 2 * n; d++) {
            int lo = d - n; int hi = (d < n) ? d : n; int bb = n - d; int i;
            if (lo < 0) lo = 0;
            for (i = lo; i <= hi; i += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(ap + (i - 1)));
                __m128i cb = _mm_loadu_si128((const __m128i *)(bp + (i + bb)));
                __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                __m256i s  = _mm256_sub_epi16(_mm256_and_si256(eq, vtwo), vone);
                __m256i dg = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(p2 + (i - 1))), s);
                __m256i u  = _mm256_loadu_si256((const __m256i *)(p1 + (i - 1)));
                __m256i l  = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m256i g  = _mm256_add_epi16(_mm256_max_epi16(u, l), vgap);
                _mm256_storeu_si256((__m256i *)(p0 + i), _mm256_max_epi16(dg, g));
            }
            if (d <= n) {
                int16_t e = (int16_t)(NW_GAP * d);
                p0[0] = e; p0[d] = e;
            }
            { int16_t *t = p2; p2 = p1; p1 = p0; p0 = t; }
        }
    }
#endif

    result = (int)p1[n];             /* after the last rotation p1 holds diagonal 2n */
    free(cbuf); free(ibuf);
    return result;
}
#endif /* __AVX2__ */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if defined(__AVX2__)
    /* int16 is exact while |score| <= 2n stays inside the type */
    if (n >= 16 && n <= 16000) return nw_wave(n, a, b);
#endif
    return nw_rows(n, a, b);
}
