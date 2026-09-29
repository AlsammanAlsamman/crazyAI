#include <stdlib.h>
#include <string.h>

#if defined(__AVX512BW__) && defined(__AVX512VL__)
#  include <immintrin.h>
#  define NW_W 32
#  define NW_SIMD16 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define NW_W 16
#  define NW_SIMD16 1
#endif

#define NW_PAD 128
#define NW_I16_MAX 16000

/* last-resort fallback: O(n) memory row DP (used only if allocation fails) */
static int nw_rowdp(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int i, j, r = 0;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (j = 0; j <= n; j++) prev[j] = -2 * j;
    for (i = 1; i <= n; i++) {
        cur[0] = -2 * i;
        for (j = 1; j <= n; j++) {
            int dg = prev[j - 1] + (a[i - 1] == b[j - 1] ? 1 : -1);
            int u = prev[j] - 2, l = cur[j - 1] - 2;
            int m = dg; if (u > m) m = u; if (l > m) m = l;
            cur[j] = m;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* anti-diagonal wavefront, int32 (large n / no AVX2); no overrun, plain C */
static int nw_i32(int n, const unsigned char *__restrict pa,
                  const unsigned char *__restrict prb)
{
    size_t sz = (size_t)n + 1 + NW_PAD;
    int *mem = (int *)calloc(3 * sz, sizeof(int));
    int *p2, *p1, *p0;
    int d, res = 0, dmax = 2 * n;
    if (!mem) return 0;
    p2 = mem; p1 = mem + sz; p0 = mem + 2 * sz;
    p2[0] = 0; p1[0] = -2; p1[1] = -2;
    for (d = 2; d <= dmax; d++) {
        int lo = d - n, hi = d - 1, off = n - d, i;
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        for (i = lo; i <= hi; i++) {
            int u = p1[i - 1], l = p1[i];
            int g = (u > l ? u : l) - 2;
            int dg = p2[i - 1] + (pa[i - 1] == prb[i + off] ? 1 : -1);
            p0[i] = g > dg ? g : dg;
        }
        if (d <= n) { p0[0] = -2 * d; p0[d] = -2 * d; }
        if (d == dmax) res = p0[n];
        { int *t = p2; p2 = p1; p1 = p0; p0 = t; }
    }
    free(mem);
    return res;
}

#ifdef NW_SIMD16
/* anti-diagonal wavefront, int16 lanes: the whole stripe judged at once */
static int nw_i16(int n, const unsigned char *__restrict pa,
                  const unsigned char *__restrict prb)
{
    size_t sz = (size_t)n + 1 + NW_PAD;
    short *mem = (short *)calloc(3 * sz, sizeof(short));
    short *p2, *p1, *p0;
    int d, res = 0, dmax = 2 * n;
    if (!mem) return nw_i32(n, pa, prb);
    p2 = mem; p1 = mem + sz; p0 = mem + 2 * sz;
    p2[0] = 0; p1[0] = -2; p1[1] = -2;
#if NW_W == 32
    {
    const __m512i vtwo = _mm512_set1_epi16(2);
    const __m512i vp1v = _mm512_set1_epi16(1);
    const __m512i vm1v = _mm512_set1_epi16(-1);
#else
    {
    const __m256i vtwo = _mm256_set1_epi16(2);
    const __m128i bp1  = _mm_set1_epi8(1);
    const __m128i bm1  = _mm_set1_epi8(-1);
#endif
    for (d = 2; d <= dmax; d++) {
        int lo = d - n, hi = d - 1, off = n - d, i;
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        for (i = lo; i <= hi; i += NW_W) {
#if NW_W == 32
            __m256i av = _mm256_loadu_si256((const __m256i *)(pa + (i - 1)));
            __m256i bv = _mm256_loadu_si256((const __m256i *)(prb + (i + off)));
            __mmask32 keq = _mm256_cmpeq_epi8_mask(av, bv);
            __m512i sv = _mm512_mask_blend_epi16(keq, vm1v, vp1v);
            __m512i x2 = _mm512_loadu_si512((const void *)(p2 + (i - 1)));
            __m512i xu = _mm512_loadu_si512((const void *)(p1 + (i - 1)));
            __m512i xl = _mm512_loadu_si512((const void *)(p1 + i));
            __m512i g  = _mm512_sub_epi16(_mm512_max_epi16(xu, xl), vtwo);
            __m512i dg = _mm512_add_epi16(x2, sv);
            _mm512_storeu_si512((void *)(p0 + i), _mm512_max_epi16(g, dg));
#else
            __m128i av = _mm_loadu_si128((const __m128i *)(pa + (i - 1)));
            __m128i bv = _mm_loadu_si128((const __m128i *)(prb + (i + off)));
            __m128i eq = _mm_cmpeq_epi8(av, bv);
            __m256i sv = _mm256_cvtepi8_epi16(_mm_blendv_epi8(bm1, bp1, eq));
            __m256i x2 = _mm256_loadu_si256((const __m256i *)(p2 + (i - 1)));
            __m256i xu = _mm256_loadu_si256((const __m256i *)(p1 + (i - 1)));
            __m256i xl = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i g  = _mm256_sub_epi16(_mm256_max_epi16(xu, xl), vtwo);
            __m256i dg = _mm256_add_epi16(x2, sv);
            _mm256_storeu_si256((__m256i *)(p0 + i), _mm256_max_epi16(g, dg));
#endif
        }
        /* boundaries written AFTER the loop: the tail overrun may touch index d */
        if (d <= n) { short v = (short)(-2 * d); p0[0] = v; p0[d] = v; }
        if (d == dmax) res = (int)p0[n];
        { short *t = p2; p2 = p1; p1 = p0; p0 = t; }
    }
    }
    free(mem);
    return res;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    size_t cs;
    unsigned char *pa, *prb;
    int k, r;
    if (n <= 0) return 0;
    cs = (size_t)n + NW_PAD;
    pa  = (unsigned char *)calloc(cs, 1);
    prb = (unsigned char *)calloc(cs, 1);
    if (!pa || !prb) { free(pa); free(prb); return nw_rowdp(n, a, b); }
    memcpy(pa, a, (size_t)n);
    for (k = 0; k < n; k++) prb[k] = (unsigned char)b[n - 1 - k];
#ifdef NW_SIMD16
    r = (n <= NW_I16_MAX) ? nw_i16(n, pa, prb) : nw_i32(n, pa, prb);
#else
    r = nw_i32(n, pa, prb);
#endif
    free(pa); free(prb);
    return r;
}
