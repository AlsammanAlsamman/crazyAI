#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------- scalar two-row fallback (tiny n / no AVX2 / alloc failure) ---------- */
static int nw_scalar(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = -2 * j;
    for (int i = 1; i <= n; i++) {
        cur[0] = -2 * i;
        const char ai = a[i - 1];
        for (int j = 1; j <= n; j++) {
            int diag = prev[j - 1] + (ai == b[j - 1] ? 1 : -1);
            int up   = prev[j] - 2;
            int left = cur[j - 1] - 2;
            int best = diag > up ? diag : up;
            if (left > best) best = left;
            cur[j] = best;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(prev); free(cur);
    return r;
}

#if defined(__AVX2__)

#if defined(__AVX512BW__) && defined(__AVX512VL__)
#define NW_AVX512 1
#define NW_W16 32
#define NW_W32 16
#else
#define NW_AVX512 0
#define NW_W16 16
#define NW_W32 8
#endif

/* A   : padded copy of a
   BR  : padded reversed copy of b, BR[k] = b[n-1-k]
   mem : 3*stride cells of scratch, stride >= n + 129            */
static int nw_wave16(int n, const char *A, const char *BR,
                     int16_t *mem, int stride)
{
    int16_t *p0 = mem, *p1 = mem + stride, *p2 = mem + 2 * stride;
#if NW_AVX512
    const __m512i vp1  = _mm512_set1_epi16(1);
    const __m512i vm1  = _mm512_set1_epi16(-1);
    const __m512i vgap = _mm512_set1_epi16(-2);
#else
    const __m256i vp1  = _mm256_set1_epi16(1);
    const __m256i vm1  = _mm256_set1_epi16(-1);
    const __m256i vgap = _mm256_set1_epi16(-2);
#endif
    const int lim = 2 * n;
    for (int d = 0; d <= lim; d++) {
        int ilo = d - n; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > n) ihi = n;
        int cnt = ihi - ilo + 1;

        const char *arow = A + (ilo - 1);
        const char *brow = BR + (n - d) + ilo;
        const int16_t *pp = p0 + (ilo - 1);
        const int16_t *pl = p1 + (ilo - 1);
        const int16_t *pu = p1 + ilo;
        int16_t *cu = p2 + ilo;

        for (int k = 0; k < cnt; k += NW_W16) {
#if NW_AVX512
            __mmask32 m = _mm256_cmpeq_epi8_mask(
                _mm256_loadu_si256((const __m256i *)(arow + k)),
                _mm256_loadu_si256((const __m256i *)(brow + k)));
            __m512i sc = _mm512_mask_blend_epi16(m, vm1, vp1);
            __m512i dv = _mm512_add_epi16(
                _mm512_loadu_si512((const void *)(pp + k)), sc);
            __m512i gv = _mm512_add_epi16(
                _mm512_max_epi16(_mm512_loadu_si512((const void *)(pl + k)),
                                 _mm512_loadu_si512((const void *)(pu + k))),
                vgap);
            _mm512_storeu_si512((void *)(cu + k), _mm512_max_epi16(dv, gv));
#else
            __m128i ma = _mm_cmpeq_epi8(
                _mm_loadu_si128((const __m128i *)(arow + k)),
                _mm_loadu_si128((const __m128i *)(brow + k)));
            __m256i m16 = _mm256_cvtepi8_epi16(ma);
            __m256i sc  = _mm256_blendv_epi8(vm1, vp1, m16);
            __m256i dv  = _mm256_add_epi16(
                _mm256_loadu_si256((const __m256i *)(pp + k)), sc);
            __m256i gv  = _mm256_add_epi16(
                _mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(pl + k)),
                                 _mm256_loadu_si256((const __m256i *)(pu + k))),
                vgap);
            _mm256_storeu_si256((__m256i *)(cu + k), _mm256_max_epi16(dv, gv));
#endif
        }
        if (d <= n) {                      /* boundary cells, written last  */
            p2[0] = (int16_t)(-2 * d);     /* dp[0][d]                      */
            p2[d] = (int16_t)(-2 * d);     /* dp[d][0]  (overwrites spill)  */
        }
        int16_t *t = p0; p0 = p1; p1 = p2; p2 = t;
    }
    return (int)p1[n];
}

static int nw_wave32(int n, const char *A, const char *BR,
                     int32_t *mem, int stride)
{
    int32_t *p0 = mem, *p1 = mem + stride, *p2 = mem + 2 * stride;
#if NW_AVX512
    const __m512i vp1  = _mm512_set1_epi32(1);
    const __m512i vm1  = _mm512_set1_epi32(-1);
    const __m512i vgap = _mm512_set1_epi32(-2);
#else
    const __m256i vp1  = _mm256_set1_epi32(1);
    const __m256i vm1  = _mm256_set1_epi32(-1);
    const __m256i vgap = _mm256_set1_epi32(-2);
#endif
    const int lim = 2 * n;
    for (int d = 0; d <= lim; d++) {
        int ilo = d - n; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > n) ihi = n;
        int cnt = ihi - ilo + 1;

        const char *arow = A + (ilo - 1);
        const char *brow = BR + (n - d) + ilo;
        const int32_t *pp = p0 + (ilo - 1);
        const int32_t *pl = p1 + (ilo - 1);
        const int32_t *pu = p1 + ilo;
        int32_t *cu = p2 + ilo;

        for (int k = 0; k < cnt; k += NW_W32) {
#if NW_AVX512
            __mmask16 m = _mm_cmpeq_epi8_mask(
                _mm_loadu_si128((const __m128i *)(arow + k)),
                _mm_loadu_si128((const __m128i *)(brow + k)));
            __m512i sc = _mm512_mask_blend_epi32(m, vm1, vp1);
            __m512i dv = _mm512_add_epi32(
                _mm512_loadu_si512((const void *)(pp + k)), sc);
            __m512i gv = _mm512_add_epi32(
                _mm512_max_epi32(_mm512_loadu_si512((const void *)(pl + k)),
                                 _mm512_loadu_si512((const void *)(pu + k))),
                vgap);
            _mm512_storeu_si512((void *)(cu + k), _mm512_max_epi32(dv, gv));
#else
            __m128i ma = _mm_cmpeq_epi8(
                _mm_loadl_epi64((const __m128i *)(arow + k)),
                _mm_loadl_epi64((const __m128i *)(brow + k)));
            __m256i m32 = _mm256_cvtepi8_epi32(ma);
            __m256i sc  = _mm256_blendv_epi8(vm1, vp1, m32);
            __m256i dv  = _mm256_add_epi32(
                _mm256_loadu_si256((const __m256i *)(pp + k)), sc);
            __m256i gv  = _mm256_add_epi32(
                _mm256_max_epi32(_mm256_loadu_si256((const __m256i *)(pl + k)),
                                 _mm256_loadu_si256((const __m256i *)(pu + k))),
                vgap);
            _mm256_storeu_si256((__m256i *)(cu + k), _mm256_max_epi32(dv, gv));
#endif
        }
        if (d <= n) {
            p2[0] = -2 * d;
            p2[d] = -2 * d;
        }
        int32_t *t = p0; p0 = p1; p1 = p2; p2 = t;
    }
    return (int)p1[n];
}
#endif /* __AVX2__ */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if defined(__AVX2__)
    if (n >= 64) {
        const int PAD = 128;
        int stride = (n + 1 + 128 + 63) & ~63;
        int use16 = (n <= 16000);
        char *sa = (char *)malloc((size_t)n + PAD);
        char *sb = (char *)malloc((size_t)n + PAD);
        void *mem = malloc((size_t)stride * 3 *
                           (use16 ? sizeof(int16_t) : sizeof(int32_t)));
        if (sa && sb && mem) {
            memcpy(sa, a, (size_t)n);
            memset(sa + n, 'N', PAD);
            for (int k = 0; k < n; k++) sb[k] = b[n - 1 - k];
            memset(sb + n, 'M', PAD);
            int r = use16 ? nw_wave16(n, sa, sb, (int16_t *)mem, stride)
                          : nw_wave32(n, sa, sb, (int32_t *)mem, stride);
            free(sa); free(sb); free(mem);
            return r;
        }
        free(sa); free(sb); free(mem);
    }
#endif
    return nw_scalar(n, a, b);
}
