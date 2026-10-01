/* Ink-worm wavefront Needleman-Wunsch score.
   match=+1, mismatch=-1, gap=-2, global, equal-length.
   Contract: int kernel(int n, const char *a, const char *b); */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__SSE2__) || defined(__AVX2__) || defined(__AVX512BW__)
#include <immintrin.h>
#endif

#define NW_PAD 64   /* slack so the worm's body may overhang the live range */

/* ---- exact row DP: used for tiny n (setup would dominate) and as a safety net ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    int sbuf[1026];
    int *base, *prev, *cur, *t;
    int i, j, r;
    if (n <= 0) return 0;
    if ((long)(n + 1) * 2 <= 1026) base = sbuf;
    else base = (int *)malloc((size_t)(n + 1) * 2 * sizeof(int));
    if (!base) return 0;
    prev = base; cur = base + (n + 1);
    for (j = 0; j <= n; ++j) prev[j] = -2 * j;
    for (i = 1; i <= n; ++i) {
        const char ai = a[i - 1];
        cur[0] = -2 * i;
        for (j = 1; j <= n; ++j) {
            int s = prev[j - 1] + ((ai == b[j - 1]) ? 1 : -1);
            int u = prev[j] - 2;
            int l = cur[j - 1] - 2;
            if (u > s) s = u;
            if (l > s) s = l;
            cur[j] = s;
        }
        t = prev; prev = cur; cur = t;
    }
    r = prev[n];
    if (base != sbuf) free(base);
    return r;
}

/* ---- 32-bit wavefront: only for n so large that 16-bit headroom (|H| <= 2n) fails ---- */
static int nw_diag32(int n, const char *a, const char *b)
{
    const int m = n + NW_PAD;
    int32_t *buf, *D2, *D1, *D0, *t;
    char *A, *Br;
    int d, i, res;
    void *mem = malloc((size_t)3 * m * sizeof(int32_t) + (size_t)2 * m + 64);
    if (!mem) return nw_rows(n, a, b);
    buf = (int32_t *)mem;
    memset(buf, 0, (size_t)3 * m * sizeof(int32_t));
    D2 = buf; D1 = buf + m; D0 = buf + 2 * m;
    A = (char *)(buf + 3 * m); Br = A + m;
    memcpy(A, a, (size_t)n); memset(A + n, 1, NW_PAD);
    for (i = 0; i < n; ++i) Br[i] = b[n - 1 - i];
    memset(Br + n, 2, NW_PAD);
    D0[0] = 0;
    t = D2; D2 = D1; D1 = D0; D0 = t;
    for (d = 1; d <= 2 * n; ++d) {
        int lo, hi; const int k0 = n - d;
        if (d <= n) { lo = 1; hi = d - 1; } else { lo = d - n; hi = n; }
        for (i = lo; i <= hi; ++i) {
            int s = D2[i - 1] + ((A[i - 1] == Br[i + k0]) ? 1 : -1);
            int u = D1[i - 1] - 2;
            int l = D1[i] - 2;
            if (u > s) s = u;
            if (l > s) s = l;
            D0[i] = s;
        }
        if (d <= n) { D0[0] = -2 * d; D0[d] = -2 * d; }
        t = D2; D2 = D1; D1 = D0; D0 = t;
    }
    res = (int)D1[n];
    free(mem);
    return res;
}

/* ---- the worm: 16-bit wavefront ---- */
static int nw_diag16(int n, const char *a, const char *b)
{
    const int m = n + NW_PAD;
    int16_t *buf, *D2, *D1, *D0, *t;
    char *A, *Br;
    int d, i, res;
    void *mem = malloc((size_t)3 * m * sizeof(int16_t) + (size_t)2 * m + 64);
    if (!mem) return nw_rows(n, a, b);
    buf = (int16_t *)mem;
    memset(buf, 0, (size_t)3 * m * sizeof(int16_t));
    D2 = buf; D1 = buf + m; D0 = buf + 2 * m;
    A = (char *)(buf + 3 * m); Br = A + m;
    /* both cords laid so the worm's body reads them in the same grain */
    memcpy(A, a, (size_t)n); memset(A + n, 1, NW_PAD);
    for (i = 0; i < n; ++i) Br[i] = b[n - 1 - i];
    memset(Br + n, 2, NW_PAD);

    D0[0] = 0;                                   /* near corner */
    t = D2; D2 = D1; D1 = D0; D0 = t;

    for (d = 1; d <= 2 * n; ++d) {
        int lo, hi; const int k0 = n - d;
        if (d <= n) { lo = 1; hi = d - 1; } else { lo = d - n; hi = n; }
        if (hi >= lo) {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            const __m512i vp1 = _mm512_set1_epi16(1);
            const __m512i vm1 = _mm512_set1_epi16(-1);
            const __m512i v2  = _mm512_set1_epi16(2);
            for (i = lo; i <= hi; i += 32) {
                __m256i ca = _mm256_loadu_si256((const __m256i *)(A + (i - 1)));
                __m256i cb = _mm256_loadu_si256((const __m256i *)(Br + (i + k0)));
                __mmask32 kq = _mm256_cmpeq_epi8_mask(ca, cb);
                __m512i sc = _mm512_mask_blend_epi16(kq, vm1, vp1);
                __m512i dg = _mm512_add_epi16(
                        _mm512_loadu_si512((const void *)(D2 + (i - 1))), sc);
                __m512i up = _mm512_loadu_si512((const void *)(D1 + (i - 1)));
                __m512i le = _mm512_loadu_si512((const void *)(D1 + i));
                __m512i gp = _mm512_sub_epi16(_mm512_max_epi16(up, le), v2);
                _mm512_storeu_si512((void *)(D0 + i), _mm512_max_epi16(dg, gp));
            }
#elif defined(__AVX2__)
            const __m256i vm1 = _mm256_set1_epi16(-1);
            const __m256i v2  = _mm256_set1_epi16(2);
            for (i = lo; i <= hi; i += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(A + (i - 1)));
                __m128i cb = _mm_loadu_si128((const __m128i *)(Br + (i + k0)));
                /* -1 where the symbols agree, 0 where they do not */
                __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                /* +1 on agreement, -1 otherwise */
                __m256i sc = _mm256_sub_epi16(vm1, _mm256_add_epi16(eq, eq));
                __m256i dg = _mm256_add_epi16(
                        _mm256_loadu_si256((const __m256i *)(D2 + (i - 1))), sc);
                __m256i up = _mm256_loadu_si256((const __m256i *)(D1 + (i - 1)));
                __m256i le = _mm256_loadu_si256((const __m256i *)(D1 + i));
                __m256i gp = _mm256_sub_epi16(_mm256_max_epi16(up, le), v2);
                _mm256_storeu_si256((__m256i *)(D0 + i), _mm256_max_epi16(dg, gp));
            }
#else
            for (i = lo; i <= hi; ++i) {   /* auto-vectorizable: no carried dep */
                int s = D2[i - 1] + ((A[i - 1] == Br[i + k0]) ? 1 : -1);
                int u = D1[i - 1] - 2;
                int l = D1[i] - 2;
                if (u > s) s = u;
                if (l > s) s = l;
                D0[i] = (int16_t)s;
            }
#endif
        }
        /* the two tray edges, written after the body so overhang cannot spoil them */
        if (d <= n) { D0[0] = (int16_t)(-2 * d); D0[d] = (int16_t)(-2 * d); }
        t = D2; D2 = D1; D1 = D0; D0 = t;
    }
    res = (int)D1[n];                            /* far corner */
    free(mem);
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < 24)     return nw_rows(n, a, b);     /* guard: setup would dominate */
    if (n <= 12000) return nw_diag16(n, a, b);   /* guard: |H| <= 2n < 32767 */
    return nw_diag32(n, a, b);                   /* guard: 16-bit headroom gone */
}
