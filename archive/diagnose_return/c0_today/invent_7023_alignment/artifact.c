#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH (-1)
#define GAP      (-2)

/* ---- scalar banded row DP (int32): fallback for no-AVX2, tiny n, huge n ---- */
static int nw_band_rows(int n, const char *restrict a, const char *restrict b, int W)
{
    const int NEG = -(1 << 28);
    int i, j, res;
    int *buf, *prev, *cur, *t;
    if (W < 1) W = 1;
    if (W > n) W = n;
    buf = (int *)malloc((size_t)2 * (size_t)(n + 2) * sizeof(int));
    if (!buf) return 0;
    prev = buf;
    cur  = buf + (n + 2);
    {   int hi0 = (W < n) ? W : n;
        for (j = 0; j <= hi0; j++) prev[j] = -2 * j;
        prev[hi0 + 1] = NEG;                       /* hi0+1 <= n+1 */
    }
    for (i = 1; i <= n; i++) {
        int lo = i - W, hi = i + W;
        const char ai = a[i - 1];
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        if (lo >= 2) cur[lo - 1] = NEG;
        else         cur[0] = (i <= W) ? -2 * i : NEG;
        if (i + W <= n) prev[i + W] = NEG;
        for (j = lo; j <= hi; j++) {
            int dg = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int up = prev[j]     + GAP;
            int lf = cur[j - 1]  + GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        t = prev; prev = cur; cur = t;
    }
    res = prev[n];
    free(buf);
    return res;
}

#if defined(__AVX2__)
/* ---- the slant: banded anti-diagonal wavefront, 16 cells per doubled shadow ----
   diagonal d = i + j, indexed by i.  cur[i] = max( p2[i-1] + s(a[i-1],b[d-i-1]),
                                                    max(p1[i-1], p1[i]) + GAP )
   b stored reversed so the floor-rope reads forward along the slant.            */
static int nw_band_diag_avx2(int n, const char *a, const char *b, int W, int NEGi)
{
    const int P = 64;
    const size_t vlen = (size_t)n + 1 + 2 * (size_t)P;
    int16_t *mem  = (int16_t *)malloc(3 * vlen * sizeof(int16_t));
    unsigned char *cbuf = (unsigned char *)malloc(2 * (size_t)(n + 96));
    int16_t *p2, *p1, *cc, *tt;
    unsigned char *ap, *bp;
    const int16_t NEG = (int16_t)NEGi;
    __m256i vNEG, vGAP, vTWO, vONE;
    int d, i, res = 0;
    size_t k;

    if (!mem || !cbuf) { free(mem); free(cbuf); return nw_band_rows(n, a, b, W); }

    for (k = 0; k < 3 * vlen; k++) mem[k] = NEG;
    p2 = mem + P;  p1 = mem + vlen + P;  cc = mem + 2 * vlen + P;

    ap = cbuf + 32;
    bp = cbuf + (size_t)(n + 96) + 32;
    memset(cbuf,                     0x01, (size_t)(n + 96));   /* fillers differ, */
    memset(cbuf + (size_t)(n + 96),  0x02, (size_t)(n + 96));   /* so never match  */
    memcpy(ap, a, (size_t)n);
    for (i = 0; i < n; i++) bp[i] = (unsigned char)b[n - 1 - i];

    vNEG = _mm256_set1_epi16(NEG);
    vGAP = _mm256_set1_epi16(-2);
    vTWO = _mm256_set1_epi16(2);
    vONE = _mm256_set1_epi16(1);

    for (d = 0; d <= 2 * n; d++) {
        int ilo = (d - W + 1) >> 1;      /* |2i - d| <= W  */
        int ihi = (d + W) >> 1;
        int t0  = n - d;                 /* bp index = t0 + i */
        if (ilo < d - n) ilo = d - n;
        if (ilo < 0)     ilo = 0;
        if (ihi > d)     ihi = d;
        if (ihi > n)     ihi = n;
        for (i = ilo; i <= ihi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(ap + i - 1));
            __m128i cb = _mm_loadu_si128((const __m128i *)(bp + t0 + i));
            __m256i sb = _mm256_sub_epi16(
                             _mm256_and_si256(
                                 _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb)), vTWO),
                             vONE);                        /* +1 match, -1 mismatch */
            __m256i dg = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), sb);
            __m256i u  = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i l  = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i g  = _mm256_add_epi16(_mm256_max_epi16(u, l), vGAP);
            __m256i r  = _mm256_max_epi16(_mm256_max_epi16(dg, g), vNEG);
            _mm256_storeu_si256((__m256i *)(cc + i), r);
        }
        /* bare floor: never walked, only fenced off */
        _mm256_storeu_si256((__m256i *)(cc + ilo - 16), vNEG);
        _mm256_storeu_si256((__m256i *)(cc + ihi +  1), vNEG);
        _mm256_storeu_si256((__m256i *)(cc + ihi + 17), vNEG);
        if (ilo == 0) cc[0] = (int16_t)(-2 * d);   /* i = 0, j = d */
        if (ihi == d) cc[d] = (int16_t)(-2 * d);   /* i = d, j = 0 */
        if (d == 2 * n) res = (int)cc[n];
        tt = p2; p2 = p1; p1 = cc; cc = tt;
    }
    free(mem); free(cbuf);
    return res;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    int i, m0 = 0, LB, W;
    if (n <= 0) return 0;

    /* the doubled shadows on the main diagonal: regime detector + lower bound */
    for (i = 0; i < n; i++) m0 += (a[i] == b[i]);
    LB = 2 * m0 - n;

    /* certificate: any path at offset >= W+1 has score <= n-5(W+1) < LB <= answer */
    W = (n - LB) / 5;
    if (W < 1) W = 1;
    if (W > n) W = n;

#if defined(__AVX2__)
    /* guard the stated risks: int16 headroom, and SIMD setup vs tiny n */
    if (n >= 24 && (2L * n + 2L * W + 128L) < 32000L)
        return nw_band_diag_avx2(n, a, b, W, -(2 * n + 2 * W + 64));
#endif
    return nw_band_rows(n, a, b, W);
}
