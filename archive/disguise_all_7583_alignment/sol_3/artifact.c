#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define NWPB 64   /* element padding on both sides of every buffer */

/* ---------------- scalar banded anti-diagonal NW ---------------- */
static int nw_band_scalar(int n, const unsigned char *A, const unsigned char *Brev,
                          int *q0, int *q1, int *q2, int W)
{
    const int NEGI = -(1 << 28);
    int m = n + 1 + 2 * NWPB, t, u;
    for (t = 0; t < m; t++) { q0[t] = NEGI; q1[t] = NEGI; q2[t] = NEGI; }
    int *dm2 = q0 + NWPB, *dm1 = q1 + NWPB, *cur = q2 + NWPB;
    for (u = 0; u <= 2 * n; u++) {
        int tt = u - W;
        int ilo = (tt <= 0) ? 0 : ((tt + 1) >> 1);
        if (u - n > ilo) ilo = u - n;
        int ihi = (u + W) >> 1;
        if (ihi > n) ihi = n;
        if (ihi > u) ihi = u;
        int base = n - u, i;
        for (i = ilo; i <= ihi; i++) {
            int s = (A[i - 1] == Brev[base + i]) ? 1 : -1;
            int v = dm2[i - 1] + s;
            int p = dm1[i - 1], q = dm1[i];
            int w = ((p > q) ? p : q) - 2;
            cur[i] = (v > w) ? v : w;
        }
        if (ilo == 0) cur[0] = -2 * u;
        if (ihi == u) cur[u] = -2 * u;
        cur[ilo - 1] = NEGI;
        cur[ihi + 1] = NEGI;
        { int *tp = dm2; dm2 = dm1; dm1 = cur; cur = tp; }
    }
    return dm1[n];
}

#if defined(__AVX2__)
/* ---------------- AVX2, 16 lanes of int16 (n <= 8000) ---------------- */
static int nw_band_i16(int n, const unsigned char *A, const unsigned char *Brev,
                       short *q0, short *q1, short *q2, int W)
{
    const short NEGS = -30000;
    int m = n + 1 + 2 * NWPB, t, u;
    for (t = 0; t < m; t++) { q0[t] = NEGS; q1[t] = NEGS; q2[t] = NEGS; }
    short *dm2 = q0 + NWPB, *dm1 = q1 + NWPB, *cur = q2 + NWPB;
    const __m256i vTWO = _mm256_set1_epi16(2);
    const __m256i vM1  = _mm256_set1_epi16(-1);
    const __m256i vP1  = _mm256_set1_epi16(1);
    for (u = 0; u <= 2 * n; u++) {
        int tt = u - W;
        int ilo = (tt <= 0) ? 0 : ((tt + 1) >> 1);
        if (u - n > ilo) ilo = u - n;
        int ihi = (u + W) >> 1;
        if (ihi > n) ihi = n;
        if (ihi > u) ihi = u;
        int base = n - u, i;
        for (i = ilo; i <= ihi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(A + (i - 1)));
            __m128i cb = _mm_loadu_si128((const __m128i *)(Brev + (base + i)));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i sc = _mm256_blendv_epi8(vM1, vP1, eq);
            __m256i dg = _mm256_loadu_si256((const __m256i *)(dm2 + (i - 1)));
            __m256i up = _mm256_loadu_si256((const __m256i *)(dm1 + (i - 1)));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(dm1 + i));
            __m256i vv = _mm256_max_epi16(
                             _mm256_adds_epi16(dg, sc),
                             _mm256_subs_epi16(_mm256_max_epi16(up, lf), vTWO));
            _mm256_storeu_si256((__m256i *)(cur + i), vv);
        }
        if (ilo == 0) cur[0] = (short)(-2 * u);
        if (ihi == u) cur[u] = (short)(-2 * u);
        cur[ilo - 1] = NEGS;
        cur[ihi + 1] = NEGS;
        { short *tp = dm2; dm2 = dm1; dm1 = cur; cur = tp; }
    }
    return (int)dm1[n];
}

/* ---------------- AVX2, 8 lanes of int32 (large n) ---------------- */
static int nw_band_i32(int n, const unsigned char *A, const unsigned char *Brev,
                       int *q0, int *q1, int *q2, int W)
{
    const int NEGI = -(1 << 28);
    int m = n + 1 + 2 * NWPB, t, u;
    for (t = 0; t < m; t++) { q0[t] = NEGI; q1[t] = NEGI; q2[t] = NEGI; }
    int *dm2 = q0 + NWPB, *dm1 = q1 + NWPB, *cur = q2 + NWPB;
    const __m256i vTWO = _mm256_set1_epi32(2);
    const __m256i vM1  = _mm256_set1_epi32(-1);
    const __m256i vP1  = _mm256_set1_epi32(1);
    for (u = 0; u <= 2 * n; u++) {
        int tt = u - W;
        int ilo = (tt <= 0) ? 0 : ((tt + 1) >> 1);
        if (u - n > ilo) ilo = u - n;
        int ihi = (u + W) >> 1;
        if (ihi > n) ihi = n;
        if (ihi > u) ihi = u;
        int base = n - u, i;
        for (i = ilo; i <= ihi; i += 8) {
            __m128i ca = _mm_loadl_epi64((const __m128i *)(A + (i - 1)));
            __m128i cb = _mm_loadl_epi64((const __m128i *)(Brev + (base + i)));
            __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
            __m256i sc = _mm256_blendv_epi8(vM1, vP1, eq);
            __m256i dg = _mm256_loadu_si256((const __m256i *)(dm2 + (i - 1)));
            __m256i up = _mm256_loadu_si256((const __m256i *)(dm1 + (i - 1)));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(dm1 + i));
            __m256i vv = _mm256_max_epi32(
                             _mm256_add_epi32(dg, sc),
                             _mm256_sub_epi32(_mm256_max_epi32(up, lf), vTWO));
            _mm256_storeu_si256((__m256i *)(cur + i), vv);
        }
        if (ilo == 0) cur[0] = -2 * u;
        if (ihi == u) cur[u] = -2 * u;
        cur[ilo - 1] = NEGI;
        cur[ihi + 1] = NEGI;
        { int *tp = dm2; dm2 = dm1; dm1 = cur; cur = tp; }
    }
    return dm1[n];
}
#endif /* __AVX2__ */

static int nw_band(int n, const unsigned char *A, const unsigned char *Brev,
                   unsigned char *rows, size_t rowb, int W)
{
#if defined(__AVX2__)
    if (n <= 8000)
        return nw_band_i16(n, A, Brev, (short *)rows,
                           (short *)(rows + rowb), (short *)(rows + 2 * rowb), W);
    return nw_band_i32(n, A, Brev, (int *)rows,
                       (int *)(rows + rowb), (int *)(rows + 2 * rowb), W);
#else
    return nw_band_scalar(n, A, Brev, (int *)rows,
                          (int *)(rows + rowb), (int *)(rows + 2 * rowb), W);
#endif
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    size_t nb   = (size_t)n + 2 * NWPB;
    size_t ne   = (size_t)n + 1 + 2 * NWPB;
    size_t seqb = (nb + 63) & ~(size_t)63;
    size_t rowb = ((ne * sizeof(int)) + 63) & ~(size_t)63;

    unsigned char *mem = (unsigned char *)malloc(2 * seqb + 3 * rowb + 64);
    if (!mem) return 0;
    unsigned char *abuf = mem;
    unsigned char *bbuf = mem + seqb;
    unsigned char *rows = mem + 2 * seqb;

    memset(abuf, 0xF1, seqb);
    memset(bbuf, 0xF2, seqb);
    unsigned char *A    = abuf + NWPB;
    unsigned char *Brev = bbuf + NWPB;

    int i, sdiag = 0;
    for (i = 0; i < n; i++) A[i] = (unsigned char)a[i];
    for (i = 0; i < n; i++) Brev[i] = (unsigned char)b[n - 1 - i];
    for (i = 0; i < n; i++) sdiag += (a[i] == b[i]) ? 1 : -1;

    /* provably sufficient band half-width from a lower bound LB: W = floor((n-LB)/5) */
    int W = (n - sdiag) / 5;
    if (W > 32) W = 32;          /* cheap probe first: sharpens LB for O(n) cost */
    if (W < 1)  W = 1;
    if (W > n)  W = n;

    int S;
    for (;;) {
        S = nw_band(n, A, Brev, rows, rowb, W);
        int need = (n - S) / 5;
        if (need <= W || W >= n) break;   /* certificate: nothing outside band can win */
        W = need;
        if (W > n) W = n;
    }
    free(mem);
    return S;
}
