/* The fold: anti-diagonal (i+j) wavefront Needleman-Wunsch.
 * Exact match to the reference DP (match +1, mismatch -1, gap -2, global).
 *   - creases: all cells with i+j = d are independent -> SIMD lanes.
 *   - two creases retained (O(n) memory, no DP table).
 *   - lit on demand: reach W = floor(2(n-m)/5) from the achievable score
 *     L = 2m-n; any path deviating D off the spine pays >= 2D gaps (the
 *     boughs are equal length), so its score <= n - 5D, so D <= (n-L)/5.
 *   - W clamped to n makes the clip non-binding => same code is the full fold.
 *   - fallbacks: no AVX2 / n < 96 / malloc failure -> plain row DP.
 */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define NW_MATCH     1
#define NW_MISMATCH (-1)
#define NW_GAP      (-2)

/* ---- the flat walker: reference-identical row DP, O(n) memory ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    int *base = (int *)malloc((size_t)2 * (size_t)(n + 1) * sizeof(int));
    if (!base) return 0;
    int *prev = base, *cur = base + (n + 1);
    for (int j = 0; j <= n; j++) prev[j] = j * NW_GAP;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * NW_GAP;
        for (int j = 1; j <= n; j++) {
            int best = prev[j - 1] + (ai == b[j - 1] ? NW_MATCH : NW_MISMATCH);
            int up = prev[j] + NW_GAP;
            int lf = cur[j - 1] + NW_GAP;
            if (up > best) best = up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(base);
    return r;
}

#if defined(__AVX2__)
/* ---- the fold, 16 sparks per crease (n <= 4000: |score| fits int16) ---- */
static int nw_fold16(int n, const unsigned char *restrict A,
                     const unsigned char *restrict R,
                     int W, short *buf, int stride)
{
    const short NEG = -20000;            /* < -(4n+1): can never win a max */
    short *p2 = buf, *p1 = buf + stride, *cu = buf + 2 * stride;
    for (int t = 0; t < 3 * stride; t++) buf[t] = NEG;
    p1[0] = 0;                           /* crease 0 holds dp[0][0] */
    const __m256i vgap = _mm256_set1_epi16((short)NW_GAP);
    const __m256i v2   = _mm256_set1_epi16(2);
    const __m256i v1   = _mm256_set1_epi16(1);
    const int dmax = 2 * n;
    for (int d = 1; d <= dmax; d++) {
        int ilo = d - n; if (ilo < 0) ilo = 0;
        int c = (d - W + 1) >> 1;   if (c > ilo) ilo = c;   /* ceil((d-W)/2) */
        int ihi = (d < n) ? d : n;
        int f = (d + W) >> 1;       if (f < ihi) ihi = f;   /* floor((d+W)/2) */
        int lo = (ilo < 1) ? 1 : ilo;
        int hi = (ihi < d - 1) ? ihi : d - 1;
        if (hi >= lo) {
            const unsigned char *pa = A + (lo - 1);
            const unsigned char *pb = R + (n - d + lo);
            const short *q2  = p2 + (lo - 1);   /* crease d-2, diag  */
            const short *q1u = p1 + (lo - 1);   /* crease d-1, up    */
            const short *q1l = p1 + lo;         /* crease d-1, left  */
            short *out = cu + lo;
            int len = hi - lo + 1;
            for (int k = 0; k < len; k += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(pa + k));
                __m128i cb = _mm_loadu_si128((const __m128i *)(pb + k));
                __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                __m256i sc = _mm256_sub_epi16(_mm256_and_si256(eq, v2), v1);
                __m256i vd = _mm256_add_epi16(
                    _mm256_loadu_si256((const __m256i *)(q2 + k)), sc);
                __m256i vu = _mm256_loadu_si256((const __m256i *)(q1u + k));
                __m256i vl = _mm256_loadu_si256((const __m256i *)(q1l + k));
                __m256i vg = _mm256_add_epi16(_mm256_max_epi16(vu, vl), vgap);
                _mm256_storeu_si256((__m256i *)(out + k),
                                    _mm256_max_epi16(vd, vg));
            }
        }
        /* rim of the crease, stamped after the sparks (order matters) */
        if (ilo == 0) cu[0] = (short)(NW_GAP * d);   /* dp[0][d] */
        if (ihi == d) cu[d] = (short)(NW_GAP * d);   /* dp[d][0] */
        if (ilo >= 1) cu[ilo - 1] = NEG;
        cu[ihi + 1] = NEG;
        short *t = p2; p2 = p1; p1 = cu; cu = t;     /* spent crease drifts off */
    }
    return (int)p1[n];                               /* crease 2n, cell (n,n) */
}

/* ---- the same fold, 8 sparks per crease, for long boughs ---- */
static int nw_fold32(int n, const unsigned char *restrict A,
                     const unsigned char *restrict R,
                     int W, int *buf, int stride)
{
    const int NEG = -(1 << 24);
    int *p2 = buf, *p1 = buf + stride, *cu = buf + 2 * stride;
    for (int t = 0; t < 3 * stride; t++) buf[t] = NEG;
    p1[0] = 0;
    const __m256i vgap = _mm256_set1_epi32(NW_GAP);
    const __m256i v2   = _mm256_set1_epi32(2);
    const __m256i v1   = _mm256_set1_epi32(1);
    const int dmax = 2 * n;
    for (int d = 1; d <= dmax; d++) {
        int ilo = d - n; if (ilo < 0) ilo = 0;
        int c = (d - W + 1) >> 1;   if (c > ilo) ilo = c;
        int ihi = (d < n) ? d : n;
        int f = (d + W) >> 1;       if (f < ihi) ihi = f;
        int lo = (ilo < 1) ? 1 : ilo;
        int hi = (ihi < d - 1) ? ihi : d - 1;
        if (hi >= lo) {
            const unsigned char *pa = A + (lo - 1);
            const unsigned char *pb = R + (n - d + lo);
            const int *q2  = p2 + (lo - 1);
            const int *q1u = p1 + (lo - 1);
            const int *q1l = p1 + lo;
            int *out = cu + lo;
            int len = hi - lo + 1;
            for (int k = 0; k < len; k += 8) {
                __m128i ca = _mm_loadl_epi64((const __m128i *)(pa + k));
                __m128i cb = _mm_loadl_epi64((const __m128i *)(pb + k));
                __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
                __m256i sc = _mm256_sub_epi32(_mm256_and_si256(eq, v2), v1);
                __m256i vd = _mm256_add_epi32(
                    _mm256_loadu_si256((const __m256i *)(q2 + k)), sc);
                __m256i vu = _mm256_loadu_si256((const __m256i *)(q1u + k));
                __m256i vl = _mm256_loadu_si256((const __m256i *)(q1l + k));
                __m256i vg = _mm256_add_epi32(_mm256_max_epi32(vu, vl), vgap);
                _mm256_storeu_si256((__m256i *)(out + k),
                                    _mm256_max_epi32(vd, vg));
            }
        }
        if (ilo == 0) cu[0] = NW_GAP * d;
        if (ihi == d) cu[d] = NW_GAP * d;
        if (ilo >= 1) cu[ilo - 1] = NEG;
        cu[ihi + 1] = NEG;
        int *t = p2; p2 = p1; p1 = cu; cu = t;
    }
    return p1[n];
}
#endif /* __AVX2__ */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if defined(__AVX2__)
    if (n < 96) return nw_rows(n, a, b);   /* guard: setup dominates when tiny */

    /* regime probe: how far off the spine can an optimal path stray?
       L = 2m - n is achievable (gap-free); a path deviating D pays >= 2D gaps
       and scores <= n - 5D, so D <= (n - L)/5 = 2(n - m)/5.                  */
    int m = 0;
    for (int i = 0; i < n; i++) m += (a[i] == b[i]);
    int W = (2 * (n - m)) / 5;
    if (W < 8) W = 8;                      /* keep creases overlapping */
    if (W > n) W = n;                      /* wide regime: clip non-binding */

    const int stride = n + 48;
    unsigned char *cbuf = (unsigned char *)malloc((size_t)2 * (size_t)(n + 64));
    if (!cbuf) return nw_rows(n, a, b);
    unsigned char *A = cbuf, *R = cbuf + (n + 64);
    memcpy(A, a, (size_t)n);
    memset(A + n, 0, 64);
    for (int i = 0; i < n; i++) R[i] = (unsigned char)b[n - 1 - i];
    memset(R + n, 1, 64);                  /* padding lanes are discarded */

    int result;
    if (n <= 4000) {
        short *sb = (short *)malloc((size_t)3 * (size_t)stride * sizeof(short));
        if (!sb) { free(cbuf); return nw_rows(n, a, b); }
        result = nw_fold16(n, A, R, W, sb, stride);
        free(sb);
    } else {
        int *ib = (int *)malloc((size_t)3 * (size_t)stride * sizeof(int));
        if (!ib) { free(cbuf); return nw_rows(n, a, b); }
        result = nw_fold32(n, A, R, W, ib, stride);
        free(ib);
    }
    free(cbuf);
    return result;
#else
    return nw_rows(n, a, b);
#endif
}
