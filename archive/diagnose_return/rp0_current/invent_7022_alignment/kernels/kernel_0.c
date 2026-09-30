#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH (-1)
#define GAP      (-2)

/* ------------------------------------------------------------------------ *
 * "a hooded memory kneeling at each crossing in turn" -- the plain path.
 * Used for tiny trays (where worm bookkeeping dominates) and on allocation
 * failure.  Rolling row, exact Needleman-Wunsch.
 * ------------------------------------------------------------------------ */
static int nw_rowdp(int n, const char *a, const char *b)
{
    int stackrow[1025];
    int *row;
    int i, j, r;
    if (n <= 0) return 0;
    row = (n + 1 <= 1025) ? stackrow
                          : (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!row) return 0;
    for (j = 0; j <= n; j++) row[j] = j * GAP;
    for (i = 1; i <= n; i++) {
        int diagp = row[0];
        char ai = a[i - 1];
        row[0] = i * GAP;
        for (j = 1; j <= n; j++) {
            int up = row[j];
            int dg = diagp + (ai == b[j - 1] ? MATCH : MISMATCH);
            int uu = up + GAP;
            int ll = row[j - 1] + GAP;
            int best = dg > uu ? dg : uu;
            if (ll > best) best = ll;
            diagp = up;
            row[j] = best;
        }
    }
    r = row[n];
    if (row != stackrow) free(row);
    return r;
}

/* ------------------------------------------------------------------------ *
 * The ink-worm, 16-bit mounds.  One body, one cell wide, sweeping the
 * anti-diagonals d = i + j.  Cords laid crosswise: b is written out
 * backwards ONCE, so that along a slant both cords read forward and a whole
 * run of "places where symbols meet" is one compare.
 *
 *   cur[i] = D[i][d-i] = max( p2[i-1] + s , p1[i-1] + GAP , p1[i] + GAP )
 *   s      = (a[i-1] == brev[n-d+i]) ? +1 : -1
 *
 * Only cells with |i-j| <= W are visited (W from the straight-grain pass).
 * Two guard mounds per diagonal (at lo-1 and hi+1) hold NEG so the band edge
 * needs no branch.  Safe because every visited cell's diagonal predecessor is
 * itself in-band, so no computed value is ever guard-derived.
 * ------------------------------------------------------------------------ */
static int nw_wave16(int n, const char *a, const char *b, int W)
{
    const short NEGV = -32000;           /* below any real value (>= -1.8n) */
    const int   sz   = n + 48;
    short *buf = (short *)malloc((size_t)3 * (size_t)sz * sizeof(short));
    char  *brb = (char  *)malloc((size_t)n + 64);
    short *p2, *p1, *cur;
    int d, result = 0;
    size_t t;
    const int twon = 2 * n;

    if (!buf || !brb) { free(buf); free(brb); return nw_rowdp(n, a, b); }
    for (t = 0; t < (size_t)n; t++) brb[t] = b[n - 1 - (int)t];
    memset(brb + n, 0, 64);
    for (t = 0; t < (size_t)3 * (size_t)sz; t++) buf[t] = NEGV;

    p2  = buf + 16;
    p1  = buf + sz + 16;
    cur = buf + 2 * sz + 16;

    for (d = 0; d <= twon; d++) {
        int lo = d - W, hi, ilo, ihi, i, base;
        lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);      /* ceil((d-W)/2), >= 0   */
        if (lo < d - n) lo = d - n;
        hi = (d + W) >> 1;                         /* floor((d+W)/2)        */
        if (hi > n) hi = n;
        if (hi > d) hi = d;

        base = n - d;
        ilo  = (lo < 1) ? 1 : lo;
        ihi  = (hi > d - 1) ? (d - 1) : hi;

        i = ilo;
#if defined(__AVX512BW__) && defined(__AVX512F__)
        {
            const __m512i vg  = _mm512_set1_epi16((short)GAP);
            const __m512i vm1 = _mm512_set1_epi16((short)-1);
            for (; i + 31 <= ihi; i += 32) {
                __m256i ca = _mm256_loadu_si256((const __m256i *)(a + i - 1));
                __m256i cb = _mm256_loadu_si256((const __m256i *)(brb + base + i));
                __m256i eq = _mm256_cmpeq_epi8(ca, cb);
                __m512i ew = _mm512_cvtepi8_epi16(eq);        /* -1 if equal */
                /* s = -1 - 2*ew :  equal -> +1 , differ -> -1 */
                __m512i s  = _mm512_sub_epi16(vm1, _mm512_add_epi16(ew, ew));
                __m512i D  = _mm512_loadu_si512((const void *)(p2 + i - 1));
                __m512i U  = _mm512_loadu_si512((const void *)(p1 + i - 1));
                __m512i L  = _mm512_loadu_si512((const void *)(p1 + i));
                __m512i r  = _mm512_max_epi16(_mm512_add_epi16(D, s),
                             _mm512_max_epi16(_mm512_add_epi16(U, vg),
                                              _mm512_add_epi16(L, vg)));
                _mm512_storeu_si512((void *)(cur + i), r);
            }
        }
#endif
#if defined(__AVX2__)
        {
            const __m256i vg = _mm256_set1_epi16((short)GAP);
            const __m256i vp = _mm256_set1_epi16((short)MATCH);
            const __m256i vm = _mm256_set1_epi16((short)MISMATCH);
            for (; i + 15 <= ihi; i += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(a + i - 1));
                __m128i cb = _mm_loadu_si128((const __m128i *)(brb + base + i));
                __m128i eq = _mm_cmpeq_epi8(ca, cb);
                __m256i ew = _mm256_cvtepi8_epi16(eq);
                __m256i s  = _mm256_blendv_epi8(vm, vp, ew);
                __m256i D  = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i U  = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i L  = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m256i r  = _mm256_max_epi16(_mm256_add_epi16(D, s),
                             _mm256_max_epi16(_mm256_add_epi16(U, vg),
                                              _mm256_add_epi16(L, vg)));
                _mm256_storeu_si256((__m256i *)(cur + i), r);
            }
        }
#endif
        {   /* scalar remainder; also the whole loop on non-x86 (auto-vec) */
            const short * __restrict q2 = p2;
            const short * __restrict q1 = p1;
            short       * __restrict cc = cur;
            const char  * __restrict aa = a;
            const char  * __restrict bb = brb;
            for (; i <= ihi; i++) {
                int s   = (aa[i - 1] == bb[base + i]) ? MATCH : MISMATCH;
                int dg  = (int)q2[i - 1] + s;
                int uu  = (int)q1[i - 1] + GAP;
                int ll  = (int)q1[i] + GAP;
                int bst = dg > uu ? dg : uu;
                if (ll > bst) bst = ll;
                cc[i] = (short)bst;
            }
        }

        if (lo == 0) cur[0] = (short)(GAP * d);    /* cell (0,d)  */
        if (hi == d) cur[d] = (short)(GAP * d);    /* cell (d,0)  */
        cur[lo - 1] = NEGV;                        /* guard mounds */
        cur[hi + 1] = NEGV;

        if (d == twon) { result = (int)cur[n]; break; }
        { short *tmp = p2; p2 = p1; p1 = cur; cur = tmp; }  /* the worm curls back */
    }
    free(buf); free(brb);
    return result;
}

/* Same worm, 32-bit mounds, for trays too large for 16-bit sand. */
static int nw_wave32(int n, const char *a, const char *b, int W)
{
    const int NEGV = -1000000000;
    const int sz   = n + 48;
    int  *buf = (int  *)malloc((size_t)3 * (size_t)sz * sizeof(int));
    char *brb = (char *)malloc((size_t)n + 64);
    int *p2, *p1, *cur;
    int d, result = 0;
    size_t t;
    const int twon = 2 * n;

    if (!buf || !brb) { free(buf); free(brb); return nw_rowdp(n, a, b); }
    for (t = 0; t < (size_t)n; t++) brb[t] = b[n - 1 - (int)t];
    memset(brb + n, 0, 64);
    for (t = 0; t < (size_t)3 * (size_t)sz; t++) buf[t] = NEGV;

    p2  = buf + 16;
    p1  = buf + sz + 16;
    cur = buf + 2 * sz + 16;

    for (d = 0; d <= twon; d++) {
        int lo = d - W, hi, ilo, ihi, i, base;
        lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);
        if (lo < d - n) lo = d - n;
        hi = (d + W) >> 1;
        if (hi > n) hi = n;
        if (hi > d) hi = d;

        base = n - d;
        ilo  = (lo < 1) ? 1 : lo;
        ihi  = (hi > d - 1) ? (d - 1) : hi;

        {
            const int  * __restrict q2 = p2;
            const int  * __restrict q1 = p1;
            int        * __restrict cc = cur;
            const char * __restrict aa = a;
            const char * __restrict bb = brb;
            for (i = ilo; i <= ihi; i++) {
                int s   = (aa[i - 1] == bb[base + i]) ? MATCH : MISMATCH;
                int dg  = q2[i - 1] + s;
                int uu  = q1[i - 1] + GAP;
                int ll  = q1[i] + GAP;
                int bst = dg > uu ? dg : uu;
                if (ll > bst) bst = ll;
                cc[i] = bst;
            }
        }

        if (lo == 0) cur[0] = GAP * d;
        if (hi == d) cur[d] = GAP * d;
        cur[lo - 1] = NEGV;
        cur[hi + 1] = NEGV;

        if (d == twon) { result = cur[n]; break; }
        { int *tmp = p2; p2 = p1; p1 = cur; cur = tmp; }
    }
    free(buf); free(brb);
    return result;
}

/* ------------------------------------------------------------------------ *
 * kernel: first the straight-grain pass (the height the worm earns walking
 * the main furrow), which both fences the tray and tells it which regime it
 * is in; then the worm.
 * ------------------------------------------------------------------------ */
int kernel(int n, const char *a, const char *b)
{
    int H = 0, W, i;
    if (n <= 0) return 0;

    /* straight grain: Hamming distance, vectorized, O(n) */
    for (i = 0; i < n; i++) H += (a[i] != b[i]);

    if (H == 0) return n;                 /* cords identical: nothing to do */

    /* Exact band: any optimal path has max offset m <= floor(2H/5).
       Proof: score = n - 2X - 5k with k = Ia = Ib >= m, so a path through
       offset m scores <= n - 5m; the straight grain already guarantees
       S* >= n - 2H; n - 5m < n - 2H whenever m > 2H/5.                     */
    W = (2 * H) / 5;
    if (W < 2) W = 2;                     /* keeps every diagonal non-empty */
    if (W > n) W = n;

    if (n < 96)     return nw_rowdp(n, a, b);   /* tray too small for a worm */
    if (n <= 15000) return nw_wave16(n, a, b, W);
    return nw_wave32(n, a, b, W);
}
