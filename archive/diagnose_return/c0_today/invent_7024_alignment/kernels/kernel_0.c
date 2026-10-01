#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH (-1)
#define GAP      (-2)

/* ---- fallback: exact NW with two rolling rows (O(n) memory) ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    int *prev, *cur, r;
    prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int v = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int l = cur[j - 1] + GAP;
            if (u > v) v = u;
            if (l > v) v = l;
            cur[j] = v;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---- the native's crossings: the straight one and every single-slip one.
       Each is a dependency-free sweep; the highest pile is a certified floor. */
static int crossing_bound(int n, const char *a, const char *b)
{
    int maxs = n - 1, best;
    if (maxs > 8) maxs = 8;
    if (maxs < 0) maxs = 0;
    best = -4 * n;
    for (int s = 0; s <= maxs; s++) {
        int L = n - s, m1 = 0, m2 = 0, s1, s2;
        for (int k = 0; k < L; k++) {
            m1 += (a[k + s] == b[k]);
            m2 += (a[k] == b[k + s]);
        }
        s1 = 2 * m1 - n - 3 * s;   /* 2s gap columns (-4s) + (2m - (n-s)) */
        s2 = 2 * m2 - n - 3 * s;
        if (s1 > best) best = s1;
        if (s2 > best) best = s2;
    }
    return best;
}

int kernel(int n, const char *a, const char *b)
{
    int16_t *mem, *p2, *p1, *cu, *B0, *B1, *B2;
    char *pa, *pb;
    int W, L, NEG, sz, d, res;

    if (n <= 0) return 0;
    if (n < 64 || n > 16000) return nw_rows(n, a, b);   /* guarded regimes */

    /* --- how far may the door break?  floor L  =>  any path reaching
       |i-j| = d burns >= 2d gap columns and loses d fire-chances, so its
       pile <= n - 5d.  Hence d <= (n-L)/5.  Take one extra for safety. --- */
    L = crossing_bound(n, a, b);
    W = (n - L) / 5 + 1;
    if (W < 2) W = 2;
    if (W > n) W = n;            /* clamp: band stops binding -> full DP */

    NEG = -2 * n - 64;           /* below every reachable score, no overflow */
    sz  = n + 96;

    mem = (int16_t *)malloc((size_t)3 * sz * sizeof(int16_t));
    pa  = (char *)malloc((size_t)n + 64);
    pb  = (char *)malloc((size_t)n + 64);
    if (!mem || !pa || !pb) { free(mem); free(pa); free(pb); return nw_rows(n, a, b); }

    memcpy(pa, a, (size_t)n);
    memset(pa + n, 0x01, 64);
    for (d = 0; d < n; d++) pb[d] = b[n - 1 - d];   /* shadow row, reversed queue */
    memset(pb + n, 0x02, 64);

    for (d = 0; d < 3 * sz; d++) mem[d] = (int16_t)NEG;
    B0 = mem + 32; B1 = mem + sz + 32; B2 = mem + 2 * sz + 32;

    B0[0] = 0;                   /* cell (0,0), diagonal d = 0 */
    p2 = B2;                     /* diagonal d-2 (empty at start) */
    p1 = B0;                     /* diagonal d-1 = 0            */
    cu = B1;

#if defined(__AVX2__)
    {
    const __m256i vtwo  = _mm256_set1_epi16(2);
    const __m256i vmone = _mm256_set1_epi16(-1);
    const __m256i vgap  = _mm256_set1_epi16(GAP);
#endif
    for (d = 1; d <= 2 * n; d++) {
        int lo = d - n, hi = d - 1, t1, hb, base, i;
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        t1 = d - W;
        if (t1 > 0) { int lb = (t1 + 1) >> 1; if (lb > lo) lo = lb; }
        hb = (d + W) >> 1; if (hb < hi) hi = hb;

        base = n - d;            /* b[j-1] == pb[base + i], contiguous ascending */
        i = lo;
#if defined(__AVX2__)
        for (; i + 15 <= hi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(pa + (i - 1)));
            __m128i cb = _mm_loadu_si128((const __m128i *)(pb + (base + i)));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i sub = _mm256_add_epi16(_mm256_and_si256(eq, vtwo), vmone);
            __m256i dg = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(p2 + (i - 1))), sub);
            __m256i g1 = _mm256_loadu_si256((const __m256i *)(p1 + (i - 1)));
            __m256i g2 = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i gp = _mm256_add_epi16(_mm256_max_epi16(g1, g2), vgap);
            _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi16(dg, gp));
        }
#endif
        for (; i <= hi; i++) {
            int sv = (pa[i - 1] == pb[base + i]) ? MATCH : MISMATCH;
            int v = (int)p2[i - 1] + sv;
            int u = (int)p1[i - 1] + GAP;
            int l = (int)p1[i] + GAP;
            if (u > v) v = u;
            if (l > v) v = l;
            cu[i] = (int16_t)v;
        }
        cu[lo - 1] = (int16_t)NEG;      /* the tower: unread on both flanks */
        cu[hi + 1] = (int16_t)NEG;
        if (d <= n && d <= W) {         /* the two grid edges, when in band */
            cu[0] = (int16_t)(GAP * d);
            cu[d] = (int16_t)(GAP * d);
        }
        { int16_t *t = p2; p2 = p1; p1 = cu; cu = t; }
    }
#if defined(__AVX2__)
    }
#endif

    res = (int)p1[n];               /* the queen's threshold, cell (n,n) */
    free(mem); free(pa); free(pb);
    return res;
}
