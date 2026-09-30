#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH (-1)
#define GAP      (-2)

/* ---------- plain two-row DP: guarded fallback for tiny n ---------- */
static int nw_full(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int r;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = GAP * j;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = GAP * i;
        for (int j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int l = cur[j - 1] + GAP;
            int bst = d; if (u > bst) bst = u; if (l > bst) bst = l;
            cur[j] = bst;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---------- banded row-wise DP, 32-bit: fallback (no AVX2 / large n) ---------- */
static int nw_band32(int n, const char *a, const char *b, int W)
{
    const int NEG = -(1 << 26);
    int *prev = (int *)malloc((size_t)(n + 3) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 3) * sizeof(int));
    int r;
    if (!prev || !cur) { free(prev); free(cur); return nw_full(n, a, b); }
    for (int j = 0; j <= n + 2; j++) { prev[j] = NEG; cur[j] = NEG; }
    for (int j = 0; j <= n && j <= W; j++) prev[j] = GAP * j;
    for (int i = 1; i <= n; i++) {
        int lo = i - W, hi = i + W;
        char ai = a[i - 1];
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        if (lo >= 2) cur[lo - 1] = NEG;
        else         cur[0] = (i <= W) ? GAP * i : NEG;
        for (int j = lo; j <= hi; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int l = cur[j - 1] + GAP;
            int bst = d; if (u > bst) bst = u; if (l > bst) bst = l;
            cur[j] = bst;
        }
        if (hi + 1 <= n + 1) cur[hi + 1] = NEG;
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

#if defined(__AVX2__)
/* ---------- the flute: 16 houses marked per beat, along one anti-diagonal,
              inside the road of width W. Only three anti-diagonals ever exist. */
static int nw_band16_avx2(int n, const char *a, const char *b, int W)
{
    const int   PAD = 64, OFF = 32;
    const short NEGS = -30000;
    int  alen = n + 2 * PAD, slen = n + 1 + 2 * OFF + 32;
    char  *pa = (char  *)malloc((size_t)alen);
    char  *pb = (char  *)malloc((size_t)alen);
    short *b0 = (short *)malloc((size_t)slen * sizeof(short));
    short *b1 = (short *)malloc((size_t)slen * sizeof(short));
    short *b2 = (short *)malloc((size_t)slen * sizeof(short));
    short *p2, *p1, *c;
    int res;
    __m256i vgap, vm1, vneg;
    if (!pa || !pb || !b0 || !b1 || !b2) {
        free(pa); free(pb); free(b0); free(b1); free(b2);
        return nw_band32(n, a, b, W);
    }
    memset(pa, 0x7f, (size_t)alen);          /* padding bases that never match */
    memset(pb, 0x5a, (size_t)alen);
    memcpy(pa + PAD, a, (size_t)n);
    for (int t = 0; t < n; t++) pb[PAD + t] = b[n - 1 - t];   /* b reversed once */
    for (int t = 0; t < slen; t++) { b0[t] = NEGS; b1[t] = NEGS; b2[t] = NEGS; }
    b1[OFF + 0] = 0;                         /* the road's first house: dp[0][0] */
    p2 = b0; p1 = b1; c = b2;
    vgap = _mm256_set1_epi16(GAP);
    vm1  = _mm256_set1_epi16(-1);
    vneg = _mm256_set1_epi16(NEGS);
    for (int d = 1; d <= 2 * n; d++) {
        int lo = d - W, hi = (d + W) >> 1;
        lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);
        if (lo < d - n) lo = d - n;
        if (hi > n) hi = n;
        if (hi > d) hi = d;
        _mm256_storeu_si256((__m256i *)(c + OFF + lo - 16), vneg);
        for (int i = lo; i <= hi; i += 16) {
            __m128i va = _mm_loadu_si128((const __m128i *)(pa + PAD + i - 1));
            __m128i vb = _mm_loadu_si128((const __m128i *)(pb + PAD + n - d + i));
            __m128i eq = _mm_cmpeq_epi8(va, vb);           /* petal = -1, knot = 0 */
            __m256i e  = _mm256_cvtepi8_epi16(eq);
            __m256i sb = _mm256_sub_epi16(vm1, _mm256_add_epi16(e, e)); /* +1 / -1 */
            __m256i x = _mm256_add_epi16(
                            _mm256_loadu_si256((const __m256i *)(p2 + OFF + i - 1)), sb);
            __m256i y = _mm256_add_epi16(
                            _mm256_loadu_si256((const __m256i *)(p1 + OFF + i - 1)), vgap);
            __m256i z = _mm256_add_epi16(
                            _mm256_loadu_si256((const __m256i *)(p1 + OFF + i)),     vgap);
            x = _mm256_max_epi16(x, y);
            x = _mm256_max_epi16(x, z);
            _mm256_storeu_si256((__m256i *)(c + OFF + i), x);
        }
        _mm256_storeu_si256((__m256i *)(c + OFF + hi + 1), vneg);
        { short *t = p2; p2 = p1; p1 = c; c = t; }
    }
    res = (int)p1[OFF + n];
    free(pa); free(pb); free(b0); free(b1); free(b2);
    return res;
}
#endif

/* ---------- Seed 3: every fist weighed, all but the lightest flung away.
   Returns the least knot-count over ALL placements of one slip.
   mm(p,q) = D[p] + (F[q-1]-F[p]) + (D[n]-D[q]),  0 <= p < q <= n.        */
static int horse_best_mm(int n, const int *D, const int *F)
{
    int bestpair = 0x3fffffff;
    int minf = D[0] - F[0];
    for (int q = 1; q <= n; q++) {
        int v = minf + (F[q - 1] - D[q]);
        if (v < bestpair) bestpair = v;
        if (q <= n - 1) { int f = D[q] - F[q]; if (f < minf) minf = f; }
    }
    return D[n] + bestpair;
}

int kernel(int n, const char *a, const char *b)
{
    int *D, *F, m, best, W, mm;

    if (n <= 0) return 0;
    if (n < 64) return nw_full(n, a, b);          /* guard: overhead on small n */

    D = (int *)malloc((size_t)(n + 1) * sizeof(int));
    F = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!D || !F) { free(D); free(F); return nw_full(n, a, b); }

    /* the fixed furrow, walked house by house: knots on the unslipped road */
    D[0] = 0;
    for (int t = 1; t <= n; t++) D[t] = D[t - 1] + (a[t - 1] != b[t - 1]);
    m = D[n];
    best = n - 2 * m;                             /* the horse standing nowhere */

    /* the horse in every doorway: second row slips forward */
    F[0] = 0;
    for (int t = 1; t <= n - 1; t++) F[t] = F[t - 1] + (a[t] != b[t - 1]);
    mm = horse_best_mm(n, D, F);
    { int s = n - 5 - 2 * mm; if (s > best) best = s; }

    /* the mirror trial: the other furrow held fixed instead (house against house) */
    F[0] = 0;
    for (int t = 1; t <= n - 1; t++) F[t] = F[t - 1] + (b[t] != a[t - 1]);
    mm = horse_best_mm(n, D, F);
    { int s = n - 5 - 2 * mm; if (s > best) best = s; }

    free(D); free(F);

    /* the kept fist tells how wide the road must be laid out:
       score <= n - 5g  =>  g* <= (n - best)/5, and |i-j| <= g* on the path.   */
    W = (n - best) / 5;
    if (W < 0) W = 0;
    if (W <= 1) return best;   /* REGIME A: one slip provably suffices. O(n).  */
    if (W > n) W = n;

    /* REGIME B: lay out the road W houses to each side. */
#if defined(__AVX2__)
    if (n <= 6000) return nw_band16_avx2(n, a, b, W);
#endif
    return nw_band32(n, a, b, W);
}
