#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH (-1)
#define GAP      (-2)
#define NEG_BIG  (-1061109568)      /* 0xC0C0C0C0 : memset-able sentinel */

/* ---------- fallback: plain two-row Needleman-Wunsch ---------------------- */
static int nw_simple(int n, const char *a, const char *b)
{
    int *prev, *cur, *tmp, i, j, r;
    prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (j = 0; j <= n; j++) prev[j] = -2 * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = -2 * i;
        for (j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int l = cur[j - 1] + GAP;
            int bst = d;
            if (u > bst) bst = u;
            if (l > bst) bst = l;
            cur[j] = bst;
        }
        tmp = prev; prev = cur; cur = tmp;
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---------- the wavefront of furrows: banded anti-diagonal DP -------------
   Cells are indexed by (t = i+j, i).  Along a fixed t nothing depends on
   anything else -- the flute's one-note-per-house clock.  Parents:
     diagonal (i-1,j-1) -> buffer t-2 at index i-1
     up       (i-1,j)   -> buffer t-1 at index i-1
     left     (i,j-1)   -> buffer t-1 at index i
   Requires W >= 1.  Each R* buffer has `slots` ints; usable index range of
   (R+2) is [-2, slots-3].                                                   */
static int nw_band(int n, const char *pa, const char *pb, int W,
                   int *R0, int *R1, int *R2, int slots)
{
    size_t nb = (size_t)slots * sizeof(int);
    int *p2, *p1, *cu, t, res = 0;
#if defined(__AVX2__)
    const __m256i vg  = _mm256_set1_epi32(GAP);
    const __m256i vp1 = _mm256_set1_epi32(MATCH);
    const __m256i vm1 = _mm256_set1_epi32(MISMATCH);
#endif
    memset(R0, 0xC0, nb); memset(R1, 0xC0, nb); memset(R2, 0xC0, nb);
    p2 = R0 + 2; p1 = R1 + 2; cu = R2 + 2;
    p2[0] = 0;                               /* dp[0][0], anti-diagonal t=0 */

    for (t = 1; t <= 2 * n; t++) {
        int lo = (t - W + 1) >> 1;           /* ceil((t-W)/2) */
        int hi = (t + W) >> 1;               /* floor((t+W)/2) */
        int c  = t - n;
        int ilo, ihi, i, *tmp;
        if (c > lo) lo = c;
        if (lo < 0) lo = 0;
        if (hi > t) hi = t;
        if (hi > n) hi = n;
        ilo = lo; ihi = hi;
        if (lo == 0) ilo = 1;                /* (0,t) is a boundary cell   */
        if (hi == t) ihi = t - 1;            /* (t,0) is a boundary cell   */

        i = ilo;
#if defined(__AVX2__)
        for (; i <= ihi; i += 8) {
            uint64_t bv, bvr;
            __m128i va, vb, eq;
            __m256i sv, d, u, l, bst;
            /* 8 beasts of furrow 1, ascending */
            va = _mm_loadl_epi64((const __m128i *)(pa + (i - 1)));
            /* 8 beasts of furrow 2, descending (j = t-i falls as i rises)  */
            memcpy(&bv, pb + (t - i - 8), 8);
            bvr = __builtin_bswap64(bv);
            vb  = _mm_loadl_epi64((const __m128i *)&bvr);
            eq  = _mm_cmpeq_epi8(va, vb);            /* lotus vs knot      */
            sv  = _mm256_blendv_epi8(vm1, vp1, _mm256_cvtepi8_epi32(eq));
            d   = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
            u   = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            l   = _mm256_loadu_si256((const __m256i *)(p1 + i));
            bst = _mm256_add_epi32(d, sv);
            bst = _mm256_max_epi32(bst, _mm256_add_epi32(u, vg));
            bst = _mm256_max_epi32(bst, _mm256_add_epi32(l, vg));
            _mm256_storeu_si256((__m256i *)(cu + i), bst);
        }
#else
        for (; i <= ihi; i++) {
            int j = t - i;
            int s = (pa[i - 1] == pb[j - 1]) ? MATCH : MISMATCH;
            int d = p2[i - 1] + s;
            int u = p1[i - 1] + GAP;
            int l = p1[i] + GAP;
            int bst = d;
            if (u > bst) bst = u;
            if (l > bst) bst = l;
            cu[i] = bst;
        }
#endif
        /* boundary cells written after the loop: a vector tail may have
           spilled junk over index t (harmless elsewhere, never read)      */
        if (lo == 0) cu[0] = -2 * t;
        if (hi == t) cu[t] = -2 * t;
        cu[lo - 1] = NEG_BIG;                /* the flying fish            */
        cu[hi + 1] = NEG_BIG;

        if (t == 2 * n) res = cu[n];
        tmp = p2; p2 = p1; p1 = cu; cu = tmp;
    }
    return res;
}

/* ------------------------------- kernel ---------------------------------- */
int kernel(int n, const char *a, const char *b)
{
    int P0 = 0, M1 = 0, M2 = 0;
    int minf1 = 0, minf2 = 0;
    int bestX1 = 0x3FFFFFFF, bestX2 = 0x3FFFFFFF;
    int q, d0, LB, s1, s2, W, Wtry, Wneed, S, slots;
    int *R, *R0, *R1, *R2;
    char *PA, *pa, *pb;

    if (n <= 0) return 0;
    if (n < 32) return nw_simple(n, a, b);

    /* ---- the horse's walk: every 0- and 1-slip trial, O(n) total --------
       trial (down at p, right at q>=p):  X = d0 + f(p) + g(q)
       f1(p)=P0[p]-M1[p]  g1(q)=M1[q]-P0[q+1]   (furrow 2 slips forward)
       f2/g2: the mirror trial (furrow 1 slips forward).                   */
    for (q = 0; q < n; q++) {
        int f1 = P0 - M1, f2 = P0 - M2, P0n, c1, c2;
        if (f1 < minf1) minf1 = f1;
        if (f2 < minf2) minf2 = f2;
        P0n = P0 + (a[q] != b[q]);
        c1 = minf1 + M1 - P0n; if (c1 < bestX1) bestX1 = c1;
        c2 = minf2 + M2 - P0n; if (c2 < bestX2) bestX2 = c2;
        if (q + 1 < n) { M1 += (a[q + 1] != b[q]); M2 += (a[q] != b[q + 1]); }
        P0 = P0n;
    }
    d0 = P0;

    /* ---- weigh the fists, keep the lightest ---------------------------- */
    LB = n - 2 * d0;                       /* the horse stands nowhere     */
    s1 = n - 5 - 2 * bestX1; if (s1 > LB) LB = s1;
    s2 = n - 5 - 2 * bestX2; if (s2 > LB) LB = s2;

    /* score <= n - 5*deviation  =>  optimal path lives in |i-j| <= W       */
    W = (n - LB) / 5;
    if (W <= 1) return LB;   /* regime A: <=1 slip is provably optimal, O(n) */
    if (W > n) W = n;

    /* ---- regime B: banded wavefront ------------------------------------ */
    slots = n + 26;
    R  = (int  *)malloc((size_t)slots * 3 * sizeof(int));
    PA = (char *)malloc((size_t)(n + 80) * 2);
    if (!R || !PA) { free(R); free(PA); return nw_simple(n, a, b); }
    memset(PA, 0, (size_t)(n + 80) * 2);
    pa = PA + 40;
    pb = PA + (n + 80) + 40;
    memcpy(pa, a, (size_t)n);
    memcpy(pb, b, (size_t)n);
    R0 = R; R1 = R + slots; R2 = R + 2 * slots;

    Wtry = (W < 16) ? W : 16;              /* cheap probe tightens LB      */
    S = nw_band(n, pa, pb, Wtry, R0, R1, R2, slots);
    Wneed = (n - S) / 5;
    if (Wneed > Wtry) {
        if (Wneed > n) Wneed = n;
        S = nw_band(n, pa, pb, Wneed, R0, R1, R2, slots);
    }
    free(R); free(PA);
    return S;
}
