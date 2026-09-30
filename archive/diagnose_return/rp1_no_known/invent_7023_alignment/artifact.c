#include <stdlib.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) \
    && (defined(__SSE2__) || defined(__AVX2__) || defined(_M_X64))
#  include <immintrin.h>
#  define KSIMD 1
#else
#  define KSIMD 0
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* ---------- fallback: exact two-row Needleman-Wunsch, O(n) memory ---------- */
static int nw_rows(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    int *mem = (int *)malloc((size_t)2 * ((size_t)n + 1) * sizeof(int));
    if (!mem) return 0;
    int *prev = mem, *cur = mem + (n + 1);
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        const char ai = a[i - 1];
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int dg = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int up = prev[j] + GAP;
            int lf = cur[j - 1] + GAP;
            int bst = dg;
            if (up > bst) bst = up;
            if (lf > bst) bst = lf;
            cur[j] = bst;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(mem);
    return r;
}

#if KSIMD
/* ---------- the sun-strike: 16 (or 8) crossings judged at once ---------- */
#if defined(__AVX2__)
#  define VW 16
   typedef __m256i kvec;
#  define KLOAD(p)     _mm256_loadu_si256((const __m256i *)(const void *)(p))
#  define KSTORE(p,x)  _mm256_storeu_si256((__m256i *)(void *)(p), (x))
#  define KADD(x,y)    _mm256_add_epi16((x),(y))
#  define KSUB(x,y)    _mm256_sub_epi16((x),(y))
#  define KMAX(x,y)    _mm256_max_epi16((x),(y))
#  define KSET1(v)     _mm256_set1_epi16((short)(v))
static inline kvec kscore(const char *pa, const char *pb)
{
    __m128i ca = _mm_loadu_si128((const __m128i *)(const void *)pa);
    __m128i cb = _mm_loadu_si128((const __m128i *)(const void *)pb);
    __m128i m8 = _mm_cmpeq_epi8(ca, cb);          /* one doubled shadow per byte */
    __m256i m  = _mm256_cvtepi8_epi16(m8);        /* -1 = same dye, 0 = different */
    return _mm256_add_epi16(_mm256_add_epi16(m, m), _mm256_set1_epi16(1)); /* +1 / -1 */
}
#else
#  define VW 8
   typedef __m128i kvec;
#  define KLOAD(p)     _mm_loadu_si128((const __m128i *)(const void *)(p))
#  define KSTORE(p,x)  _mm_storeu_si128((__m128i *)(void *)(p), (x))
#  define KADD(x,y)    _mm_add_epi16((x),(y))
#  define KSUB(x,y)    _mm_sub_epi16((x),(y))
#  define KMAX(x,y)    _mm_max_epi16((x),(y))
#  define KSET1(v)     _mm_set1_epi16((short)(v))
static inline kvec kscore(const char *pa, const char *pb)
{
    __m128i ca = _mm_loadl_epi64((const __m128i *)(const void *)pa);
    __m128i cb = _mm_loadl_epi64((const __m128i *)(const void *)pb);
    __m128i m8 = _mm_cmpeq_epi8(ca, cb);
    __m128i m  = _mm_unpacklo_epi8(m8, m8);       /* 0xFFFF = same dye */
    return _mm_add_epi16(_mm_add_epi16(m, m), _mm_set1_epi16(1));
}
#endif

/* ---------- banded anti-diagonal wavefront; exact, never tiles the floor ----------
   Correctness of the band: for equal-length a,b any alignment has
       #deletions = #insertions = g,  #aligned pairs = n-g,
       score = 2M - n - 3g   and   M <= n-g   =>   score <= n - 5g.
   The straight-diagonal alignment scores L = 2*M0 - n, so the optimum S* >= L,
   hence every optimal alignment has g <= (2n - 2*M0)/5 = W, and along its path
   |i-j| = |#del so far - #ins so far| <= g <= W.  A DP restricted to |i-j| <= W
   therefore contains an optimal path and returns S* exactly.                     */
static int nw_band(int n, const char *a, const char *b, int W)
{
    const short NEG = -32000;             /* < -2n for n <= 15000; -2n is the true floor */
    const int   PAD = 64;
    const size_t row  = (size_t)n + 3 + 2 * (size_t)PAD;
    const size_t clen = (size_t)n + 2 * (size_t)PAD;

    unsigned char *blk =
        (unsigned char *)malloc(3 * row * sizeof(short) + 2 * clen + 64);
    if (!blk) return nw_rows(n, a, b);

    short *A0 = (short *)(void *)blk;
    short *A1 = A0 + row;
    short *A2 = A1 + row;
    char  *pa = (char *)(void *)(A2 + row);
    char  *pb = pa + clen;

    memset(A0, 0, 3 * row * sizeof(short));
    memset(pa, 0x01, clen);
    memset(pb, 0x02, clen);              /* pads can never match each other */
    memcpy(pa, a, (size_t)n);
    for (int k = 0; k < n; k++) pb[k] = b[n - 1 - k];   /* the slanted rope */

    short *p2 = A0, *p1 = A1, *cu = A2;
    const kvec vg = KSET1(2);

    for (int d = 0; d <= 2 * n; d++) {               /* d = the hour of the sun */
        int t0 = d - W;
        int lo = (t0 <= 0) ? 0 : ((t0 + 1) >> 1);    /* ceil((d-W)/2) */
        int hi = (d + W) >> 1;                       /* floor((d+W)/2) */
        if (lo < d - n) lo = d - n;
        if (lo < 0)     lo = 0;
        if (hi > d)     hi = d;
        if (hi > n)     hi = n;

        int ilo = (lo < 1) ? 1 : lo;
        int ihi = (hi < d - 1) ? hi : d - 1;

        if (ihi >= ilo) {
            const char  *qa = pa + (ilo - 1);              /* wall rope, forward  */
            const char  *qb = pb + (n - d + ilo);          /* floor rope, forward */
            const short *sd = p2 + (ilo - 1);              /* dp[i-1][j-1] */
            const short *su = p1 + (ilo - 1);              /* dp[i-1][j]   */
            const short *sl = p1 + ilo;                    /* dp[i][j-1]   */
            short *dst = cu + ilo;
            int len = ihi - ilo + 1;
            for (int t = 0; t < len; t += VW) {
                kvec s  = kscore(qa + t, qb + t);
                kvec vd = KADD(KLOAD(sd + t), s);
                kvec vu = KSUB(KLOAD(su + t), vg);
                kvec vl = KSUB(KLOAD(sl + t), vg);
                KSTORE(dst + t, KMAX(KMAX(vd, vu), vl));   /* drop the bad thread-ends */
            }
        }
        if (lo == 0) cu[0] = (short)(-2 * d);              /* dp[0][d] */
        if (hi == d) cu[d] = (short)(-2 * d);              /* dp[d][0] */
        if (lo >= 1) { cu[lo - 1] = NEG; if (lo >= 2) cu[lo - 2] = NEG; }
        cu[hi + 1] = NEG;
        cu[hi + 2] = NEG;

        short *t = p2; p2 = p1; p1 = cu; cu = t;           /* keep only three rows */
    }
    int r = (int)p1[n];
    free(blk);
    return r;
}
#endif /* KSIMD */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    /* regime 1: a tiny room -- no setup, no vectors, no allocation */
    if (n < 64) {
        int bufA[66], bufB[66];
        int *prev = bufA, *cur = bufB;
        for (int j = 0; j <= n; j++) prev[j] = j * GAP;
        for (int i = 1; i <= n; i++) {
            const char ai = a[i - 1];
            cur[0] = i * GAP;
            for (int j = 1; j <= n; j++) {
                int dg = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
                int up = prev[j] + GAP;
                int lf = cur[j - 1] + GAP;
                int bst = dg;
                if (up > bst) bst = up;
                if (lf > bst) bst = lf;
                cur[j] = bst;
            }
            int *t = prev; prev = cur; cur = t;
        }
        return prev[n];
    }

#if KSIMD
    /* regime 2: the ropes announce the corridor width, then the sun does the work.
       16-bit tallies are valid while 2n < 32000.                                  */
    if (n <= 15000) {
        int m0 = 0;
        for (int i = 0; i < n; i++) m0 += (a[i] == b[i]);   /* doubled shadows */
        long long W = (2LL * (long long)(n - m0)) / 5 + 4;  /* proof + safety margin */
        if (W > n) W = n;                                   /* W == n => full grid */
        if (W < 8) W = 8;
        if (W > n) W = n;
        return nw_band(n, a, b, (int)W);
    }
#endif

    /* regime 3: too large for 16-bit tallies, or no SIMD at all */
    return nw_rows(n, a, b);
}
