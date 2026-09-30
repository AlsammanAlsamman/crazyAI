/* The horse's walk: a slip-certificate corridor with a flute-paced wavefront.
 *
 * Phase 1 (Seed 1 + Seed 3): weigh every one-slip fist in O(n) via two Kadane
 *   scans (both slip directions) plus the unslipped trial.  Keep the lightest.
 * Phase 2 (Seed 3): the kept fist's weight bounds the number of gap-pairs the
 *   true optimum may use:  d* <= (n - S)/5 = D.  Every alignment path cell then
 *   obeys |i-j| <= D.  Everything outside that corridor goes to the fish.
 * Phase 3 (Seed 2): fill the corridor anti-diagonal by anti-diagonal - one
 *   flute note per anti-diagonal, every house of that note independent, 16
 *   houses per breath in int16 SIMD.  b is stored reversed so both beast
 *   streams march forward in memory.
 *
 * Exact Needleman-Wunsch (match +1, mismatch -1, gap -2) for |a| = |b| = n.
 * Guards: tiny n -> plain rolling DP; D == 0 -> answer is the unslipped count;
 * D clamped to n so the worst case degenerates to a full SIMD wavefront and can
 * never be slower than the reference; int32 wavefront when int16 could overflow.
 */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <immintrin.h>
#endif

#define KMATCH     1
#define KMISMATCH -1
#define KGAP      -2

/* ---- guard: tiny inputs use the simple path, no machinery, no malloc ---- */
#define SMALL_N 64
static int small_dp(int n, const char *a, const char *b)
{
    int prev[SMALL_N + 1], cur[SMALL_N + 1];
    int i, j;
    for (j = 0; j <= n; j++) prev[j] = KGAP * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = KGAP * i;
        for (j = 1; j <= n; j++) {
            int d = prev[j - 1] + ((ai == b[j - 1]) ? KMATCH : KMISMATCH);
            int u = prev[j] + KGAP;
            int l = cur[j - 1] + KGAP;
            int m = (d > u) ? d : u;
            cur[j] = (m > l) ? m : l;
        }
        for (j = 0; j <= n; j++) prev[j] = cur[j];
    }
    return prev[n];
}

/* ---- Phase 1: the horse walks every doorway; all fists weighed in O(n) ---- */
static int slip_certificate(int n, const char *a, const char *b, int *S0out)
{
    int totalM = 0, q;
    int best, m0, runA, GA, runB, GB, G, S1, S0;
    for (q = 0; q < n; q++) totalM += (a[q] != b[q]);
    S0 = n - 2 * totalM;              /* the horse standing nowhere at all */
    *S0out = S0;
    best = S0;

    /* Every one-slip trial, both slip directions, as two max-subarray scans.
     * Bubble on [p,q]:  knots = totalM - M[p] - sum_{p<i<=q} (M[i] - N[i]).   */
    m0 = (a[0] != b[0]);
    runA = m0; GA = m0;
    runB = m0; GB = m0;
    for (q = 1; q < n; q++) {
        int Mq = (a[q] != b[q]);
        int Na = (a[q] != b[q - 1]);   /* furrow b slips forward */
        int Nb = (a[q - 1] != b[q]);   /* furrow a slips forward */
        int cA = runA + (Mq - Na);
        int cB = runB + (Mq - Nb);
        runA = (Mq > cA) ? Mq : cA; if (runA > GA) GA = runA;
        runB = (Mq > cB) ? Mq : cB; if (runB > GB) GB = runB;
    }
    G = (GA > GB) ? GA : GB;
    S1 = n - 5 - 2 * (totalM - G);     /* one gap-pair costs exactly 5 */
    if (S1 > best) best = S1;
    return best;                       /* the one kept fist */
}

/* ---- Phase 3a: flute-paced wavefront, int16 houses (16 per breath) ---- */
static int wave16(int n, const unsigned char *__restrict ap,
                  const unsigned char *__restrict brp, int D,
                  int16_t *blk)
{
    const int16_t SENT = -30000;       /* the fish: never wins a max */
    const int stride = n + 48;
    int16_t *B0 = blk + 8, *B1 = blk + stride + 8, *B2 = blk + 2 * stride + 8;
    int16_t *B[3];
    int t, k;
#ifdef __AVX2__
    const __m256i vneg1 = _mm256_set1_epi16(-1);
    const __m256i vgap  = _mm256_set1_epi16(2);
#endif
    for (k = 0; k < 3 * stride; k++) blk[k] = SENT;
    B[0] = B0; B[1] = B1; B[2] = B2;
    B[0][0] = 0;                        /* dp[0][0] */

    for (t = 1; t <= 2 * n; t++) {
        int16_t       *__restrict cur = B[t % 3];
        const int16_t *__restrict p1  = B[(t + 2) % 3];   /* diagonal t-1 */
        const int16_t *__restrict p2  = B[(t + 1) % 3];   /* diagonal t-2 */
        int ilo = (t > n) ? (t - n) : 0;
        int ihi = (t < n) ? t : n;
        int blo = (t > D) ? ((t - D + 1) >> 1) : 0;       /* |2i - t| <= D */
        int bhi = (t + D) >> 1;
        int lo  = (ilo > blo) ? ilo : blo;
        int hi  = (ihi < bhi) ? ihi : bhi;
        int st = lo, en = hi, off = n - t, i;
        int16_t bv = (int16_t)(KGAP * t);

        if (t <= n) {                                    /* in-corridor edges */
            if (lo == 0) { cur[0] = bv; st = 1; }
            if (hi == t) { cur[t] = bv; en = t - 1; }
        }
        i = st;
#ifdef __AVX2__
        for (; i + 15 <= en; i += 16) {
            __m128i ca  = _mm_loadu_si128((const __m128i *)(ap  + i - 1));
            __m128i cb  = _mm_loadu_si128((const __m128i *)(brp + i + off));
            __m256i eq  = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i sv  = _mm256_sub_epi16(vneg1, _mm256_add_epi16(eq, eq));
            __m256i dg  = _mm256_add_epi16(
                              _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), sv);
            __m256i uu  = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i ll  = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i mx  = _mm256_sub_epi16(_mm256_max_epi16(uu, ll), vgap);
            _mm256_storeu_si256((__m256i *)(cur + i), _mm256_max_epi16(mx, dg));
        }
#endif
        for (; i <= en; i++) {
            int s = (ap[i - 1] == brp[i + off]) ? KMATCH : KMISMATCH;
            int v = (int)p2[i - 1] + s;
            int u = (int)p1[i - 1], l = (int)p1[i];
            int w = ((u > l) ? u : l) + KGAP;
            cur[i] = (int16_t)((v > w) ? v : w);
        }
        cur[lo - 1] = SENT; cur[lo - 2] = SENT;
        cur[hi + 1] = SENT; cur[hi + 2] = SENT;
    }
    return (int)B[(2 * n) % 3][n];
}

/* ---- Phase 3b: same wavefront, int32 houses (8 per breath), large n ---- */
static int wave32(int n, const unsigned char *__restrict ap,
                  const unsigned char *__restrict brp, int D,
                  int32_t *blk)
{
    const int32_t SENT = -1000000000;
    const int stride = n + 48;
    int32_t *B0 = blk + 8, *B1 = blk + stride + 8, *B2 = blk + 2 * stride + 8;
    int32_t *B[3];
    int t, k;
#ifdef __AVX2__
    const __m256i vneg1 = _mm256_set1_epi32(-1);
    const __m256i vgap  = _mm256_set1_epi32(2);
#endif
    for (k = 0; k < 3 * stride; k++) blk[k] = SENT;
    B[0] = B0; B[1] = B1; B[2] = B2;
    B[0][0] = 0;

    for (t = 1; t <= 2 * n; t++) {
        int32_t       *__restrict cur = B[t % 3];
        const int32_t *__restrict p1  = B[(t + 2) % 3];
        const int32_t *__restrict p2  = B[(t + 1) % 3];
        int ilo = (t > n) ? (t - n) : 0;
        int ihi = (t < n) ? t : n;
        int blo = (t > D) ? ((t - D + 1) >> 1) : 0;
        int bhi = (t + D) >> 1;
        int lo  = (ilo > blo) ? ilo : blo;
        int hi  = (ihi < bhi) ? ihi : bhi;
        int st = lo, en = hi, off = n - t, i;
        int32_t bv = (int32_t)(KGAP * t);

        if (t <= n) {
            if (lo == 0) { cur[0] = bv; st = 1; }
            if (hi == t) { cur[t] = bv; en = t - 1; }
        }
        i = st;
#ifdef __AVX2__
        for (; i + 7 <= en; i += 8) {
            __m128i ca  = _mm_loadl_epi64((const __m128i *)(ap  + i - 1));
            __m128i cb  = _mm_loadl_epi64((const __m128i *)(brp + i + off));
            __m256i eq  = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
            __m256i sv  = _mm256_sub_epi32(vneg1, _mm256_add_epi32(eq, eq));
            __m256i dg  = _mm256_add_epi32(
                              _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), sv);
            __m256i uu  = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i ll  = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i mx  = _mm256_sub_epi32(_mm256_max_epi32(uu, ll), vgap);
            _mm256_storeu_si256((__m256i *)(cur + i), _mm256_max_epi32(mx, dg));
        }
#endif
        for (; i <= en; i++) {
            int s = (ap[i - 1] == brp[i + off]) ? KMATCH : KMISMATCH;
            int v = (int)p2[i - 1] + s;
            int u = (int)p1[i - 1], l = (int)p1[i];
            int w = ((u > l) ? u : l) + KGAP;
            cur[i] = (int32_t)((v > w) ? v : w);
        }
        cur[lo - 1] = SENT; cur[lo - 2] = SENT;
        cur[hi + 1] = SENT; cur[hi + 2] = SENT;
    }
    return (int)B[(2 * n) % 3][n];
}

int kernel(int n, const char *a, const char *b)
{
    int S0, Sbest, D, W, res, i;
    unsigned char *chars, *ap, *brp;

    if (n <= 0) return 0;

    /* Phase 1 + 2: the kept fist, and the corridor it certifies. */
    Sbest = slip_certificate(n, a, b, &S0);
    D = (n - Sbest) / 5;
    if (D <= 0) return S0;            /* no slip anywhere can beat the road */
    if (D > n) D = n;                 /* guard: worst case = full wavefront */

    if (n <= SMALL_N) return small_dp(n, a, b);   /* guard: tiny inputs */

    chars = (unsigned char *)malloc((size_t)2 * (n + 64));
    if (!chars) return small_dp(n < SMALL_N ? n : SMALL_N, a, b); /* unreachable */
    ap  = chars;
    brp = chars + (n + 64);
    memcpy(ap, a, (size_t)n);
    memset(ap + n, 'X', 64);
    for (i = 0; i < n; i++) brp[i] = (unsigned char)b[n - 1 - i];  /* read back */
    memset(brp + n, 'Y', 64);

    W = D;
    if (n <= 6000) {                  /* int16 houses: |dp| <= 4n < 24000 */
        int16_t *blk = (int16_t *)malloc((size_t)3 * (n + 48) * sizeof(int16_t));
        if (!blk) { free(chars); return small_dp(SMALL_N, a, b); }
        if (n >= 256 && D > 32) {     /* one narrow trial first: tighten D */
            int s = wave16(n, ap, brp, 32, blk);
            int Dn = (n - s) / 5;
            if (Dn <= 32) { free(blk); free(chars); return s; }  /* proven */
            if (Dn < W) W = Dn;
        }
        res = wave16(n, ap, brp, W, blk);
        free(blk);
    } else {
        int32_t *blk = (int32_t *)malloc((size_t)3 * (n + 48) * sizeof(int32_t));
        if (!blk) { free(chars); return small_dp(SMALL_N, a, b); }
        if (D > 32) {
            int s = wave32(n, ap, brp, 32, blk);
            int Dn = (n - s) / 5;
            if (Dn <= 32) { free(blk); free(chars); return s; }
            if (Dn < W) W = Dn;
        }
        res = wave32(n, ap, brp, W, blk);
        free(blk);
    }
    free(chars);
    return res;
}
