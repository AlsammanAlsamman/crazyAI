/* ================================================================
 * Needleman-Wunsch global alignment score of two equal-length
 * sequences.  match = +1, mismatch = -1, gap = -2.  EXACT.
 *
 *   int kernel(int n, const char *a, const char *b);
 *
 * Engine (the native's mechanism, literally):
 *   Stage A -- "the horse stands in every doorway once, and once
 *   nowhere at all": evaluate the ungapped diagonal and every
 *   single-slip alignment, in O(n), with no loop-carried dependence.
 *   Every trial is a real NW alignment, so the kept fist is a
 *   certified lower bound L <= OPT.
 *
 *   Stage B -- gap accounting: any alignment with g gaps per string
 *   scores <= n - 5g and never leaves |i-j| <= g.  With
 *   W = floor((n-L)/5) we have 5(W+1) > n-L, so every alignment with
 *   g > W scores < L and is flung to the fish.  The band of
 *   half-width W is therefore exact.  L >= -n always, so W <= 2n/5:
 *   the band is never wider than ~64% of the table, for ANY input.
 *
 *   Stage C -- the flute: fill the band along anti-diagonals, where
 *   all cells are independent, 16 houses per note (int16 AVX2).
 *   'b' is stored reversed ("nose to nose") so both character
 *   operands are forward contiguous loads.
 *
 * Guards for every risk this mechanism's own verdict names:
 *   - int16 range: real in-band cells are >= -(n + 2W + 8) >= -1.8n-8;
 *     sentinel is -30000.  int16 path only for n <= 12000; an
 *     otherwise identical int32 path covers larger n.
 *   - no AVX2 (-march=native on an older host): scalar banded DP.
 *   - tiny n (vector lanes wasted, 2W+1 < 16): scalar banded DP.
 *   - W == 0: return the native's diagonal answer directly, O(n),
 *     no DP at all.
 *   - allocation failure: degrade to the certified bound, never crash.
 * ================================================================ */

#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------------- scratch (per thread, grow-only) ---------------- */
static __thread unsigned char *nb_raw = 0;
static __thread size_t         nb_cap = 0;

static unsigned char *nb_scratch(size_t need)
{
    if (need > nb_cap) {
        unsigned char *p = (unsigned char *)realloc(nb_raw, need);
        if (!p) return 0;
        nb_raw = p;
        nb_cap = need;
    }
    return nb_raw;
}

#define NB_AL(p) ((void *)(((size_t)(p) + 63u) & ~(size_t)63u))

/* ---------------- scalar banded DP (fallback / tiny n) ----------- */
static int nb_scalar(int n, const char *a, const char *b, int W,
                     int *prev, int *cur)
{
    const int NEG = -(1 << 28);
    int i, j, jhi;

    for (j = 0; j <= n + 1; ++j) { prev[j] = NEG; cur[j] = NEG; }

    jhi = (W < n) ? W : n;                 /* row 0 inside the band   */
    for (j = 0; j <= jhi; ++j) prev[j] = -2 * j;
    if (jhi + 1 <= n + 1) prev[jhi + 1] = NEG;

    for (i = 1; i <= n; ++i) {
        int jlo = i - W, jh = i + W;
        int *t;
        if (jlo < 0) jlo = 0;
        if (jh > n)  jh  = n;
        if (jlo > 0) cur[jlo - 1] = NEG;
        for (j = jlo; j <= jh; ++j) {
            int best;
            if (j == 0) {
                best = -2 * i;
            } else {
                int s = (a[i - 1] == b[j - 1]) ? 1 : -1;
                int v = prev[j - 1] + s;
                int u = prev[j] - 2;
                int l = cur[j - 1] - 2;
                best = v;
                if (u > best) best = u;
                if (l > best) best = l;
            }
            cur[j] = best;
        }
        if (jh + 1 <= n + 1) cur[jh + 1] = NEG;
        t = prev; prev = cur; cur = t;
    }
    return prev[n];
}

/* ---------------- anti-diagonal SIMD band sweep ------------------ */
#if defined(__AVX2__)
/* One body, instantiated at two widths, so the two paths cannot drift. */
#define NB_AVX_DP(FN, TY, SFX, CVT, CHLD, LANES, NEGV)                        \
static int FN(int n, const unsigned char *ca, const unsigned char *cbr,       \
              int W, TY *b0, TY *b1, TY *b2)                                  \
{                                                                             \
    TY *db[3];                                                                \
    const __m256i vone = _mm256_set1_##SFX(1);                                \
    const __m256i vtwo = _mm256_set1_##SFX(2);                                \
    int d, i, span = n + LANES + 4;                                           \
    db[0] = b0; db[1] = b1; db[2] = b2;                                       \
    for (i = -1; i < span; ++i) {                                             \
        db[0][i] = NEGV; db[1][i] = NEGV; db[2][i] = NEGV;                    \
    }                                                                         \
    db[0][0] = 0;                              /* H(0,0) = 0, d = 0 */        \
    for (d = 1; d <= 2 * n; ++d) {                                            \
        TY       *cc = db[d % 3];                                             \
        const TY *p1 = db[(d + 2) % 3];        /* anti-diagonal d-1 */        \
        const TY *p2 = db[(d + 1) % 3];        /* anti-diagonal d-2 */        \
        int lo = d - n, hi = (d < n) ? d : n, t;                              \
        if (lo < 0) lo = 0;                                                   \
        t = (d - W + 1) >> 1; if (t > lo) lo = t;   /* ceil((d-W)/2) */       \
        t = (d + W) >> 1;     if (t < hi) hi = t;   /* floor((d+W)/2) */      \
        {                                                                     \
            const unsigned char *pa = ca  + lo;                               \
            const unsigned char *pb = cbr + (lo + n - d);                     \
            for (i = lo; i <= hi; i += LANES, pa += LANES, pb += LANES) {     \
                __m256i xa = CVT(CHLD((const __m128i *)pa));                  \
                __m256i xb = CVT(CHLD((const __m128i *)pb));                  \
                __m256i eq = _mm256_cmpeq_##SFX(xa, xb);                      \
                __m256i s  = _mm256_sub_##SFX(                                \
                                 _mm256_slli_##SFX(                           \
                                     _mm256_and_si256(eq, vone), 1), vone);   \
                __m256i dg = _mm256_loadu_si256((const __m256i *)(p2 + i-1)); \
                __m256i up = _mm256_loadu_si256((const __m256i *)(p1 + i-1)); \
                __m256i lf = _mm256_loadu_si256((const __m256i *)(p1 + i  )); \
                __m256i v  = _mm256_max_##SFX(                                \
                                 _mm256_add_##SFX(dg, s),                     \
                                 _mm256_sub_##SFX(                            \
                                     _mm256_max_##SFX(up, lf), vtwo));        \
                _mm256_storeu_si256((__m256i *)(cc + i), v);                  \
            }                                                                 \
        }                                                                     \
        cc[lo - 1] = NEGV;                    /* the flying fish */           \
        cc[hi + 1] = NEGV;                                                    \
    }                                                                         \
    return (int)db[(2 * n) % 3][n];                                           \
}

NB_AVX_DP(nb_dp16, short, epi16, _mm256_cvtepi8_epi16, _mm_loadu_si128, 16,
          (short)-30000)
NB_AVX_DP(nb_dp32, int,   epi32, _mm256_cvtepi8_epi32, _mm_loadl_epi64,  8,
          -(1 << 28))
#endif /* __AVX2__ */

/* ============================== kernel ========================== */
int kernel(int n, const char *a, const char *b)
{
    size_t nz, sz_ch, sz_r, sz_dp, need;
    unsigned char *raw, *cz, *ca, *cbr;
    int *R1, *R2;
    void *D0, *D1, *D2;
    int pre, bestpm, S0, L, W, i, k, r;

    if (n <= 0) return 0;

    nz    = (size_t)n;
    sz_ch = nz + 96;                            /* ca, cbr           */
    sz_r  = (nz + 8) * sizeof(int);             /* R1, R2            */
    sz_dp = (nz + 40) * sizeof(int);            /* each DP buffer    */
    need  = 8u * 64u + 2u * sz_ch + 2u * sz_r + 3u * sz_dp;

    raw = nb_scratch(need);

    /* ---------- Stage A: the native's trials, O(n), no recurrence
       across trials.  R1[t] and R2[t] are the two slip directions.  */
    if (!raw) {                                 /* degrade, don't die */
        int m = 0;
        for (i = 0; i < n; ++i) m += (a[i] == b[i]);
        return 2 * m - n;                       /* diagonal only      */
    }

    cz  = (unsigned char *)NB_AL(raw);
    ca  = cz;  cz = (unsigned char *)NB_AL(cz + sz_ch);
    cbr = cz;  cz = (unsigned char *)NB_AL(cz + sz_ch);
    R1  = (int *)cz; cz = (unsigned char *)NB_AL(cz + sz_r);
    R2  = (int *)cz; cz = (unsigned char *)NB_AL(cz + sz_r);
    D0  = cz;  cz = (unsigned char *)NB_AL(cz + sz_dp);
    D1  = cz;  cz = (unsigned char *)NB_AL(cz + sz_dp);
    D2  = cz;

    R1[n] = 0;
    R2[n] = 0;
    for (i = n - 1; i >= 1; --i) {
        R1[i] = R1[i + 1] + (a[i]     == b[i - 1]);  /* slip b forward */
        R2[i] = R2[i + 1] + (a[i - 1] == b[i]    );  /* slip a forward */
    }

    pre    = 0;
    bestpm = -(1 << 28);
    for (k = 0; k < n; ++k) {                   /* horse in house k   */
        int m1 = R1[k + 1], m2 = R2[k + 1];
        int mm = (m1 > m2) ? m1 : m2;
        int v  = pre + mm;
        if (v > bestpm) bestpm = v;
        pre += (a[k] == b[k]);
    }
    S0 = 2 * pre - n;                           /* horse nowhere      */
    L  = S0;
    {   int slip = 2 * bestpm - (n - 1) - 4;    /* two gap columns    */
        if (slip > L) L = slip;
    }

    /* ---------- Stage B: the band the certificate buys ------------ */
    W = (n - L) / 5;                            /* n - L >= 0 always  */
    if (W < 0) W = 0;
    if (W > n) W = n;
    if (W == 0) return S0;                      /* proven: no gaps    */

    /* ---------- Stage C: fill the band ---------------------------- */
#if defined(__AVX2__)
    if (n >= 32) {
        ca[0] = 0x01;                           /* non-ACGT sentinel  */
        memcpy(ca + 1, a, nz);
        memset(ca + 1 + nz, 0x01, sz_ch - 1 - nz);
        for (i = 0; i < n; ++i) cbr[i] = (unsigned char)b[n - 1 - i];
        memset(cbr + nz, 0x02, sz_ch - nz);     /* differs from 0x01  */

        if (n <= 12000)
            r = nb_dp16(n, ca, cbr, W,
                        (short *)D0 + 1, (short *)D1 + 1, (short *)D2 + 1);
        else
            r = nb_dp32(n, ca, cbr, W,
                        (int *)D0 + 1, (int *)D1 + 1, (int *)D2 + 1);
        return (r > L) ? r : L;
    }
#endif
    r = nb_scalar(n, a, b, W, (int *)D0, (int *)D1);
    return (r > L) ? r : L;
}
