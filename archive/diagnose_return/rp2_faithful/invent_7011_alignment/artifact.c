#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

#define AL_MATCH      1
#define AL_MISMATCH (-1)
#define AL_GAP      (-2)
#define AL_SMALL     32      /* lattice too small to be worth hanging lamp-posts for */
#define AL_I16MAX 16000      /* a 16-bit spark holds the answer only while 2n <= 32767 */

/* ---------- the flat walker: plain NW, two rolling rows (fallback regime) ---------- */
static int al_rows(int n, const unsigned char *a, const unsigned char *b)
{
    int *buf, *prev, *cur, r;
    if (n <= 0) return 0;
    buf = (int *)malloc((size_t)2 * (size_t)(n + 1) * sizeof(int));
    if (!buf) return 0;
    prev = buf; cur = buf + (n + 1);
    for (int j = 0; j <= n; ++j) prev[j] = AL_GAP * j;
    for (int i = 1; i <= n; ++i) {
        unsigned char ai = a[i - 1];
        int *t;
        cur[0] = AL_GAP * i;
        for (int j = 1; j <= n; ++j) {
            int dg = prev[j - 1] + (ai == b[j - 1] ? AL_MATCH : AL_MISMATCH);
            int up = prev[j] + AL_GAP;
            int lf = cur[j - 1] + AL_GAP;
            int bst = dg;
            if (up > bst) bst = up;
            if (lf > bst) bst = lf;
            cur[j] = bst;
        }
        t = prev; prev = cur; cur = t;
    }
    r = prev[n];
    free(buf);
    return r;
}

/* ---------- one spark, read through the three layers of the crease ---------- */
static inline int al_spark(const short *p1, const short *p2,
                           const unsigned char *a, const unsigned char *br,
                           int i, int off)
{
    int dg = (int)p2[i - 1] + ((a[i - 1] == br[i + off]) ? AL_MATCH : AL_MISMATCH);
    int g1 = (int)p1[i - 1], g2 = (int)p1[i];
    int g  = ((g1 > g2) ? g1 : g2) + AL_GAP;
    return (dg > g) ? dg : g;
}

#if defined(__AVX2__)
/* 16 cells of one crease pressed flush into one register */
static inline void al_crease16(short *cu, const short *p1, const short *p2,
                               const unsigned char *a, const unsigned char *br,
                               int i, int off)
{
    __m256i a16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(a + i - 1)));
    __m256i b16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(br + i + off)));
    __m256i eq  = _mm256_cmpeq_epi16(a16, b16);                      /* the symbols pick the colour */
    __m256i sub = _mm256_blendv_epi8(_mm256_set1_epi16(AL_MISMATCH),
                                     _mm256_set1_epi16(AL_MATCH), eq);
    __m256i gp  = _mm256_add_epi16(                                   /* layer beside: one slip */
                    _mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(p1 + i - 1)),
                                     _mm256_loadu_si256((const __m256i *)(p1 + i))),
                    _mm256_set1_epi16(AL_GAP));
    __m256i dg  = _mm256_add_epi16(                                   /* layer beneath: straight through */
                    _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), sub);
    _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi16(gp, dg));
}
#endif

#if defined(__AVX512BW__)
static inline void al_crease32(short *cu, const short *p1, const short *p2,
                               const unsigned char *a, const unsigned char *br,
                               int i, int off)
{
    __m512i a16 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(a + i - 1)));
    __m512i b16 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(br + i + off)));
    __mmask32 eq = _mm512_cmpeq_epi16_mask(a16, b16);
    __m512i sub = _mm512_mask_blend_epi16(eq, _mm512_set1_epi16(AL_MISMATCH),
                                              _mm512_set1_epi16(AL_MATCH));
    __m512i gp  = _mm512_add_epi16(
                    _mm512_max_epi16(_mm512_loadu_si512((const void *)(p1 + i - 1)),
                                     _mm512_loadu_si512((const void *)(p1 + i))),
                    _mm512_set1_epi16(AL_GAP));
    __m512i dg  = _mm512_add_epi16(_mm512_loadu_si512((const void *)(p2 + i - 1)), sub);
    _mm512_storeu_si512((void *)(cu + i), _mm512_max_epi16(gp, dg));
}
#endif

/* ---------- the fold, walked forward with 16-bit sparks ---------- */
static int al_diag_i16(int n, const unsigned char *a, const unsigned char *b)
{
    size_t lane = (size_t)n + 1 + 64;
    void *blk;
    short *L0, *L1, *L2, *p2, *p1, *cu;
    unsigned char *br;
    int res;

    blk = malloc(3 * lane * sizeof(short) + (size_t)n + 64);
    if (!blk) return al_rows(n, a, b);
    memset(blk, 0, 3 * lane * sizeof(short));
    L0 = (short *)blk; L1 = L0 + lane; L2 = L1 + lane;
    br = (unsigned char *)(L2 + lane);
    for (int t = 0; t < n; ++t) br[t] = b[n - 1 - t];   /* one bough hangs the other way */
    memset(br + n, 0xFF, 64);

    p2 = L0; p1 = L1; cu = L2;
    p2[0] = 0;                                          /* crease 0 */
    p1[0] = (short)AL_GAP; p1[1] = (short)AL_GAP;        /* crease 1: the two lamp-posts */

    for (int d = 2; d <= 2 * n; ++d) {
        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        int off = n - d;
        int cnt = hi - lo + 1;
        int i = lo;
        short *t;
#if defined(__AVX512BW__)
        if (cnt >= 32) {
            for (; i + 32 <= hi + 1; i += 32) al_crease32(cu, p1, p2, a, br, i, off);
            if (i <= hi) { al_crease32(cu, p1, p2, a, br, hi - 31, off); i = hi + 1; }
        }
#endif
#if defined(__AVX2__)
        if (cnt >= 16) {
            for (; i + 16 <= hi + 1; i += 16) al_crease16(cu, p1, p2, a, br, i, off);
            if (i <= hi) { al_crease16(cu, p1, p2, a, br, hi - 15, off); i = hi + 1; }
        }
#endif
        for (; i <= hi; ++i) cu[i] = (short)al_spark(p1, p2, a, br, i, off);
        if (d <= n) { short g = (short)(AL_GAP * d); cu[0] = g; cu[d] = g; }
        t = p2; p2 = p1; p1 = cu; cu = t;               /* the spent layer drifts off */
    }
    res = (int)p1[n];
    free(blk);
    return res;
}

/* ---------- the same fold with 32-bit sparks, for lattices past the 16-bit lamp ---------- */
static int al_diag_i32(int n, const unsigned char *a, const unsigned char *b)
{
    size_t lane = (size_t)n + 1 + 64;
    int *mem = (int *)calloc(3 * lane, sizeof(int));
    unsigned char *br = (unsigned char *)malloc((size_t)n + 64);
    int *p2, *p1, *cu, res;
    if (!mem || !br) { free(mem); free(br); return al_rows(n, a, b); }
    for (int t = 0; t < n; ++t) br[t] = b[n - 1 - t];
    memset(br + n, 0xFF, 64);
    p2 = mem; p1 = mem + lane; cu = mem + 2 * lane;
    p2[0] = 0; p1[0] = AL_GAP; p1[1] = AL_GAP;
    for (int d = 2; d <= 2 * n; ++d) {
        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        int off = n - d;
        const int *__restrict q1 = p1;
        const int *__restrict q2 = p2;
        int *__restrict w = cu;
        int *t;
        #pragma omp simd
        for (int i = lo; i <= hi; ++i) {
            int g1 = q1[i - 1], g2 = q1[i];
            int g  = ((g1 > g2) ? g1 : g2) + AL_GAP;
            int dg = q2[i - 1] + ((a[i - 1] == br[i + off]) ? AL_MATCH : AL_MISMATCH);
            w[i] = (dg > g) ? dg : g;
        }
        if (d <= n) { cu[0] = AL_GAP * d; cu[d] = AL_GAP * d; }
        t = p2; p2 = p1; p1 = cu; cu = t;
    }
    res = p1[n];
    free(mem); free(br);
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    const unsigned char *ua = (const unsigned char *)a;
    const unsigned char *ub = (const unsigned char *)b;
    if (n <= 0) return 0;
    if (n < AL_SMALL)    return al_rows(n, ua, ub);      /* regime: too small to fold */
    if (n > AL_I16MAX)   return al_diag_i32(n, ua, ub);  /* regime: spark too bright for 16 bits */
    return al_diag_i16(n, ua, ub);                       /* the fold */
}
