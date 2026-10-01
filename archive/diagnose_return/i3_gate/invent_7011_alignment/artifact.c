#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX512BW__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH      1
#define MISMATCH (-1)
#define GAP      (-2)

/* regime gates: below WF_MIN_N the lattice costs more than it saves;
   above WF_MAX_N a 16-bit spark would burn through its cell (-2n < -32768). */
#define WF_MIN_N   64
#define WF_MAX_N   16000
#define WF_PAD     64          /* slack so a vector may overrun a crease safely */

/* ---- ground-walker: exact two-row Needleman-Wunsch, O(n) memory ---- */
static int nw_rows(int n, const char *restrict a, const char *restrict b)
{
    if (n <= 0) return 0;
    int *restrict row = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!row) return 0;
    for (int j = 0; j <= n; ++j) row[j] = j * GAP;
    for (int i = 1; i <= n; ++i) {
        int diag = row[0];
        row[0] = i * GAP;
        int left = row[0];
        const char ai = a[i - 1];
        for (int j = 1; j <= n; ++j) {
            int up   = row[j];
            int best = diag + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int g    = ((up > left) ? up : left) + GAP;
            if (g > best) best = g;
            row[j] = best;
            diag   = up;
            left   = best;
        }
    }
    int r = row[n];
    free(row);
    return r;
}

/* ---- the folded sheet: anti-diagonal wavefront, three creases live ---- */
int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < WF_MIN_N || n > WF_MAX_N) return nw_rows(n, a, b);

    const size_t clen = (size_t)n + WF_PAD;            /* padded symbol boughs */
    const size_t dlen = (size_t)n + 1 + WF_PAD;        /* one crease           */
    const size_t ibytes = 3 * dlen * sizeof(int16_t);

    unsigned char *mem = (unsigned char *)malloc(ibytes + 2 * clen + 64);
    if (!mem) return nw_rows(n, a, b);

    int16_t *buf = (int16_t *)mem;
    char *ap = (char *)(mem + ibytes);
    char *br = ap + clen;

    memset(buf, 0, ibytes);
    memcpy(ap, a, (size_t)n);
    memset(ap + n, 0, WF_PAD);
    for (int k = 0; k < n; ++k) br[k] = b[n - 1 - k];  /* bough hung from the far end */
    memset(br + n, 1, WF_PAD);                          /* padding never matches ap's  */

    int16_t *p2 = buf;                 /* crease d-2 */
    int16_t *p1 = buf + dlen;          /* crease d-1 */
    int16_t *cu = buf + 2 * dlen;      /* crease d   */

    p2[0] = 0;                         /* d = 0 : dp[0][0]            */
    p1[0] = (int16_t)GAP;              /* d = 1 : dp[0][1]            */
    p1[1] = (int16_t)GAP;              /*         dp[1][0]            */

#if defined(__AVX512BW__)
    const __m512i vg  = _mm512_set1_epi16((short)GAP);
    const __m512i vm1 = _mm512_set1_epi16((short)-1);
#elif defined(__AVX2__)
    const __m256i vg  = _mm256_set1_epi16((short)GAP);
    const __m256i vm1 = _mm256_set1_epi16((short)-1);
#endif

    for (int d = 2; d <= 2 * n; ++d) {
        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        const int len = hi - lo + 1;
        const char *pa = ap + (lo - 1);
        const char *pb = br + (lo + n - d);

#if defined(__AVX512BW__)
        for (int k = 0; k < len; k += 32) {
            __m256i ca = _mm256_loadu_si256((const __m256i *)(pa + k));
            __m256i cb = _mm256_loadu_si256((const __m256i *)(pb + k));
            __m512i eq = _mm512_cvtepi8_epi16(_mm256_cmpeq_epi8(ca, cb));
            __m512i s  = _mm512_sub_epi16(vm1, _mm512_add_epi16(eq, eq)); /* +1 / -1 */
            __m512i x  = _mm512_add_epi16(
                             _mm512_loadu_si512((const void *)(p2 + lo + k - 1)), s);
            __m512i g  = _mm512_add_epi16(_mm512_max_epi16(
                             _mm512_loadu_si512((const void *)(p1 + lo + k - 1)),
                             _mm512_loadu_si512((const void *)(p1 + lo + k))), vg);
            _mm512_storeu_si512((void *)(cu + lo + k), _mm512_max_epi16(x, g));
        }
#elif defined(__AVX2__)
        for (int k = 0; k < len; k += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(pa + k));
            __m128i cb = _mm_loadu_si128((const __m128i *)(pb + k));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i s  = _mm256_sub_epi16(vm1, _mm256_add_epi16(eq, eq)); /* +1 / -1 */
            __m256i x  = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(p2 + lo + k - 1)), s);
            __m256i g  = _mm256_add_epi16(_mm256_max_epi16(
                             _mm256_loadu_si256((const __m256i *)(p1 + lo + k - 1)),
                             _mm256_loadu_si256((const __m256i *)(p1 + lo + k))), vg);
            _mm256_storeu_si256((__m256i *)(cu + lo + k), _mm256_max_epi16(x, g));
        }
#else
        {
            int16_t       *restrict cc  = cu + lo;
            const int16_t *restrict q2  = p2 + lo - 1;
            const int16_t *restrict q1a = p1 + lo - 1;
            const int16_t *restrict q1b = p1 + lo;
            for (int k = 0; k < len; ++k) {
                int s = (pa[k] == pb[k]) ? MATCH : MISMATCH;
                int x = (int)q2[k] + s;
                int u = (int)q1a[k], v = (int)q1b[k];
                int g = ((u > v) ? u : v) + GAP;
                cc[k] = (int16_t)((x > g) ? x : g);
            }
        }
#endif
        if (d <= n) {                       /* the two lamp-posts of this crease */
            int16_t e = (int16_t)(-2 * d);
            cu[0] = e;                      /* dp[0][d] */
            cu[d] = e;                      /* dp[d][0] */
        }
        { int16_t *t = p2; p2 = p1; p1 = cu; cu = t; }   /* bottom layer drifts off */
    }

    int res = (int)p1[n];
    free(mem);
    return res;
}
