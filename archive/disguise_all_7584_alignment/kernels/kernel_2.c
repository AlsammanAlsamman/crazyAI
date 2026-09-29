#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define NEG16V (-29000)
#define NEG32V (-(1 << 28))

/* Banded global NW over the anti-diagonals.
   A[t][i] = dp[i][t-i], stored at physical index i+1 (slot 0 / hi+2 are -INF guards).
   Band: |i - j| <= K.  Returns the best in-band alignment score (<= true optimum,
   == true optimum whenever K >= max gap-pair count of some optimal path). */
static int nw_band_i16(int n, const char *a, const char *brev, int K, short *scratch)
{
    const int stride = n + 3;
    short *d2 = scratch, *d1 = scratch + stride, *d0 = scratch + 2 * stride, *tp;
    const int T = 2 * n;
    int res = 0, z;

    for (z = 0; z < 3 * stride; z++) scratch[z] = (short)NEG16V;

    for (int t = 0;; t++) {
        int lo = (t - K + 1) >> 1;      /* ceil((t-K)/2) */
        int hi = (t + K) >> 1;          /* floor((t+K)/2) */
        if (lo < 0) lo = 0;
        if (lo < t - n) lo = t - n;
        if (hi > n) hi = n;
        if (hi > t) hi = t;

        d0[lo] = (short)NEG16V;         /* guard for logical lo-1 */
        d0[hi + 2] = (short)NEG16V;     /* guard for logical hi+1 */

        int vlo = lo, vhi = hi;
        if (lo == 0) { d0[1] = (short)(-2 * t); vlo = 1; }          /* dp[0][t]  */
        if (hi == t) { d0[t + 1] = (short)(-2 * t); vhi = t - 1; }  /* dp[t][0]  */

        int i = vlo;
#if defined(__AVX2__)
        {
            const __m256i vm1 = _mm256_set1_epi16(-1);
            const __m256i vg  = _mm256_set1_epi16(2);
            const int base = n - t;
            for (; i + 15 <= vhi; i += 16) {
                __m128i av = _mm_loadu_si128((const __m128i *)(a + i - 1));
                __m128i bv = _mm_loadu_si128((const __m128i *)(brev + base + i));
                __m256i mk = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(av, bv)); /* -1 / 0 */
                __m256i sc = _mm256_sub_epi16(vm1, _mm256_add_epi16(mk, mk)); /* +1/-1 */
                __m256i dg = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(d2 + i)), sc);
                __m256i u  = _mm256_loadu_si256((const __m256i *)(d1 + i));
                __m256i l  = _mm256_loadu_si256((const __m256i *)(d1 + i + 1));
                __m256i g  = _mm256_sub_epi16(_mm256_max_epi16(u, l), vg);
                _mm256_storeu_si256((__m256i *)(d0 + i + 1),
                                    _mm256_max_epi16(dg, g));
            }
        }
#endif
        {
            const short *__restrict p2 = d2;
            const short *__restrict p1 = d1;
            short       *__restrict p0 = d0;
            const char  *__restrict pa = a;
            const char  *__restrict pb = brev + (n - t);
            for (; i <= vhi; i++) {
                int s = (pa[i - 1] == pb[i]) ? 1 : -1;
                int best = p2[i] + s;
                int u = p1[i] - 2;
                int l = p1[i + 1] - 2;
                if (u > best) best = u;
                if (l > best) best = l;
                p0[i + 1] = (short)best;
            }
        }
        if (t == T) { res = d0[n + 1]; break; }
        tp = d2; d2 = d1; d1 = d0; d0 = tp;
    }
    return res;
}

static int nw_band_i32(int n, const char *a, const char *brev, int K, int *scratch)
{
    const int stride = n + 3;
    int *d2 = scratch, *d1 = scratch + stride, *d0 = scratch + 2 * stride, *tp;
    const int T = 2 * n;
    int res = 0, z;

    for (z = 0; z < 3 * stride; z++) scratch[z] = NEG32V;

    for (int t = 0;; t++) {
        int lo = (t - K + 1) >> 1;
        int hi = (t + K) >> 1;
        if (lo < 0) lo = 0;
        if (lo < t - n) lo = t - n;
        if (hi > n) hi = n;
        if (hi > t) hi = t;

        d0[lo] = NEG32V;
        d0[hi + 2] = NEG32V;

        int vlo = lo, vhi = hi;
        if (lo == 0) { d0[1] = -2 * t; vlo = 1; }
        if (hi == t) { d0[t + 1] = -2 * t; vhi = t - 1; }

        int i = vlo;
#if defined(__AVX2__)
        {
            const __m256i vm1 = _mm256_set1_epi32(-1);
            const __m256i vg  = _mm256_set1_epi32(2);
            const int base = n - t;
            for (; i + 7 <= vhi; i += 8) {
                __m128i av = _mm_loadl_epi64((const __m128i *)(a + i - 1));
                __m128i bv = _mm_loadl_epi64((const __m128i *)(brev + base + i));
                __m256i mk = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(av, bv));
                __m256i sc = _mm256_sub_epi32(vm1, _mm256_add_epi32(mk, mk));
                __m256i dg = _mm256_add_epi32(
                                 _mm256_loadu_si256((const __m256i *)(d2 + i)), sc);
                __m256i u  = _mm256_loadu_si256((const __m256i *)(d1 + i));
                __m256i l  = _mm256_loadu_si256((const __m256i *)(d1 + i + 1));
                __m256i g  = _mm256_sub_epi32(_mm256_max_epi32(u, l), vg);
                _mm256_storeu_si256((__m256i *)(d0 + i + 1),
                                    _mm256_max_epi32(dg, g));
            }
        }
#endif
        {
            const int  *__restrict p2 = d2;
            const int  *__restrict p1 = d1;
            int        *__restrict p0 = d0;
            const char *__restrict pa = a;
            const char *__restrict pb = brev + (n - t);
            for (; i <= vhi; i++) {
                int s = (pa[i - 1] == pb[i]) ? 1 : -1;
                int best = p2[i] + s;
                int u = p1[i] - 2;
                int l = p1[i + 1] - 2;
                if (u > best) best = u;
                if (l > best) best = l;
                p0[i + 1] = best;
            }
        }
        if (t == T) { res = d0[n + 1]; break; }
        tp = d2; d2 = d1; d1 = d0; d0 = tp;
    }
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    /* (1) the straight walk: ungapped diagonal score, O(n), auto-vectorized. */
    int matches = 0;
    for (int i = 0; i < n; i++) matches += (a[i] == b[i]);
    int best = 2 * matches - n;

    /* (2) accounting identity  score = n - 5k - 2MM  =>  k <= (n - best)/5,
           and |i-j| <= k, so this is a provably sufficient band radius.     */
    int Kcap = (n - best) / 5;
    if (Kcap <= 0) return best;              /* no gap can ever pay: exact  */
    if (Kcap > n) Kcap = n;

    char *brev = (char *)malloc((size_t)n + 64);
    if (!brev) return best;
    for (int i = 0; i < n; i++) brev[i] = b[n - 1 - i];
    memset(brev + n, 0, 64);

    const int use16 = (n <= 15000);          /* |scores| <= 1.8n < 29000 */
    size_t cells = 3 * ((size_t)n + 3);
    void *scr = malloc(cells * (use16 ? sizeof(short) : sizeof(int)) + 64);
    if (!scr) { free(brev); return best; }

    /* (3) probe with a narrow band, let the score it finds shrink the band. */
    int K = Kcap < 16 ? Kcap : 16;
    for (;;) {
        int s = use16 ? nw_band_i16(n, a, brev, K, (short *)scr)
                      : nw_band_i32(n, a, brev, K, (int *)scr);
        if (s > best) best = s;
        int need = (n - best) / 5;           /* best <= optimum => need >= k_opt */
        if (K >= need) break;                /* band provably contained an optimum */
        long long nk = (long long)K * 16;
        if (nk > need) nk = need;
        K = (int)nk;
    }

    free(scr);
    free(brev);
    return best;
}
