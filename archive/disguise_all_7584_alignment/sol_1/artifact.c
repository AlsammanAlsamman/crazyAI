#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* O(n)-memory exact Needleman-Wunsch; used for tiny n, huge n, or no AVX2. */
static int nw_scalar(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int r = 0;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = -2 * j;
    for (int i = 1; i <= n; i++) {
        const char ai = a[i - 1];
        cur[0] = -2 * i;
        for (int j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? 1 : -1);
            int u = prev[j] - 2;
            int l = cur[j - 1] - 2;
            int m = d > u ? d : u;
            if (l > m) m = l;
            cur[j] = m;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if defined(__AVX2__)
    if (n < 64 || n > 16000) return nw_scalar(n, a, b);
    {
        size_t bs = (((size_t)n + 96) + 31) & ~(size_t)31;   /* bytes, mult of 32  */
        size_t ss = (((size_t)n + 96) + 15) & ~(size_t)15;   /* shorts, mult of 16 */
        unsigned char *raw = (unsigned char *)malloc(2 * bs + 6 * ss + 64);
        unsigned char *base;
        char *pa, *pbr;
        short *d2, *d1, *d0, *p2, *p1, *cu;
        int d, res;
        const __m256i vone = _mm256_set1_epi16(1);
        const __m256i vtwo = _mm256_set1_epi16(2);

        if (!raw) return nw_scalar(n, a, b);
        base = (unsigned char *)(((uintptr_t)raw + 31) & ~(uintptr_t)31);
        pa  = (char *)base;
        pbr = (char *)(base + bs);
        d2  = (short *)(base + 2 * bs);
        d1  = (short *)(base + 2 * bs + 2 * ss);
        d0  = (short *)(base + 2 * bs + 4 * ss);

        memcpy(pa, a, (size_t)n);
        memset(pa + n, 0, bs - (size_t)n);
        for (int k = 0; k < n; k++) pbr[k] = b[n - 1 - k];
        memset(pbr + n, 1, bs - (size_t)n);
        memset(d2, 0, 2 * ss);
        memset(d1, 0, 2 * ss);
        memset(d0, 0, 2 * ss);

        p2 = d2; p1 = d1; cu = d0;
        p2[0] = 0;                       /* diagonal 0: H[0][0]            */
        p1[0] = -2; p1[1] = -2;          /* diagonal 1: H[0][1], H[1][0]   */

        for (d = 2; d <= 2 * n; d++) {
            int ilo = d - n; if (ilo < 1) ilo = 1;
            int ihi = d - 1; if (ihi > n) ihi = n;
            {
                const char  *ap = pa  - 1;          /* index by i -> a[i-1]      */
                const char  *bp = pbr + (n - d);    /* index by i -> b[d-i-1]    */
                const short *q2 = p2  - 1;          /* index by i -> H[i-1][j-1] */
                const short *qu = p1  - 1;          /* index by i -> H[i-1][j]   */
                const short *ql = p1;               /* index by i -> H[i][j-1]   */
                int i;
                for (i = ilo; i <= ihi; i += 32) {
                    __m256i m0 = _mm256_cvtepi8_epi16(
                        _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(ap + i)),
                                       _mm_loadu_si128((const __m128i *)(bp + i))));
                    __m256i m1 = _mm256_cvtepi8_epi16(
                        _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(ap + i + 16)),
                                       _mm_loadu_si128((const __m128i *)(bp + i + 16))));
                    __m256i g0 = _mm256_loadu_si256((const __m256i *)(q2 + i));
                    __m256i g1 = _mm256_loadu_si256((const __m256i *)(q2 + i + 16));
                    __m256i u0 = _mm256_loadu_si256((const __m256i *)(qu + i));
                    __m256i u1 = _mm256_loadu_si256((const __m256i *)(qu + i + 16));
                    __m256i l0 = _mm256_loadu_si256((const __m256i *)(ql + i));
                    __m256i l1 = _mm256_loadu_si256((const __m256i *)(ql + i + 16));
                    /* diag + (match? +1 : -1)  ==  diag - 2*mask - 1 */
                    __m256i s0 = _mm256_sub_epi16(
                                     _mm256_sub_epi16(g0, _mm256_add_epi16(m0, m0)), vone);
                    __m256i s1 = _mm256_sub_epi16(
                                     _mm256_sub_epi16(g1, _mm256_add_epi16(m1, m1)), vone);
                    __m256i h0 = _mm256_sub_epi16(_mm256_max_epi16(u0, l0), vtwo);
                    __m256i h1 = _mm256_sub_epi16(_mm256_max_epi16(u1, l1), vtwo);
                    _mm256_storeu_si256((__m256i *)(cu + i),      _mm256_max_epi16(s0, h0));
                    _mm256_storeu_si256((__m256i *)(cu + i + 16), _mm256_max_epi16(s1, h1));
                }
            }
            if (d <= n) {                    /* grid edges: H[0][d] = H[d][0] = -2d */
                cu[0] = (short)(-2 * d);
                cu[d] = (short)(-2 * d);
            }
            { short *t = p2; p2 = p1; p1 = cu; cu = t; }
        }

        res = p1[n];                         /* H[n][n] on diagonal 2n */
        free(raw);
        return res;
    }
#else
    return nw_scalar(n, a, b);
#endif
}
