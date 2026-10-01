#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2
#define NEGINF   (-(1 << 28))   /* deeper than any real mound (>= -4n), safe under repeated +GAP */

/* ---- plain exact two-row DP: tiny trays, and the wide-band no-AVX2 regime ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    int *prev, *cur, *tmp, i, j, r;
    prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (j = 0; j <= n; j++) prev[j] = j * GAP;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * GAP;
        for (j = 1; j <= n; j++) {
            int s  = (ai == b[j - 1]) ? MATCH : MISMATCH;
            int t1 = prev[j - 1] + s;
            int t2 = (prev[j] > cur[j - 1] ? prev[j] : cur[j - 1]) + GAP;
            cur[j] = t1 > t2 ? t1 : t2;
        }
        tmp = prev; prev = cur; cur = tmp;
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---- the ink-worm: one body-wide front along d = i+j, confined to |i-j| <= k ----
 * cur[i]  = dp[i][d-i]
 *         = max( p2[i-1] + s(a[i-1], b[d-i-1]),          (straight along the grain)
 *                max(p1[i-1], p1[i]) + GAP )             (the free sideways slip)
 * p1 = diagonal d-1, p2 = diagonal d-2. Nothing is kept but the sand itself.
 */
static int nw_band(int n, const char *a, const char *b, int k)
{
    const int PAD = 40;
    size_t stride = (size_t)n + 2 + 2 * (size_t)PAD;
    int *mem   = (int *)malloc(3 * stride * sizeof(int));
    char *cbuf = (char *)malloc(2 * ((size_t)n + 2 * (size_t)PAD));
    int *X, *Y, *Z, *cur, *p1, *p2;
    char *A, *R;
    int d, i, res;
    size_t t;

    if (!mem || !cbuf) { free(mem); free(cbuf); return nw_rows(n, a, b); }

    for (t = 0; t < 3 * stride; t++) mem[t] = NEGINF;
    X = mem + PAD;                    /* usable indices -PAD .. n+1+PAD */
    Y = mem + stride + PAD;
    Z = mem + 2 * stride + PAD;

    /* the two cords: north cord forward, east cord reversed, both padded with
       characters that can never agree with anything real */
    A = cbuf;
    R = cbuf + (size_t)n + 2 * (size_t)PAD;
    memcpy(A, a, (size_t)n);
    memset(A + n, 0x01, 2 * (size_t)PAD);
    for (i = 0; i < n; i++) R[i] = b[n - 1 - i];
    memset(R + n, 0x02, 2 * (size_t)PAD);

    Y[0] = 0;                         /* diagonal d = 0, the near corner */
    cur = X; p1 = Y; p2 = Z;          /* Z stands for the empty diagonal d = -1 */

    for (d = 1; d <= 2 * n; d++) {
        int lo = d - n, hi = d, lb, ub, ilo, ihi, q;
        if (lo < 0) lo = 0;
        lb = (d - k + 1) >> 1;        /* ceil((d-k)/2) */
        if (lb > lo) lo = lb;
        if (hi > n) hi = n;
        ub = (d + k) >> 1;            /* floor((d+k)/2) */
        if (ub < hi) hi = ub;

        ilo = lo > 1 ? lo : 1;
        ihi = hi < d - 1 ? hi : d - 1;
        q   = n - d;                  /* b[d-i-1] == R[q+i], ascending in i */
        i   = ilo;

#if defined(__AVX2__)
        {
            const __m256i vgap = _mm256_set1_epi32(GAP);
            const __m256i vpos = _mm256_set1_epi32(MATCH);
            const __m256i vneg = _mm256_set1_epi32(MISMATCH);
            for (; i + 7 <= ihi; i += 8) {
                /* SEED 1: the cords crossing marks eight furrow-points at once */
                __m128i av = _mm_loadl_epi64((const __m128i *)(A + i - 1));
                __m128i bv = _mm_loadl_epi64((const __m128i *)(R + q + i));
                __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(av, bv));
                __m256i s  = _mm256_blendv_epi8(vneg, vpos, eq);
                /* SEED 3: the sideways slip is a one-lane offset, spending nothing */
                __m256i dg = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i up = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i lf = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m256i best = _mm256_max_epi32(
                        _mm256_add_epi32(dg, s),
                        _mm256_add_epi32(_mm256_max_epi32(up, lf), vgap));
                _mm256_storeu_si256((__m256i *)(cur + i), best);
            }
        }
#endif
        for (; i <= ihi; i++) {
            int s  = (A[i - 1] == R[q + i]) ? MATCH : MISMATCH;
            int t1 = p2[i - 1] + s;
            int t2 = (p1[i - 1] > p1[i] ? p1[i - 1] : p1[i]) + GAP;
            cur[i] = t1 > t2 ? t1 : t2;
        }

        /* the two tray edges, where a cord has run out entirely */
        if (lo == 0) cur[0] = GAP * d;
        if (hi == d) cur[d] = GAP * d;

        /* fence the furrow: one junction beyond each rim holds no mound at all */
        cur[lo - 1] = NEGINF;
        cur[hi + 1] = NEGINF;

        { int *sw = p2; p2 = p1; p1 = cur; cur = sw; }
    }

    res = p1[n];                      /* far corner: dp[n][n], diagonal d = 2n */
    free(mem); free(cbuf);
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    int i, D, k;
    long long kk;

    if (n <= 0) return 0;
    if (n < 64) return nw_rows(n, a, b);          /* guard: tiny tray, no worm */

    /* walk the straight grain once and count the sand it costs: this is the
       runtime regime test.  D small -> narrow furrow; D large -> whole tray. */
    D = 0;
    for (i = 0; i < n; i++) D += (a[i] != b[i]);

    /* any optimal path uses g gaps per cord with n-5g >= n-2D, so g <= 2D/5,
       and |i-j| <= g.  Half-width 2D/5 is provably sufficient; +2 for slack. */
    kk = (2LL * (long long)D) / 5 + 2;
    k  = (kk > (long long)n) ? n : (int)kk;       /* k == n -> band inactive */
    if (k < 2) k = 2;

#if !defined(__AVX2__)
    /* no vector lanes and a wide furrow: the row DP has better locality */
    if (2 * k >= n) return nw_rows(n, a, b);
#endif
    return nw_band(n, a, b, k);
}
