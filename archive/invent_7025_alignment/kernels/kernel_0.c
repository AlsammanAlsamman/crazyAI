#include <stdlib.h>
#include <immintrin.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

static int scalar_nw(int n, const char *a, const char *b) {
    int *dp = (int *)malloc((size_t)(n + 1) * (n + 1) * sizeof(int));
    for (int i = 0; i <= n; i++) dp[i * (n + 1) + 0] = i * GAP;
    for (int j = 0; j <= n; j++) dp[0 * (n + 1) + j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            int diag = dp[(i - 1) * (n + 1) + (j - 1)] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int up   = dp[(i - 1) * (n + 1) + j] + GAP;
            int left = dp[i * (n + 1) + (j - 1)] + GAP;
            int best = diag;
            if (up > best) best = up;
            if (left > best) best = left;
            dp[i * (n + 1) + j] = best;
        }
    }
    int result = dp[n * (n + 1) + n];
    free(dp);
    return result;
}

int kernel(int n, const char *a, const char *b) {
    if (n < 64) return scalar_nw(n, a, b);

    /* Anti-diagonal wavefront: dp(i,j) depends only on cells at i+j-1 and
       i+j-2, so a whole anti-diagonal (the "houses" of one trial) is
       mutually independent and is judged together with AVX2, 8 lanes
       (int32) at a time -- one "flute note" (loop step) per diagonal,
       many houses per note. */
    size_t buf_sz = (size_t)(n + 1) * sizeof(int);
    int *prev2 = (int *)malloc(buf_sz);
    int *prev1 = (int *)malloc(buf_sz);
    int *cur   = (int *)malloc(buf_sz);

    prev2[0] = 0;     /* d = 0 : (0,0)  */
    prev1[0] = GAP;    /* d = 1 : (0,1)  */
    prev1[1] = GAP;    /* d = 1 : (1,0)  */

    const __m256i vmatch    = _mm256_set1_epi32(MATCH);
    const __m256i vmismatch = _mm256_set1_epi32(MISMATCH);
    const __m256i vgap      = _mm256_set1_epi32(GAP);

    for (int d = 2; d <= 2 * n; d++) {
        int i_lo = d - n; if (i_lo < 0) i_lo = 0;

        if (i_lo == 0) cur[0] = d * GAP;   /* top boundary,  j = d */
        if (d <= n)    cur[d] = d * GAP;   /* left boundary, j = 0 */

        int lo = i_lo < 1 ? 1 : i_lo;
        int hi = (d - 1 < n) ? (d - 1) : n;

        int i = lo;
        while (i <= hi) {
            int remaining = hi - i + 1;
            if (remaining >= 8) {
                int abuf[8], bbuf[8];
                for (int t = 0; t < 8; t++) {
                    int ii = i + t;
                    int jj = d - ii;
                    abuf[t] = (unsigned char)a[ii - 1];
                    bbuf[t] = (unsigned char)b[jj - 1];
                }
                __m256i av = _mm256_loadu_si256((const __m256i *)abuf);
                __m256i bv = _mm256_loadu_si256((const __m256i *)bbuf);
                __m256i eq = _mm256_cmpeq_epi32(av, bv);
                __m256i add = _mm256_blendv_epi8(vmismatch, vmatch, eq);

                __m256i diagv = _mm256_loadu_si256((const __m256i *)&prev2[i - 1]);
                __m256i upv   = _mm256_loadu_si256((const __m256i *)&prev1[i - 1]);
                __m256i leftv = _mm256_loadu_si256((const __m256i *)&prev1[i]);

                diagv = _mm256_add_epi32(diagv, add);
                upv   = _mm256_add_epi32(upv, vgap);
                leftv = _mm256_add_epi32(leftv, vgap);

                __m256i best = _mm256_max_epi32(diagv, upv);
                best = _mm256_max_epi32(best, leftv);

                _mm256_storeu_si256((__m256i *)&cur[i], best);
                i += 8;
            } else {
                for (int t = 0; t < remaining; t++) {
                    int ii = i + t;
                    int jj = d - ii;
                    int diag = prev2[ii - 1] + (a[ii - 1] == b[jj - 1] ? MATCH : MISMATCH);
                    int up   = prev1[ii - 1] + GAP;
                    int left = prev1[ii] + GAP;
                    int best = diag;
                    if (up > best) best = up;
                    if (left > best) best = left;
                    cur[ii] = best;
                }
                i += remaining;
            }
        }

        int *tmp = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur = tmp;
    }

    int result = prev1[n];
    free(prev2);
    free(prev1);
    free(cur);
    return result;
}
