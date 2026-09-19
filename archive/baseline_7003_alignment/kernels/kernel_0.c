#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int *prev2 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal k-2 */
    int *prev1 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal k-1 */
    int *cur   = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal k   */

    /* k = 0 : single cell dp[0][0] = 0 */
    prev2[0] = 0;

    /* k = 1 : dp[0][1] = GAP, dp[1][0] = GAP (if n >= 1, always true here) */
    {
        int lo = (1 - n > 0) ? (1 - n) : 0;
        int hi = (1 < n) ? 1 : n;
        for (int i = lo; i <= hi; i++) {
            int j = 1 - i;
            prev1[i] = (i == 0) ? j * GAP : i * GAP;
        }
    }

    for (int k = 2; k <= 2 * n; k++) {
        int lo = (k - n > 0) ? (k - n) : 0;
        int hi = (k < n) ? k : n;
        int iStart = lo, iEnd = hi;

        /* boundary cells (only present while k <= n) */
        if (lo == 0) {
            cur[0] = k * GAP;          /* dp[0][k] */
            iStart = 1;
        }
        if (hi == k && k <= n) {
            cur[k] = k * GAP;          /* dp[k][0] */
            if (iEnd == k) iEnd = k - 1;
        }

        #pragma omp simd
        for (int i = iStart; i <= iEnd; i++) {
            int j = k - i;
            int diag = prev2[i - 1] + ((a[i - 1] == b[j - 1]) ? MATCH : MISMATCH);
            int up   = prev1[i - 1] + GAP;
            int left = prev1[i]     + GAP;
            int m = diag > up ? diag : up;
            m = m > left ? m : left;
            cur[i] = m;
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
