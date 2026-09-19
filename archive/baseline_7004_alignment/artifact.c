#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    /* Reverse b once so the per-diagonal character comparison becomes a
       contiguous, constant-offset access instead of a backward stride. */
    char *rb = (char*)malloc((size_t)n);
    for (int t = 0; t < n; t++) rb[t] = b[n - 1 - t];

    int *buf0 = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *buf1 = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *buf2 = (int*)malloc((size_t)(n + 1) * sizeof(int));

    int *prev2 = buf0; /* diagonal k-2 */
    int *prev1 = buf1; /* diagonal k-1 */
    int *cur   = buf2; /* diagonal k   */

    int total_diag = 2 * n;
    for (int k = 0; k <= total_diag; k++) {
        int rowlo = (k > n) ? (k - n) : 0;
        int rowhi = (k < n) ? k : n;
        int lo = (k - n > 1) ? (k - n) : 1;
        int hi = (k - 1 < n) ? (k - 1) : n;

        if (lo <= hi) {
            int off = n - k;
            const int  * __restrict p2  = prev2;
            const int  * __restrict p1  = prev1;
            int        * __restrict pc  = cur;
            const char * __restrict pa  = a;
            const char * __restrict prb = rb;

            #pragma omp simd
            for (int i = lo; i <= hi; i++) {
                int sc    = (pa[i - 1] == prb[i + off]) ? MATCH : MISMATCH;
                int diagv = p2[i - 1] + sc;
                int upv   = p1[i - 1] + GAP;
                int leftv = p1[i]     + GAP;
                int best  = diagv > upv ? diagv : upv;
                if (leftv > best) best = leftv;
                pc[i] = best;
            }
        }
        if (rowlo == 0) cur[0] = -2 * k;   /* dp[0][k] boundary */
        if (rowhi == k) cur[k] = -2 * k;   /* dp[k][0] boundary */

        int *t = prev2; prev2 = prev1; prev1 = cur; cur = t;
    }

    int result = prev1[n];

    free(rb); free(buf0); free(buf1); free(buf2);
    return result;
}
