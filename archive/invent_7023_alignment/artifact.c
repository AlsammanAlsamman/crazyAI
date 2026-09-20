#include <stdlib.h>
#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *restrict a, const char *restrict b) {
    if (n <= 0) return 0;

    int  *prev2 = (int  *)malloc((size_t)(n + 1) * sizeof(int));
    int  *prev1 = (int  *)malloc((size_t)(n + 1) * sizeof(int));
    int  *cur   = (int  *)malloc((size_t)(n + 1) * sizeof(int));
    char *rb    = (char *)malloc((size_t)n);

    /* "the floor rope hung at a slant": reverse b so that walking the
       anti-diagonal reads BOTH ropes forward and contiguously. */
    for (int k = 0; k < n; k++) rb[k] = b[n - 1 - k];

    /* seed diagonal d = 0 : dp[0][0] = 0 */
    prev2[0] = 0;
    /* seed diagonal d = 1 : dp[0][1] = GAP, dp[1][0] = GAP */
    prev1[0] = GAP;
    prev1[1] = GAP;

    for (int d = 2; d <= 2 * n; d++) {
        int ilo = (d - n > 0) ? d - n : 0;
        int ihi = (d < n) ? d : n;

        if (ilo == 0) cur[0] = prev1[0] + GAP;         /* i=0,j=d: left only  */
        if (ihi == d) cur[d] = prev1[d - 1] + GAP;      /* i=d,j=0: up only    */

        int mid_lo = (ilo == 0) ? 1 : ilo;
        int mid_hi = (ihi == d) ? d - 1 : ihi;

        const char *restrict aa  = a;
        const char *restrict rbb = rb;
        int  *restrict p2 = prev2;
        int  *restrict p1 = prev1;
        int  *restrict c  = cur;
        int base = n - d;   /* rb[base+i] == b[d-i-1] : contiguous forward */

        #pragma omp simd
        for (int i = mid_lo; i <= mid_hi; i++) {
            int sc    = (aa[i - 1] == rbb[base + i]) ? MATCH : MISMATCH;
            int diagv = p2[i - 1] + sc;
            int upv   = p1[i - 1] + GAP;
            int leftv = p1[i]     + GAP;
            int best  = diagv;
            if (upv   > best) best = upv;
            if (leftv > best) best = leftv;
            c[i] = best;
        }

        int *tmp = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur   = tmp;
    }

    int result = prev1[n];

    free(prev2);
    free(prev1);
    free(cur);
    free(rb);

    return result;
}
