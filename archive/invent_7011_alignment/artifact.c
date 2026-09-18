#include <stdlib.h>
#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *buf0 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf1 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf2 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    char *rb  = (char *)malloc((size_t)n);
    for (int k = 0; k < n; k++) rb[k] = b[n - 1 - k];

    int *prev2 = buf0; /* crease d-2 */
    int *prev1 = buf1; /* crease d-1 */
    int *cur   = buf2; /* crease d   */

    /* crease d = 0 */
    prev2[0] = 0;
    /* crease d = 1 */
    prev1[0] = GAP; /* (0,1) */
    prev1[1] = GAP; /* (1,0) */

    for (int d = 2; d <= 2 * n; d++) {
        int i_lo = d - n; if (i_lo < 0) i_lo = 0;
        int i_hi = d;     if (i_hi > n) i_hi = n;

        int left_boundary = (i_lo == 0);       /* i = 0, j = d  */
        int top_boundary  = (d <= n);           /* i = d, j = 0  */

        if (left_boundary) cur[0] = prev1[0] + GAP;

        int interior_lo = left_boundary ? 1 : i_lo;
        int interior_hi = top_boundary ? d - 1 : i_hi;
        if (interior_hi > n) interior_hi = n;

        #pragma omp simd
        for (int i = interior_lo; i <= interior_hi; i++) {
            char match = (a[i - 1] == rb[n - d + i]) ? MATCH : MISMATCH;
            int diagv = prev2[i - 1] + match;
            int upv   = prev1[i - 1] + GAP;
            int leftv = prev1[i]     + GAP;
            int best = diagv;
            if (upv   > best) best = upv;
            if (leftv > best) best = leftv;
            cur[i] = best;
        }

        if (top_boundary) cur[d] = prev1[d - 1] + GAP;

        int *tmp = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur = tmp;
    }

    int result = prev1[n];
    free(buf0); free(buf1); free(buf2); free(rb);
    return result;
}
