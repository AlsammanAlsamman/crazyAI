#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *prev2 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diagonal k-2 */
    int *prev1 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diagonal k-1 */
    int *cur   = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diagonal k   */

    /* k = 0 : cell (0,0) */
    prev2[0] = 0;

    /* k = 1 : cells (0,1) and (1,0) */
    prev1[0] = GAP;
    prev1[1] = GAP;

    for (int k = 2; k <= 2 * n; k++) {
        int ilo = (k - n > 0) ? (k - n) : 0;
        int ihi = (k < n) ? k : n;

        if (ilo == 0) {                 /* boundary cell i=0, j=k : one "bank" edge */
            cur[0] = k * GAP;
            ilo = 1;
        }
        if (ihi == k) {                 /* boundary cell i=k, j=0 : the other "bank" edge */
            cur[k] = k * GAP;
            ihi = k - 1;
        }

        /* the "shallows": interior of the wavefront, fully independent cells */
        #pragma omp parallel for simd schedule(static) if(ihi - ilo > 256)
        for (int i = ilo; i <= ihi; i++) {
            int j = k - i;
            int diag = prev2[i - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int up   = prev1[i - 1] + GAP;
            int left = prev1[i]     + GAP;
            int best = diag;
            if (up   > best) best = up;
            if (left > best) best = left;
            cur[i] = best;
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
