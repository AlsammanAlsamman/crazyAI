#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    /* Small/medium work: two rolling rows, O(n) memory, cache-tight, single hand.
       No point spinning up other weavers for a short table. */
    if (n < 2000) {
        int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
        int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
        for (int j = 0; j <= n; j++) prev[j] = j * GAP;
        for (int i = 1; i <= n; i++) {
            cur[0] = i * GAP;
            char ai = a[i - 1];
            for (int j = 1; j <= n; j++) {
                int diag = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
                int up   = prev[j] + GAP;
                int left = cur[j - 1] + GAP;
                int best = diag;
                if (up   > best) best = up;
                if (left > best) best = left;
                cur[j] = best;
            }
            int *tmp = prev; prev = cur; cur = tmp;
        }
        int result = prev[n];
        free(prev);
        free(cur);
        return result;
    }

    /* Large work: anti-diagonal wavefront. Every peg on diagonal k depends only
       on diagonals k-1 and k-2, so a whole diagonal can be set at once. */
    int N = n + 1;
    int *d0 = (int *)malloc((size_t)N * sizeof(int)); /* diagonal k-2 */
    int *d1 = (int *)malloc((size_t)N * sizeof(int)); /* diagonal k-1 */
    int *d2 = (int *)malloc((size_t)N * sizeof(int)); /* diagonal k   */

    d0[0] = 0;          /* k=0: (0,0) */
    d1[0] = 1 * GAP;    /* k=1: (1,0) */
    d1[1] = 1 * GAP;    /* k=1: (0,1) */

    for (int k = 2; k <= 2 * n; k++) {
        int i_lo = (k - n) > 0 ? (k - n) : 0;
        int i_hi = (k < n) ? k : n;

        #pragma omp parallel for schedule(static) if((i_hi - i_lo) > 512)
        for (int i = i_lo; i <= i_hi; i++) {
            int j = k - i;
            if (i == 0) {
                d2[i] = j * GAP;
            } else if (j == 0) {
                d2[i] = i * GAP;
            } else {
                int diag = d0[i - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
                int up   = d1[i - 1] + GAP;
                int left = d1[i] + GAP;
                int best = diag;
                if (up   > best) best = up;
                if (left > best) best = left;
                d2[i] = best;
            }
        }

        int *tmp = d0; d0 = d1; d1 = d2; d2 = tmp;
    }

    int result = d1[n];
    free(d0);
    free(d1);
    free(d2);
    return result;
}
