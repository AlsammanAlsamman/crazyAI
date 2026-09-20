#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int len = n + 1;
    int *prev2 = (int*)malloc((size_t)len * sizeof(int)); /* diagonal d-2 */
    int *prev1 = (int*)malloc((size_t)len * sizeof(int)); /* diagonal d-1 */
    int *cur   = (int*)malloc((size_t)len * sizeof(int)); /* diagonal d   */

    /* d = 0 : the single corner bead */
    prev2[0] = 0;
    /* d = 1 : the two beads adjacent to the corner */
    prev1[0] = GAP;                 /* (0,1) */
    if (len > 1) prev1[1] = GAP;    /* (1,0) */

    int maxd = 2 * n;
    const int PARWIDTH = 256; /* only call many hands when the crease is wide enough to pay for it */

    for (int d = 2; d <= maxd; d++) {
        int i_lo = d - n; if (i_lo < 0) i_lo = 0;
        int i_hi = d;     if (i_hi > n) i_hi = n;

        /* the two edge beads of this crease, at most one at each end */
        if (i_lo == 0) cur[0] = d * GAP;      /* (0, d) */
        if (i_hi == d) cur[d] = d * GAP;      /* (d, 0) */

        int lo = i_lo; if (lo == 0) lo = 1;
        int hi = i_hi; if (i_hi == d) hi = d - 1;
        int width = hi - lo + 1;

        if (width > 0) {
            #pragma omp parallel for simd schedule(static) if(width > PARWIDTH)
            for (int i = lo; i <= hi; i++) {
                int j = d - i;
                int diagv = prev2[i - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
                int upv   = prev1[i - 1] + GAP;
                int leftv = prev1[i]     + GAP;
                int best  = diagv;
                if (upv   > best) best = upv;
                if (leftv > best) best = leftv;
                cur[i] = best;
            }
        }

        int *tmp = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur   = tmp;
    }

    int result = prev1[n];
    free(prev2); free(prev1); free(cur);
    return result;
}
