#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    /* reversed copy of b: br[m] = b[n-1-m], lets both operand streams
       be read forward/contiguously as the anti-diagonal sweeps i upward */
    char *br = (char *)malloc((size_t)n * sizeof(char));
    for (int m = 0; m < n; m++) br[m] = b[n - 1 - m];

    int *D2 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diag k-2 */
    int *D1 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diag k-1 */
    int *D0 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* anti-diag k   */

    int result = 0;
    int twoN = 2 * n;

    for (int k = 0; k <= twoN; k++) {
        int lo = (k > n) ? (k - n) : 0;
        int hi = (k < n) ? k : n;

        if (k <= n) {
            /* both boundary cells (i=0,j=k) and (i=k,j=0) have value k*GAP */
            int bv = k * GAP;
            D0[0] = bv;
            D0[k] = bv;
        }

        int ilo = (lo < 1) ? 1 : lo;
        int ihiCap = k - 1;
        int ihi = (hi < ihiCap) ? hi : ihiCap;

        const char * restrict arow = a;
        const char * restrict brow = br;
        int * restrict d2 = D2;
        int * restrict d1 = D1;
        int * restrict d0 = D0;
        int off = n - k;

        #pragma omp simd
        for (int i = ilo; i <= ihi; i++) {
            int match  = (arow[i - 1] == brow[off + i]) ? MATCH : MISMATCH;
            int diagv  = d2[i - 1] + match;
            int upv    = d1[i - 1] + GAP;
            int leftv  = d1[i]     + GAP;
            int best = diagv;
            if (upv   > best) best = upv;
            if (leftv > best) best = leftv;
            d0[i] = best;
        }

        if (k == twoN) {
            result = D0[n];
        }

        int *tmp = D2;
        D2 = D1;
        D1 = D0;
        D0 = tmp;
    }

    free(D2);
    free(D1);
    free(D0);
    free(br);

    return result;
}
