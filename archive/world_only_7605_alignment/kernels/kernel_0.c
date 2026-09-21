#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    size_t width = (size_t)n + 1;
    int *prev = (int *)malloc(width * sizeof(int));
    int *curr = (int *)malloc(width * sizeof(int));
    if (!prev || !curr) { free(prev); free(curr); return 0; }

    for (size_t j = 0; j <= (size_t)n; j++) prev[j] = (int)j * GAP;

    for (int i = 1; i <= n; i++) {
        curr[0] = i * GAP;
        const char ai = a[i - 1];

        const int  * restrict pv = prev;
        int        * restrict cv = curr;
        const char * restrict bp = b;

        /* Pass 1: fully independent across j -> vectorizable.
         * pre[j] = max(diag, up), both derived only from finished row i-1. */
        #pragma omp simd
        for (int j = 1; j <= n; j++) {
            int diag = pv[j - 1] + ((ai == bp[j - 1]) ? MATCH : MISMATCH);
            int up   = pv[j] + GAP;
            cv[j] = diag > up ? diag : up;
        }

        /* Pass 2: serial gap-open propagation from the left.
         * Cheap 2-op chain (add + select), cannot vectorize but is short. */
        for (int j = 1; j <= n; j++) {
            int left = cv[j - 1] + GAP;
            if (left > cv[j]) cv[j] = left;
        }

        int *tmp = prev;
        prev = curr;
        curr = tmp;
    }

    int result = prev[n];
    free(prev);
    free(curr);
    return result;
}
