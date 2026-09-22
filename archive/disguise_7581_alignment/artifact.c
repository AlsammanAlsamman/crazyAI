#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *g    = (int *)malloc((size_t)(n + 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    for (int i = 1; i <= n; i++) {
        const char ai = a[i - 1];
        const int row0 = i * GAP;   /* cur[0] boundary */

        const int *restrict prevp = prev;
        const char *restrict bp = b;
        int *restrict gp = g;

        gp[0] = row0;               /* g[0] represents the k=0 boundary term */

        /* Fully independent across j: computes m[j] = max(diag, up), then
         * g[j] = m[j] + 2*j. No dependency on j-1, so this vectorizes. */
        #pragma omp simd
        for (int j = 1; j <= n; j++) {
            int s = (ai == bp[j - 1]) ? MATCH : MISMATCH;
            int diagv = prevp[j - 1] + s;
            int upv   = prevp[j] + GAP;
            int m     = diagv > upv ? diagv : upv;
            gp[j] = m + 2 * j;
        }

        /* Exact closed form for cur[j] = max(m[j], cur[j-1]+GAP), proven by
         * induction: cur[j] = max_{0<=k<=j} g[k] - 2*j. Only a cheap
         * running-max remains serial; everything else above is parallel. */
        int running = row0;
        int *restrict curp = cur;
        curp[0] = row0;
        for (int j = 1; j <= n; j++) {
            if (gp[j] > running) running = gp[j];
            curp[j] = running - 2 * j;
        }

        int *tmp = prev;
        prev = cur;
        cur = tmp;
    }

    int result = prev[n];
    free(prev);
    free(cur);
    free(g);
    return result;
}
