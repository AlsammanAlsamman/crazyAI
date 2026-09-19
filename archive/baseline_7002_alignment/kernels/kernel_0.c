#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cand = (int *)malloc((size_t)(n + 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    for (int i = 1; i <= n; i++) {
        const char ac = a[i - 1];
        const int *restrict pr = prev;
        const char *restrict bb = b;
        int *restrict cd = cand;

        /* No loop-carried dependency here: diag/up only read the
           already-finished previous row, so this vectorizes freely. */
        #pragma omp simd
        for (int j = 1; j <= n; j++) {
            int eq   = (ac == bb[j - 1]);
            int diag = pr[j - 1] + MISMATCH + 2 * eq;  /* eq? MATCH : MISMATCH */
            int up   = pr[j] + GAP;
            cd[j] = diag > up ? diag : up;
        }

        /* Serial part is now minimal: fold in the left/gap chain. */
        int *restrict cr = cur;
        cr[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int left = cr[j - 1] + GAP;
            cr[j] = cd[j] > left ? cd[j] : left;
        }

        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[n];
    free(prev);
    free(cur);
    free(cand);
    return result;
}
