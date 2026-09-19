#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    for (int i = 1; i <= n; i++) {
        cur[0] = i * GAP;
        const char ai = a[i - 1];
        const int *__restrict pv = prev;
        int *__restrict cv = cur;
        const char *__restrict bb = b;

        /* Pass 1: fully independent per-j work -> auto-vectorizable */
        #pragma omp simd
        for (int j = 1; j <= n; j++) {
            int diag = pv[j - 1] + ((ai == bb[j - 1]) ? MATCH : MISMATCH);
            int up   = pv[j] + GAP;
            cv[j] = diag > up ? diag : up;
        }

        /* Pass 2: strictly sequential left-dependency (gap extension) scan */
        for (int j = 1; j <= n; j++) {
            int left = cv[j - 1] + GAP;
            if (left > cv[j]) cv[j] = left;
        }

        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[n];
    free(prev);
    free(cur);
    return result;
}
