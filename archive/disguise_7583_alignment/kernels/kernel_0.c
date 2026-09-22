#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    int *restrict prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *restrict cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *restrict sc   = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        const char *restrict bp = b;

        /* independent per-column match/mismatch scores: auto-vectorizable */
        for (int j = 0; j < n; j++) {
            sc[j] = (ai == bp[j]) ? MATCH : MISMATCH;
        }

        cur[0] = i * GAP;
        int diag = prev[0];

        for (int j = 1; j <= n; j++) {
            int d    = diag + sc[j - 1];
            int up   = prev[j] + GAP;
            int left = cur[j - 1] + GAP;

            int best = d > up ? d : up;
            if (left > best) best = left;

            cur[j] = best;
            diag = prev[j];
        }

        int *t = prev; prev = cur; cur = t;
    }

    int result = prev[n];
    free(prev);
    free(cur);
    free(sc);
    return result;
}
