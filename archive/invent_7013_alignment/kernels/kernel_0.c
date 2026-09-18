#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    int *prev = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *diag = (int*)malloc((size_t)(n + 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * GAP;

        /* "I dip my brush only where the two runes share a color": for every
           column this comparison is independent of every other column, so
           it vectorizes freely (no left/up dependency here at all). */
        #pragma omp simd
        for (int j = 1; j <= n; j++) {
            int match = (ai == b[j - 1]);
            diag[j] = prev[j - 1] + (match ? MATCH : MISMATCH);
        }

        /* "the creature hops rung to rung, always further right, resting on
           the ancestor row that holds up the corridor": this step is the
           genuinely sequential part (left-dependency), so it stays serial. */
        for (int j = 1; j <= n; j++) {
            int best = diag[j];
            int up   = prev[j] + GAP;
            int left = cur[j - 1] + GAP;
            if (up   > best) best = up;
            if (left > best) best = left;
            cur[j] = best;
        }

        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[n];
    free(prev);
    free(cur);
    free(diag);
    return result;
}
