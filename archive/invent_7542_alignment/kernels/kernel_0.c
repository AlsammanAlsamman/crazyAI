#include <stdlib.h>
#include <limits.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2
#define NEG_INF (INT_MIN / 2)

static inline int max3(int x, int y, int z) {
    int m = x;
    if (y > m) m = y;
    if (z > m) m = z;
    return m;
}

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    /* "shell for shell": hold both strings side by side at matching
       counted positions and press a coal print (score +1) only where
       the old-speech names truly agree; leave mismatched stone bare
       (-1). This is the single forward pass of SEED 1 + SEED 2. */
    long S0 = 0;
    for (int i = 0; i < n; i++)
        S0 += (a[i] == b[i]) ? MATCH : MISMATCH;

    /* Any path that could tie-or-beat the direct thread spends g
       gap-pairs with g <= (n - S0) / 5 (score <= n - 5g, and OPT >= S0
       because the direct path is always legal). Drift from the
       diagonal never exceeds g. W is that radius, with a +1 safety
       margin -- the one place we "pay a fixed length" up front,
       computed once, not rediscovered cell by cell. Clamped to n so
       the worst case degrades to a full pass, never worse. */
    long Wl = (n - S0 + 4) / 5 + 1;
    if (Wl < 0) Wl = 0;
    if (Wl > n) Wl = n;
    int W = (int)Wl;

    int width = 2 * W + 1;
    int *prev = (int *)malloc((size_t)width * sizeof(int));
    int *cur  = (int *)malloc((size_t)width * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }

    for (int k = 0; k < width; k++) {
        int j = (0 - W) + k;
        prev[k] = (j < 0 || j > n) ? NEG_INF : j * GAP;
    }

    for (int i = 1; i <= n; i++) {
        int lo = (i - W < 0) ? 0 : i - W;
        int hi = (i + W > n) ? n : i + W;
        for (int k = 0; k < width; k++) {
            int j = (i - W) + k;
            if (j < lo || j > hi) { cur[k] = NEG_INF; continue; }
            if (j == 0) { cur[k] = i * GAP; continue; }
            int diagv = prev[k] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int upv   = (k + 1 < width) ? prev[k + 1] + GAP : NEG_INF;
            int leftv = (k - 1 >= 0)    ? cur[k - 1] + GAP  : NEG_INF;
            cur[k] = max3(diagv, upv, leftv);
        }
        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[W];
    free(prev);
    free(cur);
    return result;
}
