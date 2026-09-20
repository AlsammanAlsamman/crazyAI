STORY

Two ribbons of small marks are brought before me, each of the same length, drawn from the four glyphs the sphere allows. I lay them along the two edges of a plane, as the room always insists sphere and stretch are the same paint. Where the grid would run corner to corner I strike the true corridor first — a slender lane hugging the main slant, a mount coiled tight, wide enough at first to be barely wider than the slant itself. I walk it once, scoring each step: a step straight through both ribbons together, credit if the glyphs agree, cost if they quarrel; a step where one ribbon halts and the other slips ahead, always a heavier cost, a slip out of step. I do not fill the far corners of the grid unless the walk gives me cause to. Before I set the lane's score down as final, I put a harder question to the plane itself, not by walking the far corridors but by reasoning about them: how good could any path possibly score if it strayed beyond my lane's edge? Every stray step it would need, and need again to return, costs more than any run of agreements could ever repay — I can name the very worst such a path could score without ever tracing it, a shadow cast onto the plane rather than a walked thing. If my lane already beats that shadow, the far corridors are dead corridors, treasures already known to hold nothing better, and I set down the lane's number as the true one. If it does not yet beat the shadow, I double the lane's width, walk again, ask again — each doubling paid for mostly by the walk it replaces, never piling wastefully atop the last. For a short ribbon this ceremony costs more than it saves, so below a small length I simply walk the whole plane corner to corner, plainly, and take that number instead. I call no second pair of hands unless the lane has grown to fill most of the room; a narrow lane is a task for one walker alone.

ARTIFACT

```c
#include <stdlib.h>
#include <limits.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2
#define NEG_INF  (INT_MIN/2)

/* Exact global-alignment score restricted to the band |i-j|<=w.
   Cells outside the band are unreachable (NEG_INF). */
static int band_score(int n, const char *a, const char *b, int w) {
    if (w > n) w = n;
    int width = 2 * w + 1;
    int *prev = (int *)malloc((size_t)width * sizeof(int));
    int *cur  = (int *)malloc((size_t)width * sizeof(int));
    for (int k = 0; k < width; k++) prev[k] = NEG_INF;
    for (int j = 0; j <= w && j <= n; j++) prev[j + w] = j * GAP;

    for (int i = 1; i <= n; i++) {
        for (int k = 0; k < width; k++) cur[k] = NEG_INF;
        int jlo = i - w; if (jlo < 0) jlo = 0;
        int jhi = i + w; if (jhi > n) jhi = n;
        for (int j = jlo; j <= jhi; j++) {
            int k = j - i + w;
            int best = NEG_INF;
            if (j >= 1) {
                int pj = j - 1, pi = i - 1;
                if (pj >= pi - w && pj <= pi + w) {
                    int kd = pj - pi + w;
                    int diag = prev[kd] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
                    if (diag > best) best = diag;
                }
            }
            {
                int pi = i - 1;
                if (j >= pi - w && j <= pi + w) {
                    int ku = j - pi + w;
                    int up = prev[ku] + GAP;
                    if (up > best) best = up;
                }
            }
            if (j >= 1) {
                int lj = j - 1;
                if (lj >= i - w && lj <= i + w) {
                    int kl = lj - i + w;
                    int left = cur[kl] + GAP;
                    if (left > best) best = left;
                }
            }
            cur[k] = best;
        }
        int *tmp = prev; prev = cur; cur = tmp;
    }
    int result = prev[w];
    free(prev);
    free(cur);
    return result;
}

/* Plain Needleman-Wunsch, used for tiny n and as the final fallback
   once the band has grown to cover the whole grid anyway. */
static int full_nw(int n, const char *a, const char *b) {
    int *dp = (int *)malloc((size_t)(n + 1) * (n + 1) * sizeof(int));
    for (int i = 0; i <= n; i++) dp[i * (n + 1) + 0] = i * GAP;
    for (int j = 0; j <= n; j++) dp[0 * (n + 1) + j] = j * GAP;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++) {
            int diag = dp[(i - 1) * (n + 1) + (j - 1)] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int up = dp[(i - 1) * (n + 1) + j] + GAP;
            int left = dp[i * (n + 1) + (j - 1)] + GAP;
            int best = diag;
            if (up > best) best = up;
            if (left > best) best = left;
            dp[i * (n + 1) + j] = best;
        }
    int result = dp[n * (n + 1) + n];
    free(dp);
    return result;
}

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;
    if (n < 64) return full_nw(n, a, b);   /* too small a job to share out or band */

    int w = 16;
    for (;;) {
        if (w >= n) return full_nw(n, a, b); /* band already covers everything */
        int s = band_score(n, a, b, w);
        /* Any path leaving the band by more than w needs at least w+1
           extra indels in one direction (and as many back), each pair
           costing at least 5 relative to an all-match diagonal step
           (losing a +1 and paying -2 twice), so no such path can beat
           n - 5*(w+1). If our banded score already meets that bound,
           it is provably the true global optimum. */
        long bound = (long)n - 5L * ((long)w + 1);
        if ((long)s >= bound) return s;
        w <<= 1;
    }
}
```

PREDICTION: 6