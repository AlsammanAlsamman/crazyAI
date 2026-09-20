#include <stdlib.h>

#define MATCH     1
#define MISM_MAG  1
#define GAP_MAG   2

/* Small-n fallback: plain row-major two-row DP. Cheap and branch-simple;
   used below the point where the diagonal/vector machinery's own setup
   cost (reversing b, three rolling buffers, boundary bookkeeping) could
   plausibly outweigh what it saves. */
static int kernel_small(int n, const char *restrict a, const char *restrict b) {
    int len = n + 1;
    int *restrict prev = (int *)malloc(sizeof(int) * (size_t)len);
    int *restrict cur  = (int *)malloc(sizeof(int) * (size_t)len);
    for (int j = 0; j <= n; j++) prev[j] = j * (-GAP_MAG);
    for (int i = 1; i <= n; i++) {
        cur[0] = i * (-GAP_MAG);
        for (int j = 1; j <= n; j++) {
            int diag = prev[j - 1] + ((a[i - 1] == b[j - 1]) ? MATCH : -MISM_MAG);
            int up   = prev[j] - GAP_MAG;
            int left = cur[j - 1] - GAP_MAG;
            int best = diag;
            if (up > best) best = up;
            if (left > best) best = left;
            cur[j] = best;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int result = prev[n];
    free(prev); free(cur);
    return result;
}

/* Anti-diagonal wavefront DP: the DP grid is a DAG whose natural
   topological order is the anti-diagonal k = i + j, so every "junction"
   the worm settles on is final the first time -- the metaphor's own
   "curl back and retest" is therefore provably never useful here and is
   dropped. What survives, taken literally:
     - "no ledger apart from the sand itself"  -> only 3 rolling
       diagonals (O(n) memory) instead of the full (n+1)x(n+1) tray.
     - "one continuous reading, never broken into separate glances" ->
       every cell of one anti-diagonal is independent of its neighbours
       on that same diagonal, so it is handed to the compiler as one
       contiguous, gather-free vector loop (b is pre-reversed so both
       string indices advance with the same +1 stride as i increases). */
int kernel(int n, const char *restrict a, const char *restrict b) {
    if (n <= 0) return 0;
    if (n < 64) return kernel_small(n, a, b);

    int len = n + 1;

    char *restrict brev = (char *)malloc((size_t)n);
    for (int m = 0; m < n; m++) brev[m] = b[n - 1 - m];

    int *restrict prev2 = (int *)malloc(sizeof(int) * (size_t)len); /* diagonal k-2 */
    int *restrict prev1 = (int *)malloc(sizeof(int) * (size_t)len); /* diagonal k-1 */
    int *restrict cur   = (int *)malloc(sizeof(int) * (size_t)len); /* diagonal k   */

    prev1[0] = 0; /* diagonal k=0: the single cell (0,0) */

    for (int k = 1; k <= 2 * n; k++) {
        int lo = (k - n > 0) ? (k - n) : 0;
        int hi = (k < n) ? k : n;
        int istart = lo, iend = hi;

        if (lo == 0) {                  /* i = 0 : top-row boundary, dp[0][k] */
            cur[0] = k * (-GAP_MAG);
            istart = 1;
        }
        if (k <= n) {                   /* i = k : left-column boundary, dp[k][0] */
            cur[k] = k * (-GAP_MAG);
            if (iend == k) iend = k - 1;
        }

        int off = n - k; /* brev[off + i] == b[k - i - 1], forward stride in i */
        for (int i = istart; i <= iend; i++) {
            int diagScore = prev2[i - 1] + ((a[i - 1] == brev[off + i]) ? MATCH : -MISM_MAG);
            int upScore   = prev1[i - 1] - GAP_MAG;   /* straight along a's grain  */
            int leftScore = prev1[i]     - GAP_MAG;   /* sideways: the one stumble */
            int best = diagScore;
            if (upScore   > best) best = upScore;
            if (leftScore > best) best = leftScore;
            cur[i] = best;
        }

        int *t = prev2;   /* "a beaten mound is thrown away entirely" -- rotate, don't keep */
        prev2 = prev1;
        prev1 = cur;
        cur = t;
    }

    int result = prev1[n];

    free(prev2);
    free(prev1);
    free(cur);
    free(brev);
    return result;
}
