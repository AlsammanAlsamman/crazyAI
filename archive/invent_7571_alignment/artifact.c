#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    /* The table's two edges are the strings; the cord's body only ever needs
       the ring-row it is on and the one just behind it -- everything older
       is swept from the stone (SEED 3). So: O(n) memory, not O(n^2). */
    int *restrict prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *restrict cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *restrict mrow = (int *)malloc((size_t)(n + 1) * sizeof(int));

    for (int j = 0; j <= n; j++) prev[j] = j * GAP;

    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];

        /* "the girls with ground faces mark each square with the letter it
           must answer to, using chalk cut from the dirt FIRST" -- the
           agreement coin for a whole row is decided before the cord ever
           crawls it. This pass has no loop-carried dependency (each square's
           chalk mark depends only on the two letters meeting there), so it
           is free to auto-vectorize under -O3 -march=native. */
        for (int j = 1; j <= n; j++) {
            mrow[j] = (ai == b[j - 1]) ? MATCH : MISMATCH;
        }

        /* The single one-cell cord crawls this row left to right. At each
           ring it may step forward along one thread (up), sideways along
           the other (left), aslant through both (diag), or curl back onto
           the ring just behind it to record a slip (a gap, via up/left).
           It presses the coin already chalked, adds/spends the carried
           weight, and its whole worn trail behind it is this one row. */
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int diag = prev[j - 1] + mrow[j];
            int up   = prev[j]     + GAP;
            int left = cur[j - 1]  + GAP;
            int best = diag > up ? diag : up;   /* branchless: no compare-and-
                                                    branch drama, just weighing
                                                    two coins against each other */
            best = left > best ? left : best;
            cur[j] = best;
        }

        /* the ring the cord stands on becomes a fixed household (window/
           cylinder) the moment the cord leaves it */
        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[n];   /* the last ring's weight -- the only coin kept */
    free(prev);
    free(cur);
    free(mrow);
    return result;
}
