#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    /* Single "crawler": one row-wide body that lays one unbroken trail
       across the grid, carrying forward only the pigment (scores) it
       needs -- the row above (prev) and the row it is currently laying
       down (curr) -- and lets everything else wash away, unremembered. */
    size_t w = (size_t)n + 1;
    int *prev = (int *)malloc(w * sizeof(int));
    int *curr = (int *)malloc(w * sizeof(int));

    /* Row 0: rest-state at the first peg -- j gap steps from the origin. */
    for (size_t j = 0; j < w; j++) prev[j] = (int)j * GAP;

    for (int i = 1; i <= n; i++) {
        curr[0] = i * GAP;               /* column-0 boundary for this row */
        int diag = prev[0];              /* dp[i-1][0], carried as the crawler advances */
        char ai = a[i - 1];
        for (int j = 1; j <= n; j++) {
            int up_val   = prev[j];      /* the drop left by the cell above */
            int left_val = curr[j - 1];  /* the drop left by the cell behind */
            int cross    = diag + (ai == b[j - 1] ? MATCH : MISMATCH); /* diagonal: agreement/slip-in-letter */
            int side_up   = up_val   + GAP; /* orthogonal step: slipped letter, paid inline */
            int side_left = left_val + GAP; /* orthogonal step: slipped letter, paid inline */

            int best = cross;
            if (side_up   > best) best = side_up;
            if (side_left > best) best = side_left;

            diag = up_val;   /* the cell above becomes the new diagonal predecessor
                                 once the crawler steps one cell further along the row */
            curr[j] = best;  /* the drop settles here before the body moves on */
        }
        /* Curl at the wall: swap the just-laid trail into "prev" and
           reuse the old prev's memory for the next row. */
        int *tmp = prev;
        prev = curr;
        curr = tmp;
    }

    int result = prev[n];
    free(prev);
    free(curr);
    return result;
}
