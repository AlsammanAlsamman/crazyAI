STORY:

I lay the two strings edge to edge along a board of small stone cells, one string down the left flank, one along the top rim, and I do not walk it corner by corner counting one match then the next the way a mind alone in a cave counts shadows - I send a single crawling body across the board's own diagonals, the way a river cuts every forest it crosses in one pass instead of tree by tree. First, while the body waits coiled at the origin, I set the two straight walls still: down the left edge each cell falls by the price of a gap for every step from the top, along the top rim each cell falls the same way for every step from the left - these are fixed before the crawl begins, ground that does not move again. Then the body uncoils and moves not rightward, not downward, but slantwise, and at each halt of its crawl it does not occupy one cell but an entire slanted seam of the board at once - every cell whose two distances from the walls add to the same tally - because none of those cells needs anything from its neighbors along that same seam, only from the two seams already laid behind it, the one touching corner-wise and the two touching edge-wise. Where that seam is narrow, near the two far corners, or where the whole board itself is too small a field to bother, I fill it myself, swift and alone, one hand moving cell to cell, since summoning many hands to lay three or four stones costs more than the laying. Where the seam runs wide - the board's true belly - I call on many hands together, each taking its own stretch of the seam, none waiting on another within it, all comparing their letter against the letter across, choosing the better of three paths: through the corner with a match or a mismatch, down from above, in from beside. I throw away nothing until the very end, keeping the whole board until the crawl reaches the farthest corner, and only then do I read that single stone and let the rest of the board go still and forgotten. This guards against the bad season where the board is too small for many hands to be worth the calling - I never summon them then, only when the seam itself is wide enough to earn it.

PREDICTION: 3.0

ARTIFACT:
```c
#include <stdlib.h>
#include <omp.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    int w = n + 1;
    int *dp = (int *)malloc((size_t)w * (size_t)w * sizeof(int));

    for (int i = 0; i <= n; i++) dp[(size_t)i * w + 0] = i * GAP;
    for (int j = 0; j <= n; j++) dp[0 * w + j] = j * GAP;

    const int PAR_THRESHOLD = 64; /* below this seam width, working alone beats calling many hands */

    for (int d = 2; d <= 2 * n; d++) {
        int i_lo = (d - n) > 1 ? (d - n) : 1;
        int i_hi = (d - 1) < n ? (d - 1) : n;
        int len = i_hi - i_lo + 1;
        if (len <= 0) continue;

        if (len >= PAR_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (int i = i_lo; i <= i_hi; i++) {
                int j = d - i;
                int diag = dp[(size_t)(i - 1) * w + (j - 1)] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
                int up = dp[(size_t)(i - 1) * w + j] + GAP;
                int left = dp[(size_t)i * w + (j - 1)] + GAP;
                int best = diag;
                if (up > best) best = up;
                if (left > best) best = left;
                dp[(size_t)i * w + j] = best;
            }
        } else {
            for (int i = i_lo; i <= i_hi; i++) {
                int j = d - i;
                int diag = dp[(size_t)(i - 1) * w + (j - 1)] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
                int up = dp[(size_t)(i - 1) * w + j] + GAP;
                int left = dp[(size_t)i * w + (j - 1)] + GAP;
                int best = diag;
                if (up > best) best = up;
                if (left > best) best = left;
                dp[(size_t)i * w + j] = best;
            }
        }
    }

    int result = dp[(size_t)n * w + n];
    free(dp);
    return result;
}
```