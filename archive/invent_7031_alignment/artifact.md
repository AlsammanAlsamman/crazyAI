STORY:

I take the ribbon of chalk squares that would otherwise make me walk it flat, corner to corner, one throw of the stone after another, and I fold it instead, the way market-women fold bolt-cloth against its own bias so the far edge meets the near one without a single extra step of walking. The fold runs along the slanting middle — the diagonal where a mark on the left ribbon and a mark on the right ribbon first come level with each other — and at that crease I can read many cells at once, layer pressed on layer, the way a jewel in the net answers every other jewel in the same instant rather than waiting its turn. What stays still are the two creases already folded and set down behind me — I never disturb them, only borrow from them, the diagonal two folds back for a match-or-mismatch, the diagonal one fold back twice over for the gap-moves up and left. What moves is the new crease itself, one boundary bead threaded onto each end by hand — the edge of the sheet is a single fixed stone, no argument needed — while the many beads between the ends I hand out to as many gripping hands as the crease is wide enough to be worth calling, the way the crowd only stampedes when there is a crowd to stampede; a crease too narrow gets walked by my own two hands alone, since summoning the whole net over one gem wastes more breath than it saves. I throw away the oldest of the three creases the moment its beads are spent, keeping only the two live ones and the one being strung. I wait, at the end, only for the last fold to close on itself — the single bead where both ribbons run out together — and that bead is the number the market has been shivering toward the whole time.

ARTIFACT:
```c
#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int len = n + 1;
    int *prev2 = (int*)malloc((size_t)len * sizeof(int)); /* diagonal d-2 */
    int *prev1 = (int*)malloc((size_t)len * sizeof(int)); /* diagonal d-1 */
    int *cur   = (int*)malloc((size_t)len * sizeof(int)); /* diagonal d   */

    /* d = 0 : the single corner bead */
    prev2[0] = 0;
    /* d = 1 : the two beads adjacent to the corner */
    prev1[0] = GAP;                 /* (0,1) */
    if (len > 1) prev1[1] = GAP;    /* (1,0) */

    int maxd = 2 * n;
    const int PARWIDTH = 256; /* only call many hands when the crease is wide enough to pay for it */

    for (int d = 2; d <= maxd; d++) {
        int i_lo = d - n; if (i_lo < 0) i_lo = 0;
        int i_hi = d;     if (i_hi > n) i_hi = n;

        /* the two edge beads of this crease, at most one at each end */
        if (i_lo == 0) cur[0] = d * GAP;      /* (0, d) */
        if (i_hi == d) cur[d] = d * GAP;      /* (d, 0) */

        int lo = i_lo; if (lo == 0) lo = 1;
        int hi = i_hi; if (i_hi == d) hi = d - 1;
        int width = hi - lo + 1;

        if (width > 0) {
            #pragma omp parallel for simd schedule(static) if(width > PARWIDTH)
            for (int i = lo; i <= hi; i++) {
                int j = d - i;
                int diagv = prev2[i - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
                int upv   = prev1[i - 1] + GAP;
                int leftv = prev1[i]     + GAP;
                int best  = diagv;
                if (upv   > best) best = upv;
                if (leftv > best) best = leftv;
                cur[i] = best;
            }
        }

        int *tmp = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur   = tmp;
    }

    int result = prev1[n];
    free(prev2); free(prev1); free(cur);
    return result;
}
```

PREDICTION: 3.2