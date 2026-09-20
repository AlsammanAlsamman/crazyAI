STORY

I lay the two skeins side by side along the table, and between them raise a lattice of pegs — one row along the top for the first skein, one column down the side for the second, and the corner peg I mark with nothing, since two things not yet met owe each other nothing. I never build the whole lattice in stone before I start; I carry only two rows of pegs at a time, the one just finished and the one I am setting, for holding a whole field of them costs more than the answer is worth. At each peg I weigh three neighbors already set — above, to the left, and above-and-left — and take whichever path costs least: joining thread to thread that agree earns a bead of gain, joining thread that differ costs a bead, and letting one thread wait idle while the other passes costs two beads, for that is the price of slipping out of step. I keep only the kindest of the three at each peg and let the rest fall away unremembered. When the work is small I walk it alone, peg by peg, quick and quiet, for calling other hands to so short a table wastes more breath than it saves. When the skeins run long, I turn the lattice on its slant and call the other weavers to lay a whole diagonal at once, since every peg on a slant needs only what the last two slants already gave. The number waiting at the farthest corner is the one I carry back.

ARTIFACT

```c
#include <stdlib.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    /* Small/medium work: two rolling rows, O(n) memory, cache-tight, single hand.
       No point spinning up other weavers for a short table. */
    if (n < 2000) {
        int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
        int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
        for (int j = 0; j <= n; j++) prev[j] = j * GAP;
        for (int i = 1; i <= n; i++) {
            cur[0] = i * GAP;
            char ai = a[i - 1];
            for (int j = 1; j <= n; j++) {
                int diag = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
                int up   = prev[j] + GAP;
                int left = cur[j - 1] + GAP;
                int best = diag;
                if (up   > best) best = up;
                if (left > best) best = left;
                cur[j] = best;
            }
            int *tmp = prev; prev = cur; cur = tmp;
        }
        int result = prev[n];
        free(prev);
        free(cur);
        return result;
    }

    /* Large work: anti-diagonal wavefront. Every peg on diagonal k depends only
       on diagonals k-1 and k-2, so a whole diagonal can be set at once. */
    int N = n + 1;
    int *d0 = (int *)malloc((size_t)N * sizeof(int)); /* diagonal k-2 */
    int *d1 = (int *)malloc((size_t)N * sizeof(int)); /* diagonal k-1 */
    int *d2 = (int *)malloc((size_t)N * sizeof(int)); /* diagonal k   */

    d0[0] = 0;          /* k=0: (0,0) */
    d1[0] = 1 * GAP;    /* k=1: (1,0) */
    d1[1] = 1 * GAP;    /* k=1: (0,1) */

    for (int k = 2; k <= 2 * n; k++) {
        int i_lo = (k - n) > 0 ? (k - n) : 0;
        int i_hi = (k < n) ? k : n;

        #pragma omp parallel for schedule(static) if((i_hi - i_lo) > 512)
        for (int i = i_lo; i <= i_hi; i++) {
            int j = k - i;
            if (i == 0) {
                d2[i] = j * GAP;
            } else if (j == 0) {
                d2[i] = i * GAP;
            } else {
                int diag = d0[i - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
                int up   = d1[i - 1] + GAP;
                int left = d1[i] + GAP;
                int best = diag;
                if (up   > best) best = up;
                if (left > best) best = left;
                d2[i] = best;
            }
        }

        int *tmp = d0; d0 = d1; d1 = d2; d2 = tmp;
    }

    int result = d1[n];
    free(d0);
    free(d1);
    free(d2);
    return result;
}
```

PREDICTION: 4