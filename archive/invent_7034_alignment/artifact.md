STORY

I take the two strands into my hands the way I take any two things that must be told apart and told alike at once: not as a solid block, but as a rope I feed out and pull back, one loop of wire at a time. I never build the whole loom - the great square lattice of every crossing - because a lattice that big forgets itself as fast as it remembers, and the fan of my dark starts thrashing at nothing. Instead I hold only the last ring and the ring before it, the way the sequence ring holds against a dissolving kite: the far rows go dark and are thrown away the instant the near ones are lit, so my hands never carry more than a sliver of the whole shape. Where the strands agree I let the wire go taut and glow; where they differ I let it sag but not snap - a bruise, not a break; and where one strand must slip a beat out of step with the other, I lay a short dead length of wire and move on, paying for the slip but never refusing it. For a small job - two short strands, a modest weave - I do this alone, hand over hand, because calling other hands to a small loom only tangles it; the calendar-glyphs of setup and gathering would outweigh the weaving itself. But when the strands run long, I turn the lattice on its side and work it by its slanted rings instead of its straight rows - each ring a shelf of independent crossings that many hands can tension together in the same breath, waiting only for the ring before it to finish glowing before the next ring is touched. What is left at the far corner, where the last two rings meet, is the single number I hand over: the world's whole agreement, paid down for every slip.

ARTIFACT
```c
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

static int kernel_serial(int n, const char *a, const char *b) {
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
            if (up > best) best = up;
            if (left > best) best = left;
            cur[j] = best;
        }
        int *tmp = prev; prev = cur; cur = tmp;
    }
    int result = prev[n];
    free(prev); free(cur);
    return result;
}

static int kernel_diagonal(int n, const char *a, const char *b) {
    int m = n + 1;
    int *d0 = (int *)malloc((size_t)m * sizeof(int)); /* diagonal k-2 */
    int *d1 = (int *)malloc((size_t)m * sizeof(int)); /* diagonal k-1 */
    int *d2 = (int *)malloc((size_t)m * sizeof(int)); /* diagonal k   */

    d0[0] = 0;              /* k=0: i=0,j=0 */
    d1[0] = 1 * GAP;        /* k=1: i=0,j=1 */
    d1[1] = 1 * GAP;        /* k=1: i=1,j=0 */

    #pragma omp parallel
    {
        for (int k = 2; k <= 2 * n; k++) {
            int ilo = k - n; if (ilo < 0) ilo = 0;
            int ihi = k;     if (ihi > n) ihi = n;

            #pragma omp for schedule(static)
            for (int i = ilo; i <= ihi; i++) {
                int j = k - i;
                int val;
                if (i == 0)      val = j * GAP;
                else if (j == 0) val = i * GAP;
                else {
                    int diag = d0[i - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
                    int up   = d1[i - 1] + GAP;
                    int left = d1[i] + GAP;
                    int best = diag;
                    if (up > best) best = up;
                    if (left > best) best = left;
                    val = best;
                }
                d2[i] = val;
            }

            #pragma omp single
            {
                int *tmp = d0; d0 = d1; d1 = d2; d2 = tmp;
            }
        }
    }

    int result = d1[n];
    free(d0); free(d1); free(d2);
    return result;
}

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;
    if (n < 1500) return kernel_serial(n, a, b);
    return kernel_diagonal(n, a, b);
}
```

PREDICTION: 4.0