#include <stdlib.h>
#define MATCH 1
#define MISMATCH -1
#define GAP -2

static inline int sc(char x, char y) { return (x == y) ? MATCH : MISMATCH; }

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *prefix_diag = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *suf_plus    = (int *)malloc((size_t)n * sizeof(int));
    int *suf_minus   = (int *)malloc((size_t)n * sizeof(int));

    /* Seed 1: both watches melt at the identical rate -- the synchronized,
       no-slip comparison, one shared position i for both strings. */
    prefix_diag[0] = 0;
    for (int i = 0; i < n; i++)
        prefix_diag[i + 1] = prefix_diag[i] + sc(a[i], b[i]);

    /* What the beads would look like downstream of a single one-groove
       carriage-crossing, in either direction (a leads, or b leads). */
    suf_plus[n - 1]  = 0;
    suf_minus[n - 1] = 0;
    for (int i = n - 2; i >= 0; i--) {
        suf_plus[i]  = suf_plus[i + 1]  + sc(a[i], b[i + 1]);
        suf_minus[i] = suf_minus[i + 1] + sc(a[i + 1], b[i]);
    }

    int two_gap = 2 * GAP;  /* the one permitted crossing: the opening toll
                                plus the forced closing toll at the far edge */
    int best = prefix_diag[n];

    /* Seed 3: test every branch of the dead tree (every candidate
       slip position) independently, in parallel, and keep only the
       branch where the beads land together best downstream. */
    #pragma omp parallel
    {
        int local_best = prefix_diag[n];
        #pragma omp for nowait
        for (int t1 = 0; t1 < n; t1++) {
            int splus  = prefix_diag[t1] + two_gap + suf_plus[t1];
            int sminus = prefix_diag[t1] + two_gap + suf_minus[t1];
            if (splus  > local_best) local_best = splus;
            if (sminus > local_best) local_best = sminus;
        }
        #pragma omp critical
        {
            if (local_best > best) best = local_best;
        }
    }

    free(prefix_diag);
    free(suf_plus);
    free(suf_minus);
    return best;
}
