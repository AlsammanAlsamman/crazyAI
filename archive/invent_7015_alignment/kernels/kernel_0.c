#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2
#define NEG_INF (-1000000000)

static inline int sc(char x, char y) { return (x == y) ? MATCH : MISMATCH; }

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    /* m[i]     = ash-mark test walking the hinge straight (a[i] vs b[i]) */
    /* mshAB[t] = walk after A's thread has skipped one bead (a leads)   */
    /* mshBA[t] = walk after B's thread has skipped one bead (b leads)   */
    int *m      = (int *)malloc((size_t)n * sizeof(int));
    int *mshAB  = (int *)malloc((size_t)n * sizeof(int));
    int *mshBA  = (int *)malloc((size_t)n * sizeof(int));
    int *PM     = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *PMS_AB = (int *)malloc((size_t)n * sizeof(int));
    int *PMS_BA = (int *)malloc((size_t)n * sizeof(int));

    #pragma omp parallel for simd
    for (int i = 0; i < n; i++) m[i] = sc(a[i], b[i]);

    mshAB[0] = 0;
    mshBA[0] = 0;
    #pragma omp parallel for simd
    for (int t = 1; t < n; t++) {
        mshAB[t] = sc(a[t], b[t - 1]);
        mshBA[t] = sc(b[t], a[t - 1]);
    }

    PM[0] = 0;
    for (int k = 1; k <= n; k++) PM[k] = PM[k - 1] + m[k - 1];
    int S0 = PM[n]; /* the attempt where the slack-spool is never spent */

    PMS_AB[0] = 0;
    PMS_BA[0] = 0;
    for (int k = 1; k < n; k++) {
        PMS_AB[k] = PMS_AB[k - 1] + mshAB[k];
        PMS_BA[k] = PMS_BA[k - 1] + mshBA[k];
    }

    /* every point along the row, folded into one running-max scan */
    int bestAB = NEG_INF, bestBA = NEG_INF;
    {
        int runMaxL_AB = NEG_INF, runMaxL_BA = NEG_INF;
        for (int q = 0; q < n; q++) {
            int LqAB = PM[q] - PMS_AB[q];
            int LqBA = PM[q] - PMS_BA[q];
            if (LqAB > runMaxL_AB) runMaxL_AB = LqAB;
            if (LqBA > runMaxL_BA) runMaxL_BA = LqBA;

            int SMq1 = S0 - PM[q + 1];
            int candAB = runMaxL_AB + PMS_AB[q] + SMq1;
            int candBA = runMaxL_BA + PMS_BA[q] + SMq1;
            if (candAB > bestAB) bestAB = candAB;
            if (candBA > bestBA) bestBA = candBA;
        }
        bestAB += 2 * GAP;
        bestBA += 2 * GAP;
    }

    int result = S0;
    if (bestAB > result) result = bestAB;
    if (bestBA > result) result = bestBA;

    free(m); free(mshAB); free(mshBA);
    free(PM); free(PMS_AB); free(PMS_BA);
    return result;
}
