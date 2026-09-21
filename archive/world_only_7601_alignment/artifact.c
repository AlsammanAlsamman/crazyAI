#include <stdlib.h>

#define MATCH_S    1
#define MISMATCH_S (-1)
#define GAP        (-2)

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int *diagP2  = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal s-2 */
    int *diagP1  = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal s-1 */
    int *diagCur = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal s   */
    int *ai      = (int *)malloc((size_t)n * sizeof(int));
    int *brevi   = (int *)malloc((size_t)n * sizeof(int));

    for (int k = 0; k < n; k++) {
        ai[k]    = (int)(unsigned char)a[k];
        brevi[k] = (int)(unsigned char)b[n - 1 - k];
    }

    /* s = 0 : dp[0][0] = 0 */
    diagP2[0] = 0;
    /* s = 1 : dp[0][1] = GAP, dp[1][0] = GAP */
    diagP1[0] = GAP;
    diagP1[1] = GAP;

    for (int s = 2; s <= 2 * n; s++) {
        int iLo = s - n; if (iLo < 1) iLo = 1;
        int iHi = s - 1; if (iHi > n) iHi = n;

        if (s <= n) {
            diagCur[0] = s * GAP; /* i=0, j=s : row-0 boundary   */
            diagCur[s] = s * GAP; /* i=s, j=0 : column-0 boundary */
        }

        int base = n - s; /* brevi index = base + i, increases with i */

        const int * restrict rai      = ai;
        const int * restrict rbrevi   = brevi;
        const int * restrict rdiagP2  = diagP2;
        const int * restrict rdiagP1  = diagP1;
        int       * restrict rdiagCur = diagCur;

        #pragma omp simd
        for (int i = iLo; i <= iHi; i++) {
            int sc    = (rai[i - 1] == rbrevi[base + i]) ? MATCH_S : MISMATCH_S;
            int diagv = rdiagP2[i - 1] + sc;
            int upv   = rdiagP1[i - 1] + GAP;
            int leftv = rdiagP1[i]     + GAP;
            int best  = diagv > upv ? diagv : upv;
            best      = best  > leftv ? best : leftv;
            rdiagCur[i] = best;
        }

        int *tmp = diagP2;
        diagP2 = diagP1;
        diagP1 = diagCur;
        diagCur = tmp;
    }

    int result = diagP1[n];

    free(diagP2);
    free(diagP1);
    free(diagCur);
    free(ai);
    free(brevi);

    return result;
}
