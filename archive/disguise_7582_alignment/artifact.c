#include <stdlib.h>
#include <omp.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int *buf0 = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *buf1 = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *buf2 = (int*)malloc((size_t)(n + 1) * sizeof(int));
    char *br  = (char*)malloc((size_t)n);
    for (int x = 0; x < n; x++) br[x] = b[n - 1 - x];

    int *bufs[3] = { buf0, buf1, buf2 };

    /* k = 0 : dp[0][0] = 0 */
    bufs[0][0] = 0;

    /* k = 1 : boundary row/col only */
    {
        int k = 1;
        int imin = (k - n) > 0 ? (k - n) : 0;
        int imax = (k < n) ? k : n;
        int *cur = bufs[k % 3];
        for (int i = imin; i <= imax; i++) {
            int j = k - i;
            cur[i] = (i == 0) ? j * GAP : i * GAP;
        }
    }

    int total = 2 * n;
    int use_par = (n > 2000);

    #pragma omp parallel if(use_par)
    {
        for (int k = 2; k <= total; k++) {
            int imin = (k - n) > 0 ? (k - n) : 0;
            int imax = (k < n) ? k : n;
            int *cur   = bufs[k % 3];
            int *prev1 = bufs[(k + 2) % 3]; /* diagonal k-1 */
            int *prev2 = bufs[(k + 1) % 3]; /* diagonal k-2 */

            #pragma omp for schedule(static)
            for (int i = imin; i <= imax; i++) {
                int j = k - i;
                int val;
                if (i == 0) {
                    val = j * GAP;
                } else if (j == 0) {
                    val = i * GAP;
                } else {
                    int diagv = prev2[i - 1] +
                                (a[i - 1] == br[n - j] ? MATCH : MISMATCH);
                    int upv   = prev1[i - 1] + GAP;
                    int leftv = prev1[i]     + GAP;
                    int best = diagv;
                    if (upv   > best) best = upv;
                    if (leftv > best) best = leftv;
                    val = best;
                }
                cur[i] = val;
            }
        }
    }

    int answer = bufs[total % 3][n];

    free(buf0); free(buf1); free(buf2); free(br);
    return answer;
}
