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
