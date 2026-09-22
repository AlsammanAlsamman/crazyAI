#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAPMAG 2   /* gap penalty magnitude; contributes -GAPMAG to the score */

static inline int imax3(int x, int y, int z) {
    int m = (x > y) ? x : y;
    return (m > z) ? m : z;
}

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int *buf0 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf1 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf2 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    char *brev = (char *)malloc((size_t)n * sizeof(char));

    for (int t = 0; t < n; t++) brev[t] = b[n - 1 - t];

    int *D0 = buf0; /* diagonal k-2 */
    int *D1 = buf1; /* diagonal k-1 */
    int *D2 = buf2; /* diagonal k, being filled now */

    int totalK = 2 * n;
    for (int k = 0; k <= totalK; k++) {
        int i_min = (k - n > 0) ? (k - n) : 0;
        int i_max = (k < n) ? k : n;
        int lo = i_min, hi = i_max;

        /* boundary cells: i==0 (j==k) or j==0 (i==k), only discoverable
           because diagonal k-1/k-2 (i.e. everything "before" it) is
           already fully computed -- same rule as the reference DP. */
        if (lo == 0) {
            D2[0] = -(k * GAPMAG);
            lo = 1;
        }
        if (hi == k && k <= n) {
            D2[k] = -(k * GAPMAG);
            hi = k - 1;
        }

        #pragma omp simd
        for (int i = lo; i <= hi; i++) {
            int aOff = i - 1;
            int bOff = n - k + i;   /* brev[bOff] == b[j-1], unit stride in i */
            int sc = (a[aOff] == brev[bOff]) ? MATCH : MISMATCH;
            int diag = D0[i - 1] + sc;
            int up   = D1[i - 1] - GAPMAG;
            int left = D1[i]     - GAPMAG;
            D2[i] = imax3(diag, up, left);
        }

        int *tmp = D0;
        D0 = D1;
        D1 = D2;
        D2 = tmp;
    }

    int result = D1[n];

    free(buf0);
    free(buf1);
    free(buf2);
    free(brev);
    return result;
}
