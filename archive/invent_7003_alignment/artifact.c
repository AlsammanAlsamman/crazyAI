#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    /* reversed b so that, along an anti-diagonal, both a[] and the
       relevant b-index move forward together -> contiguous access */
    char *brev = (char *)malloc((size_t)n * sizeof(char));
    for (int k = 0; k < n; k++) brev[k] = b[n - 1 - k];

    /* the "pond": three fixed, same-size buffers, reused every stride
       (never a full n x n grid) */
    int *buf0 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf1 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *buf2 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *PD2 = buf0; /* anti-diagonal s-2 */
    int *PD1 = buf1; /* anti-diagonal s-1 */
    int *CUR = buf2; /* anti-diagonal s   */

    int total = 2 * n;

    #pragma omp parallel default(shared)
    {
        for (int s = 0; s <= total; s++) {
            int istart = (s - n) > 0 ? (s - n) : 0;
            int iend   = (s < n) ? s : n;
            int lo = (istart > 1) ? istart : 1;
            int hi = (iend < s - 1) ? iend : (s - 1);

            if (lo <= hi) {
                /* the many caliper-pairs on this stride, judged at once */
                #pragma omp for schedule(static)
                for (int i = lo; i <= hi; i++) {
                    int j = s - i;
                    int m = n - s + i; /* brev index for b[j-1] */
                    int sc   = (a[i - 1] == brev[m]) ? MATCH : MISMATCH;
                    int diag = PD2[i - 1] + sc;
                    int up   = PD1[i - 1] + GAP;
                    int left = PD1[i]     + GAP;
                    int best = diag;
                    if (up   > best) best = up;
                    if (left > best) best = left;
                    CUR[i] = best;
                }
            }

            #pragma omp single
            {
                if (s <= n) {
                    if (s == 0) {
                        CUR[0] = 0;
                    } else {
                        CUR[0] = PD1[0] + GAP;     /* leg-0 paused: i=0 */
                        CUR[s] = PD1[s - 1] + GAP; /* leg-1 paused: j=0 */
                    }
                }
                /* the calipers close: rotate the still pond forward */
                int *tmp = PD2;
                PD2 = PD1;
                PD1 = CUR;
                CUR = tmp;
            }
        }
    }

    int result = PD1[n];

    free(brev);
    free(buf0);
    free(buf1);
    free(buf2);
    return result;
}
