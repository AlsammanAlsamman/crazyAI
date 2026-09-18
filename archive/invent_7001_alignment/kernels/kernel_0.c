#include <stdlib.h>
#include <omp.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

/* Anti-diagonal ("wavefront") Needleman-Wunsch: the two sequences are two
 * "planks" of symbol-codes; only three O(n) diagonals ("the hatful of
 * smells kept cool on the sill") are ever alive in memory at once, never
 * the full O(n^2) grid. Every position pair that shares a diagonal is fed
 * to the scoring "tube" simultaneously, so the sweep can be split across
 * threads/SIMD lanes instead of forced into strict row-major order. The
 * recurrence itself is untouched, so the result is bit-exact with the
 * reference O(n^2) table. */
int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    int m = n + 1;
    int *bufA = (int *)malloc(sizeof(int) * (size_t)m);
    int *bufB = (int *)malloc(sizeof(int) * (size_t)m);
    int *bufC = (int *)malloc(sizeof(int) * (size_t)m);

    int *dp2 = bufA; /* diagonal s-2 */
    int *dp1 = bufB; /* diagonal s-1 */
    int *dp0 = bufC; /* diagonal s   (being written) */

    #pragma omp parallel
    {
        for (int s = 0; s <= 2 * n; s++) {
            int i_lo = (s - n > 0) ? (s - n) : 0;
            int i_hi = (s < n) ? s : n;

            #pragma omp for simd schedule(static) if(i_hi - i_lo > 256)
            for (int i = i_lo; i <= i_hi; i++) {
                int j = s - i;
                int val;
                if (i == 0) {
                    val = j * GAP;
                } else if (j == 0) {
                    val = i * GAP;
                } else {
                    int diagv = dp2[i - 1] + ((a[i - 1] == b[j - 1]) ? MATCH : MISMATCH);
                    int upv   = dp1[i - 1] + GAP;
                    int leftv = dp1[i]     + GAP;
                    int best  = diagv;
                    if (upv   > best) best = upv;
                    if (leftv > best) best = leftv;
                    val = best;
                }
                dp0[i] = val;
            }

            #pragma omp single
            {
                int *tmp = dp2;
                dp2 = dp1;
                dp1 = dp0;
                dp0 = tmp;
            }
        }
    }

    int result = dp1[n];
    free(bufA); free(bufB); free(bufC);
    return result;
}
