#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2
#define SMALL_N 32          /* below this, skip diagonal restructuring: not enough work to amortize it */
#define PAR_THRESHOLD 4096  /* only thread-parallelize a diagonal with this many independent cells */

static int kernel_small(int n, const char *a, const char *b) {
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int diag = prev[j - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int up = prev[j] + GAP;
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

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;
    if (n < SMALL_N) return kernel_small(n, a, b); /* guard: fallback for the risky small-n case */

    /* Anti-diagonal wavefront: every cell of diagonal d depends only on
       diagonals d-1 and d-2, so all cells of one diagonal are mutually
       independent -- a whole "stretch" is judged at once (SIMD, and
       threads once the stretch is big enough), instead of bead by bead. */

    char *brev = (char *)malloc((size_t)n * sizeof(char));
    for (int k = 0; k < n; k++) brev[k] = b[n - 1 - k];

    int *d0 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal d-2 */
    int *d1 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal d-1 */
    int *d2 = (int *)malloc((size_t)(n + 1) * sizeof(int)); /* diagonal d (being filled) */

    d0[0] = 0;     /* dp[0][0] */
    d1[0] = GAP;   /* dp[0][1] */
    d1[1] = GAP;   /* dp[1][0] */

    int total_diag = 2 * n;

    for (int d = 2; d <= total_diag; d++) {
        int i_lo = (d - n > 0) ? (d - n) : 0;
        int i_hi = (d < n) ? d : n;
        int lo = i_lo, hi = i_hi;

        if (lo == 0) { d2[0] = d * GAP; lo = 1; }               /* top row: dp[0][j] = j*GAP */
        int right_is_boundary = (i_hi == d && d <= n);
        if (right_is_boundary) { d2[hi] = hi * GAP; hi -= 1; }  /* left col: dp[i][0] = i*GAP */

        int off = n - d; /* brev[off + i] == b[j - 1] for j = d - i */
        const int * restrict rd0 = d0;
        const int * restrict rd1 = d1;
        int * restrict rd2 = d2;
        const char * restrict ra = a;
        const char * restrict rb = brev;
        int span = hi - lo + 1;

        if (span > 0) {
            #pragma omp parallel for simd schedule(static) if(span > PAR_THRESHOLD)
            for (int i = lo; i <= hi; i++) {
                int diagv = rd0[i - 1] + (ra[i - 1] == rb[off + i] ? MATCH : MISMATCH);
                int upv = rd1[i - 1] + GAP;
                int leftv = rd1[i] + GAP;
                int best = diagv;
                if (upv > best) best = upv;
                if (leftv > best) best = leftv;
                rd2[i] = best;
            }
        }

        int *tmp = d0; d0 = d1; d1 = d2; d2 = tmp;
    }

    int result = d1[n];
    free(d0); free(d1); free(d2); free(brev);
    return result;
}
