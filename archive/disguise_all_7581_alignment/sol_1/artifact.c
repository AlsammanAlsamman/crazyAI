#include <stdlib.h>
#include <stdint.h>

/* Needleman-Wunsch, match=+1 mismatch=-1 gap=-2.
   Single judge, single chalk: one thread, scalar, strict row-major,
   every cell visited, each from (up, left, diag).                     */

static __thread int   *nw_row_raw = NULL;   /* rolling DP row            */
static __thread size_t nw_row_cap = 0;
static __thread int   *nw_sc_raw  = NULL;   /* 4 score rows (disjoint)   */
static __thread size_t nw_sc_cap  = 0;

static inline int *nw_align64(int *p)
{
    uintptr_t u = ((uintptr_t)p + 63u) & ~(uintptr_t)63u;
    return (int *)u;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    const size_t m   = (size_t)n + 1u;
    const size_t pad = (m + 15u) & ~(size_t)15u;   /* keep rows 64B-aligned */

    if (nw_row_cap < pad) {
        free(nw_row_raw);
        nw_row_raw = (int *)malloc(pad * sizeof(int) + 64u);
        if (!nw_row_raw) { nw_row_cap = 0; return 0; }
        nw_row_cap = pad;
    }
    if (nw_sc_cap < pad * 4u) {
        free(nw_sc_raw);
        nw_sc_raw = (int *)malloc(pad * 4u * sizeof(int) + 64u);
        if (!nw_sc_raw) { nw_sc_cap = 0; return 0; }
        nw_sc_cap = pad * 4u;
    }

    int * const rowb = nw_align64(nw_row_raw);
    int * const scb  = nw_align64(nw_sc_raw);

    /* ---- score tables: S_c[j] = (c == b[j-1]) ? +3 : +1  (= s + 2) ---- */
    {
        int * restrict S0 = scb;
        int * restrict S1 = scb + pad;
        int * restrict S2 = scb + 2u * pad;
        int * restrict S3 = scb + 3u * pad;
        for (size_t j = 0; j <= (size_t)n; ++j) {
            S0[j] = 1; S1[j] = 1; S2[j] = 1; S3[j] = 1;
        }
        for (int j = 1; j <= n; ++j)
            scb[(size_t)((b[j - 1] >> 1) & 3) * pad + (size_t)j] = 3;
    }

    /* ---- row 0 : H[0][j] = -2j  =>  G[0][j] = H + 2j = 0 ---- */
    {
        int * restrict r = rowb;
        for (size_t j = 0; j <= (size_t)n; ++j) r[j] = 0;
    }

    /* ---- G[i][j] = max( G[i-1][j-1]+s+2 , G[i-1][j]-2 , G[i][j-1] ) ---- */
    for (int i = 1; i <= n; ++i) {
        const int * restrict S   = scb + (size_t)((a[i - 1] >> 1) & 3) * pad;
        int       * restrict row = rowb;

        int gd   = row[0];          /* G[i-1][0] : diagonal for j = 1 */
        int left = -2 * i;          /* G[i][0]   = H[i][0] + 0        */
        row[0]   = left;

        int j = 1;
        for (; j + 3 <= n; j += 4) {
            const int u0 = row[j];
            const int u1 = row[j + 1];
            const int u2 = row[j + 2];
            const int u3 = row[j + 3];

            int x0 = gd + S[j];          /* diag terms: off critical path */
            int x1 = u0 + S[j + 1];
            int x2 = u1 + S[j + 2];
            int x3 = u2 + S[j + 3];

            const int t0 = u0 - 2;       /* up terms: off critical path   */
            const int t1 = u1 - 2;
            const int t2 = u2 - 2;
            const int t3 = u3 - 2;

            if (t0 > x0) x0 = t0;
            if (t1 > x1) x1 = t1;
            if (t2 > x2) x2 = t2;
            if (t3 > x3) x3 = t3;

            /* the only loop-carried chain: four 1-cycle selects */
            if (x0 > left) left = x0;  row[j]     = left;
            if (x1 > left) left = x1;  row[j + 1] = left;
            if (x2 > left) left = x2;  row[j + 2] = left;
            if (x3 > left) left = x3;  row[j + 3] = left;

            gd = u3;
        }
        for (; j <= n; ++j) {
            const int u = row[j];
            int x = gd + S[j];
            const int t = u - 2;
            if (t > x) x = t;
            if (x > left) left = x;
            row[j] = left;
            gd = u;
        }
    }

    return rowb[n] - 2 * n;          /* H[n][n] = G[n][n] - 2n */
}
