#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <immintrin.h>
#endif

/* 16-byte edge record: one memory stream per adjacency walk. */
typedef struct { double w; int v; int pad; } Edge;

/* Pip's full sweep of the board: return index of the smallest entry in a[0..k). */
static inline int scan_min(const double *a, int k)
{
    int b;
    double bv;

    if (k < 8) {
        b = 0; bv = a[0];
        for (int i = 1; i < k; i++) if (a[i] < bv) { bv = a[i]; b = i; }
        return b;
    }

#if defined(__AVX2__)
    {
        __m256d m0 = _mm256_loadu_pd(a);
        __m256d m1 = _mm256_loadu_pd(a + 4);
        int i = 8;
        for (; i + 16 <= k; i += 16) {
            __m256d x0 = _mm256_loadu_pd(a + i);
            __m256d x1 = _mm256_loadu_pd(a + i + 4);
            __m256d x2 = _mm256_loadu_pd(a + i + 8);
            __m256d x3 = _mm256_loadu_pd(a + i + 12);
            m0 = _mm256_min_pd(m0, x0);
            m1 = _mm256_min_pd(m1, x1);
            m0 = _mm256_min_pd(m0, x2);
            m1 = _mm256_min_pd(m1, x3);
        }
        for (; i + 8 <= k; i += 8) {
            m0 = _mm256_min_pd(m0, _mm256_loadu_pd(a + i));
            m1 = _mm256_min_pd(m1, _mm256_loadu_pd(a + i + 4));
        }
        {
            __m256d mv = _mm256_min_pd(m0, m1);
            __m128d lo = _mm256_castpd256_pd128(mv);
            __m128d hi = _mm256_extractf128_pd(mv, 1);
            __m128d mm = _mm_min_pd(lo, hi);
            mm = _mm_min_sd(mm, _mm_unpackhi_pd(mm, mm));
            bv = _mm_cvtsd_f64(mm);
        }
        for (; i < k; i++) if (a[i] < bv) bv = a[i];

        /* locate the line holding that value */
        {
            __m256d bvv = _mm256_set1_pd(bv);
            int j = 0;
            for (; j + 4 <= k; j += 4) {
                __m256d x = _mm256_loadu_pd(a + j);
                int msk = _mm256_movemask_pd(_mm256_cmp_pd(x, bvv, _CMP_EQ_OQ));
                if (msk) return j + __builtin_ctz((unsigned)msk);
            }
            for (; j < k; j++) if (a[j] == bv) return j;
        }
        return 0;
    }
#else
    b = 0; bv = a[0];
    for (int i = 1; i < k; i++) if (a[i] < bv) { bv = a[i]; b = i; }
    return b;
#endif
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m < 0) m = 0;

    int    *off = (int *)   malloc((size_t)(n + 1) * sizeof(int));
    int    *pos = (int *)   malloc((size_t)n * sizeof(int));   /* board slot / -1 blank / -2 locked */
    int    *ci  = (int *)   malloc((size_t)n * sizeof(int));   /* which tree each board line is */
    double *cd  = (double *)malloc((size_t)n * sizeof(double));/* the guesses, contiguous */
    Edge   *E   = (Edge *)  malloc((size_t)(m > 0 ? m : 1) * sizeof(Edge));

    if (!off || !pos || !ci || !cd || !E) {
        free(off); free(pos); free(ci); free(cd); free(E);
        return;
    }

    /* ---- CSR by counting sort; `pos` doubles as histogram, `ci` as fill cursor ---- */
    memset(pos, 0, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) pos[src[i]]++;
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + pos[i];
    memcpy(ci, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int p = ci[u]++;
        E[p].w = weight[i];
        E[p].v = dst[i];
    }
    memset(pos, 0xFF, (size_t)n * sizeof(int));   /* every tree: blank line (-1) */

    /* ---- the board starts with one line: the Big Oak at zero ---- */
    int k = 1;
    cd[0] = 0.0; ci[0] = source; pos[source] = 0;

    while (k > 0) {
        /* full sweep of every live line, smallest wins */
        int j = scan_min(cd, k);
        int u = ci[j];
        double du = cd[j];

        /* lock it in and erase its line (swap-with-last keeps lines contiguous) */
        k--;
        cd[j] = cd[k];
        ci[j] = ci[k];
        pos[ci[j]] = j;
        pos[u] = -2;
        dist_out[u] = du;

        /* walk the bridges out of it */
        {
            int e   = off[u];
            int end = off[u + 1];
            for (; e < end; e++) {
                int v = E[e].v;
                int p = pos[v];
                if (p == -2) continue;                 /* already locked */
                double nd = du + E[e].w;
                if (p >= 0) {
                    if (nd < cd[p]) cd[p] = nd;        /* erase + rewrite one line */
                } else {
                    pos[v] = k; cd[k] = nd; ci[k] = v; k++;   /* first real guess */
                }
            }
        }
    }

    free(off); free(pos); free(ci); free(cd); free(E);
}
