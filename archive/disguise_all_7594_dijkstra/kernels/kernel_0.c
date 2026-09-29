#include <stdlib.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

typedef struct { double w; int v; int pad; } Edge;

/* index of the smallest value in a[0..k), k >= 1; ties -> lowest index */
static inline int frontier_argmin(const double *a, int k)
{
#if defined(__AVX2__)
    if (k >= 16) {
        __m256d v0 = _mm256_loadu_pd(a + 0);
        __m256d v1 = _mm256_loadu_pd(a + 4);
        __m256i b0 = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i b1 = _mm256_setr_epi64x(4, 5, 6, 7);
        __m256i cur = _mm256_setr_epi64x(8, 9, 10, 11);
        const __m256i inc4 = _mm256_set1_epi64x(4);
        int i = 8;
        for (; i + 8 <= k; i += 8) {
            __m256d x0 = _mm256_loadu_pd(a + i);
            __m256d x1 = _mm256_loadu_pd(a + i + 4);
            __m256d c0 = _mm256_cmp_pd(x0, v0, _CMP_LT_OQ);
            __m256i cur2 = _mm256_add_epi64(cur, inc4);
            __m256d c1 = _mm256_cmp_pd(x1, v1, _CMP_LT_OQ);
            v0 = _mm256_min_pd(v0, x0);
            v1 = _mm256_min_pd(v1, x1);
            b0 = _mm256_blendv_epi8(b0, cur,  _mm256_castpd_si256(c0));
            b1 = _mm256_blendv_epi8(b1, cur2, _mm256_castpd_si256(c1));
            cur = _mm256_add_epi64(cur2, inc4);
        }
        {
            __m256d cm = _mm256_cmp_pd(v1, v0, _CMP_LT_OQ);
            __m256d vm = _mm256_min_pd(v0, v1);
            __m256i bm = _mm256_blendv_epi8(b0, b1, _mm256_castpd_si256(cm));
            double vals[4]; long long ids[4];
            double bv; int best;
            _mm256_storeu_pd(vals, vm);
            _mm256_storeu_si256((__m256i *)ids, bm);
            bv = vals[0]; best = (int)ids[0];
            if (vals[1] < bv) { bv = vals[1]; best = (int)ids[1]; }
            if (vals[2] < bv) { bv = vals[2]; best = (int)ids[2]; }
            if (vals[3] < bv) { bv = vals[3]; best = (int)ids[3]; }
            for (; i < k; i++) if (a[i] < bv) { bv = a[i]; best = i; }
            return best;
        }
    }
#endif
    {
        double bv = a[0];
        int best = 0, i;
        for (i = 1; i < k; i++) if (a[i] < bv) { bv = a[i]; best = i; }
        return best;
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    int *off, *pos, *act;
    Edge *E;
    double *actd;
    int i, k, bigmem;

    if (n <= 0) return;
    for (i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- CSR (counting sort, shifted-offset trick) ---- */
    off = (int *)calloc((size_t)n + 2, sizeof(int));
    for (i = 0; i < m; i++) off[src[i] + 2]++;
    for (i = 2; i <= n + 1; i++) off[i] += off[i - 1];
    E = (Edge *)malloc((size_t)m * sizeof(Edge));
    for (i = 0; i < m; i++) {
        int u = src[i];
        int p = off[u + 1]++;
        E[p].v = dst[i];
        E[p].w = weight[i];
    }
    /* now edges of u are E[off[u] .. off[u+1]) */

    /* ---- frontier: trees with a number written, no gold star yet ---- */
    pos  = (int *)malloc((size_t)n * sizeof(int));
    act  = (int *)malloc((size_t)n * sizeof(int));
    actd = (double *)malloc(((size_t)n + 8) * sizeof(double));
    for (i = 0; i < n; i++) pos[i] = -1;

    act[0] = source; actd[0] = 0.0; pos[source] = 0;
    k = 1;
    bigmem = ((double)n * sizeof(double) > 1048576.0);

    while (k > 0) {
        int bi = frontier_argmin(actd, k);
        int u = act[bi];
        double du = actd[bi];
        const Edge *e;
        int deg, j, last;

        last = --k;                      /* remove u from the frontier */
        if (bi != last) {
            int lu = act[last];
            act[bi] = lu;
            actd[bi] = actd[last];
            pos[lu] = bi;
        }
        pos[u] = -2;                     /* gold star: settled forever */

        e = E + off[u];
        deg = off[u + 1] - off[u];

        if (bigmem) {
            for (j = 0; j + 8 < deg; j++) {
                int v = e[j].v;
                double nd = du + e[j].w;
                __builtin_prefetch(&dist_out[e[j + 8].v], 1, 1);
                if (nd < dist_out[v]) {
                    int p = pos[v];
                    dist_out[v] = nd;
                    if (p >= 0) actd[p] = nd;
                    else if (p == -1) { pos[v] = k; act[k] = v; actd[k] = nd; k++; }
                }
            }
            for (; j < deg; j++) {
                int v = e[j].v;
                double nd = du + e[j].w;
                if (nd < dist_out[v]) {
                    int p = pos[v];
                    dist_out[v] = nd;
                    if (p >= 0) actd[p] = nd;
                    else if (p == -1) { pos[v] = k; act[k] = v; actd[k] = nd; k++; }
                }
            }
        } else {
            for (j = 0; j < deg; j++) {
                int v = e[j].v;
                double nd = du + e[j].w;
                if (nd < dist_out[v]) {
                    int p = pos[v];
                    dist_out[v] = nd;
                    if (p >= 0) actd[p] = nd;
                    else if (p == -1) { pos[v] = k; act[k] = v; actd[k] = nd; k++; }
                }
            }
        }
    }

    free(off); free(E); free(pos); free(act); free(actd);
}
