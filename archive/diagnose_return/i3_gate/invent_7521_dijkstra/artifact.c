#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ------------------------------------------------------------------ *
 * The garden, built once: every thread that leaves a stone, grouped
 * by the stone it leaves from (CSR), priced one direction at a time.
 * ------------------------------------------------------------------ */
static void build_csr(int n, int m,
                      const int *restrict src, const int *restrict dst,
                      const double *restrict w,
                      int *restrict off, int *restrict edst, double *restrict ew)
{
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    for (int i = 0; i < m; i++) {           /* off[] used as a moving cursor */
        int u = src[i];
        int p = off[u]++;
        edst[p] = dst[i];
        ew[p]   = w[i];
    }
    for (int i = n; i > 0; i--) off[i] = off[i - 1];   /* slide the cursors back */
    off[0] = 0;
}

/* ------------------------------------------------------------------ *
 * THE FLOCK.  One circuit = one horizontal argmin over every unlocked
 * stone at once.  A locked stone carries INFINITY, so no bird can land
 * on it and no branch is needed to exclude it.  bi < 0 means every
 * remaining letter is still unreadable -> the traveler never needed
 * them -> stop (SEED 3).
 * ------------------------------------------------------------------ */
static int argmin_range(const double *restrict a, int lo, int n, double *bestout)
{
    double best = INFINITY;
    int bi = -1;
    int i = lo;
#if defined(__AVX2__)
    if (n - lo >= 16) {
        __m256d b0 = _mm256_set1_pd(INFINITY), b1 = b0;
        __m256i k0 = _mm256_set1_epi64x(-1),  k1 = k0;
        __m256i c0 = _mm256_set_epi64x(lo + 3, lo + 2, lo + 1, lo + 0);
        __m256i c1 = _mm256_set_epi64x(lo + 7, lo + 6, lo + 5, lo + 4);
        const __m256i step = _mm256_set1_epi64x(8);
        for (; i + 8 <= n; i += 8) {
            __m256d v0 = _mm256_loadu_pd(a + i);
            __m256d v1 = _mm256_loadu_pd(a + i + 4);
            __m256d m0 = _mm256_cmp_pd(v0, b0, _CMP_LT_OQ);
            __m256d m1 = _mm256_cmp_pd(v1, b1, _CMP_LT_OQ);
            b0 = _mm256_blendv_pd(b0, v0, m0);
            b1 = _mm256_blendv_pd(b1, v1, m1);
            k0 = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(k0),
                                                      _mm256_castsi256_pd(c0), m0));
            k1 = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(k1),
                                                      _mm256_castsi256_pd(c1), m1));
            c0 = _mm256_add_epi64(c0, step);
            c1 = _mm256_add_epi64(c1, step);
        }
        double    bv[8];
        long long bx[8];
        _mm256_storeu_pd(bv, b0);
        _mm256_storeu_pd(bv + 4, b1);
        _mm256_storeu_si256((__m256i *)bx, k0);
        _mm256_storeu_si256((__m256i *)(bx + 4), k1);
        for (int t = 0; t < 8; t++)
            if (bx[t] >= 0 && bv[t] < best) { best = bv[t]; bi = (int)bx[t]; }
    }
#endif
    for (; i < n; i++) { double v = a[i]; if (v < best) { best = v; bi = i; } }
    *bestout = best;
    return bi;
}

/* thick garden: send the birds. */
static void solve_flock(int n, const int *restrict off, const int *restrict edst,
                        const double *restrict ew, int source, double *restrict dist)
{
    double        *scan    = (double *)malloc((size_t)n * sizeof(double));
    unsigned char *settled = (unsigned char *)calloc((size_t)n, 1);
    if (!scan || !settled) { free(scan); free(settled); return; }

    for (int i = 0; i < n; i++) scan[i] = INFINITY;
    scan[source] = 0.0;

    int lo = 0;
    for (int it = 0; it < n; it++) {
        while (lo < n && settled[lo]) lo++;
        if (lo >= n) break;

        double best;
        int u = argmin_range(scan, lo, n, &best);
        if (u < 0) break;                    /* threads into bramble: drop them */

        settled[u] = 1;
        scan[u] = INFINITY;                  /* locked; never chalked again */

        double du = dist[u];
        int e0 = off[u], e1 = off[u + 1];
        for (int e = e0; e < e1; e++) {      /* cast, price, rechalk in place */
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; scan[v] = nd; }
        }
    }
    free(scan);
    free(settled);
}

/* ------------------------------------------------------------------ *
 * THE KNOT.  Sparse fallback: 4-ary lazy-deletion heap.  A superseded
 * thread is not untied -- it is let drop when a bird reaches it.
 * ------------------------------------------------------------------ */
typedef struct { double d; int u; int pad; } HItem;   /* 16 bytes */

static inline void h4_push(HItem *restrict h, int *hs, double d, int u)
{
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}

static inline int h4_pop(HItem *restrict h, int *hs, double *dout)
{
    int top_u = h[0].u;
    *dout = h[0].d;
    int sz = --(*hs);
    if (sz > 0) {
        HItem last = h[sz];
        int i = 0;
        for (;;) {
            int c = 4 * i + 1;
            if (c >= sz) break;
            int e = c + 4; if (e > sz) e = sz;
            int bc = c; double bd = h[c].d;
            for (int k = c + 1; k < e; k++) if (h[k].d < bd) { bd = h[k].d; bc = k; }
            if (bd >= last.d) break;
            h[i] = h[bc];
            i = bc;
        }
        h[i] = last;
    }
    return top_u;
}

static void solve_knot(int n, int m, const int *restrict off, const int *restrict edst,
                       const double *restrict ew, int source, double *restrict dist)
{
    unsigned char *done = (unsigned char *)calloc((size_t)n, 1);
    HItem *h = (HItem *)malloc((size_t)(m + 2) * sizeof(HItem));
    if (!done || !h) { free(done); free(h); return; }
    int hs = 0;
    h4_push(h, &hs, 0.0, source);
    while (hs > 0) {
        double d;
        int u = h4_pop(h, &hs, &d);
        if (done[u]) continue;               /* dropped thread, thrown away */
        done[u] = 1;
        double du = dist[u];
        int e0 = off[u], e1 = off[u + 1];
        for (int e = e0; e < e1; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; h4_push(h, &hs, nd, v); }
        }
    }
    free(done);
    free(h);
}

/* ------------------------------------------------------------------ */
void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    int    *off  = (int *)   malloc((size_t)(n + 1) * sizeof(int));
    int    *edst = (int *)   malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !edst || !ew) { free(off); free(edst); free(ew); return; }

    build_csr(n, m, src, dst, weight, off, edst, ew);

    /* count the threads in the garden: thick -> flock, thin -> knot */
    int thick = ((double)m * 32.0 >= (double)n * (double)n);
    if (thick) solve_flock(n, off, edst, ew, source, dist_out);
    else       solve_knot(n, m, off, edst, ew, source, dist_out);

    free(off); free(edst); free(ew);
}
