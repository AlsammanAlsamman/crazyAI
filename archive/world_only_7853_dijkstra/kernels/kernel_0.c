#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { double d; int u; } HeapItem;

/* 4-ary heap: fewer levels / better cache locality than binary heap */
static inline void hpush4(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem hpop4(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    int last = --(*hs);
    h[0] = h[last];
    int i = 0, sz = *hs;
    while (1) {
        int c0 = 4 * i + 1;
        if (c0 >= sz) break;
        int best = c0;
        double bd = h[c0].d;
        int c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
        if (c1 < sz && h[c1].d < bd) { best = c1; bd = h[c1].d; }
        if (c2 < sz && h[c2].d < bd) { best = c2; bd = h[c2].d; }
        if (c3 < sz && h[c3].d < bd) { best = c3; bd = h[c3].d; }
        if (bd >= h[i].d) break;
        HeapItem t = h[i]; h[i] = h[best]; h[best] = t;
        i = best;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* Build CSR (counting sort by src) */
    int *deg = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew = (double *)malloc((size_t)m * sizeof(double));
    int *cursor = (int *)malloc((size_t)n * sizeof(int));
    memcpy(cursor, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cursor[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(deg);
    free(cursor);

    /* Choose algorithm: O(n^2) array scan for small/dense graphs
       (low constant factor, no heap overhead, cache-friendly),
       else O((n+m) log n) 4-ary heap for large sparse graphs. */
    double density = (double)m / ((double)n * (double)n + 1.0);
    int use_array = (n <= 2048) || (density > 0.05);

    if (use_array) {
        char *done = (char *)calloc((size_t)n, 1);
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int i = 0; i < n; i++) {
                if (!done[i] && dist_out[i] < best) { best = dist_out[i]; u = i; }
            }
            if (u < 0) break;
            done[u] = 1;
            double du = dist_out[u];
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
        free(done);
    } else {
        char *done = (char *)calloc((size_t)n, 1);
        HeapItem *heap = (HeapItem *)malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        hpush4(heap, &hs, 0.0, source);
        while (hs > 0) {
            HeapItem top = hpop4(heap, &hs);
            int u = top.u;
            if (done[u]) continue;
            done[u] = 1;
            double du = dist_out[u];
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    hpush4(heap, &hs, nd, v);
                }
            }
        }
        free(done);
        free(heap);
    }

    free(off);
    free(edst);
    free(ew);
}
