#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { double d; int u; } HeapItem;

static inline void heap_push(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 2; /* 4-ary heap parent */
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem heap_pop(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    int last = --(*hs);
    h[0] = h[last];
    int i = 0;
    while (1) {
        int c0 = 4 * i + 1;
        if (c0 >= *hs) break;
        int c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
        int s = c0;
        double sv = h[c0].d;
        if (c1 < *hs && h[c1].d < sv) { s = c1; sv = h[c1].d; }
        if (c2 < *hs && h[c2].d < sv) { s = c2; sv = h[c2].d; }
        if (c3 < *hs && h[c3].d < sv) { s = c3; sv = h[c3].d; }
        if (sv >= h[i].d) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;

    /* --- Build CSR (same counting-sort approach as reference, O(n+m)) --- */
    int *off  = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *deg  = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];

    int    *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew   = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));

    memset(deg, 0, (size_t)n * sizeof(int)); /* reuse as fill counters */
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + deg[u]++;
        edst[pos] = dst[i];
        ew[pos]   = weight[i];
    }
    free(deg);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    /* --- Cost model: pick array-scan vs heap --- */
    double logn = log2((double)n + 2.0);
    double cost_array = (double)n * (double)n;
    double cost_heap  = (double)(n + m) * logn * 6.0;

    if (cost_array <= cost_heap) {
        /* O(n^2) array-scan Dijkstra, no priority structure at all */
        char *done = (char*)calloc((size_t)n, 1);
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] < best) { best = dist_out[v]; u = v; }
            }
            if (u < 0) break; /* everything remaining is unreachable */
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
        /* 4-ary heap Dijkstra with lazy deletion */
        char *done = (char*)calloc((size_t)n, 1);
        HeapItem *heap = (HeapItem*)malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        heap_push(heap, &hs, 0.0, source);
        while (hs > 0) {
            HeapItem top = heap_pop(heap, &hs);
            int u = top.u;
            if (done[u]) continue;
            done[u] = 1;
            double du = dist_out[u];
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    heap_push(heap, &hs, nd, v);
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
