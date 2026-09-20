#include <stdlib.h>
#include <math.h>

typedef struct { double d; int u; } HeapItem;

static inline void hswap(HeapItem *h, int i, int j) { HeapItem t = h[i]; h[i] = h[j]; h[j] = t; }

static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= h[i].d) break;
        hswap(h, p, i);
        i = p;
    }
}

static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    int last = --(*hs);
    h[0] = h[last];
    int i = 0;
    while (1) {
        int l = 2*i+1, r = 2*i+2, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        hswap(h, s, i);
        i = s;
    }
    return top;
}

/* small-graph fast path: plain O(n^2) array scan, no CSR/heap build overhead */
static void dijkstra_dense(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    char *done = calloc((size_t)n, 1);
    for (int iter = 0; iter < n; iter++) {
        int u = -1; double best = INFINITY;
        for (int i = 0; i < n; i++) if (!done[i] && dist_out[i] < best) { best = dist_out[i]; u = i; }
        if (u < 0) break;
        done[u] = 1;
        for (int e = 0; e < m; e++) {
            if (src[e] != u) continue;
            int v = dst[e];
            double nd = best + weight[e];
            if (nd < dist_out[v]) dist_out[v] = nd;
        }
    }
    free(done);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;
    if (m <= 0) {
        for (int i = 0; i < n; i++) dist_out[i] = (i == source) ? 0.0 : INFINITY;
        return;
    }
    if (n <= 128) {
        dijkstra_dense(n, m, src, dst, weight, source, dist_out);
        return;
    }

    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(fill);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    char *done = calloc((size_t)n, 1);
    HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0;
    hpush(heap, &hs, 0.0, source);

    while (hs > 0) {
        HeapItem top = hpop(heap, &hs);
        int u = top.u;
        if (done[u]) continue;
        if (top.d > dist_out[u]) continue;
        done[u] = 1;
        double du = dist_out[u];
        int e0 = off[u], e1 = off[u + 1];
        for (int e = e0; e < e1; e++) {
            int v = edst[e];
            if (done[v]) continue;
            double nd = du + ew[e];
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                hpush(heap, &hs, nd, v);
            }
        }
    }

    free(deg); free(off); free(edst); free(ew); free(done); free(heap);
}
