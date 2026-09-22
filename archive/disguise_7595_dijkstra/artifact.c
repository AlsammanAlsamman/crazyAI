#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { double d; int u; } HeapItem;

/* 4-ary heap: children of i are 4i+1..4i+4, parent of i is (i-1)/4 */
static inline void heap_sift_up(HeapItem *h, int i) {
    while (i > 0) {
        int p = (i - 1) >> 2;
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
    int n = *hs;
    while (1) {
        int c0 = (i << 2) + 1;
        if (c0 >= n) break;
        int c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
        int s = c0; double sd = h[c0].d;
        if (c1 < n && h[c1].d < sd) { s = c1; sd = h[c1].d; }
        if (c2 < n && h[c2].d < sd) { s = c2; sd = h[c2].d; }
        if (c3 < n && h[c3].d < sd) { s = c3; sd = h[c3].d; }
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
    return top;
}

static void dijkstra_heap(int n, int m, const int *off, const int *edst,
                           const double *ew, int source, double *dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    char *done = calloc((size_t)n, 1);
    HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0;
    heap[hs].d = 0.0; heap[hs].u = source; hs++;
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
                int i = hs++;
                heap[i].d = nd; heap[i].u = v;
                heap_sift_up(heap, i);
            }
        }
    }
    free(done); free(heap);
}

static void dijkstra_array(int n, const int *off, const int *edst,
                            const double *ew, int source, double *dist_out) {
    double *scan = malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) { dist_out[i] = INFINITY; scan[i] = INFINITY; }
    dist_out[source] = 0.0; scan[source] = 0.0;
    for (int iter = 0; iter < n; iter++) {
        int u = -1; double best = INFINITY;
        for (int i = 0; i < n; i++) {
            double s = scan[i];
            if (s < best) { best = s; u = i; }
        }
        if (u < 0) break;
        scan[u] = INFINITY;
        double du = dist_out[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; scan[v] = nd; }
        }
    }
    free(scan);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    int *cursor = malloc((size_t)n * sizeof(int));
    memcpy(cursor, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cursor[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(deg); free(cursor);

    /* crossover: n^2 (array scan) vs (n+m)*log2(n) (heap) */
    double logn = 1.0;
    { unsigned x = (unsigned)(n > 1 ? n : 2); while (x > 1) { logn += 1.0; x >>= 1; } }
    double array_cost = (double)n * (double)n;
    double heap_cost = ((double)n + (double)m) * logn;

    if (array_cost <= heap_cost) {
        dijkstra_array(n, off, edst, ew, source, dist_out);
    } else {
        dijkstra_heap(n, m, off, edst, ew, source, dist_out);
    }

    free(off); free(edst); free(ew);
}
