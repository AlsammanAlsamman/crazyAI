#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---------- CSR build (shared) ---------- */
static void build_csr(int n, int m, const int *src, const int *dst, const double *weight,
                       int **off_out, int **edst_out, double **ew_out) {
    int *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    int *edst = (int*)malloc((size_t)m * sizeof(int));
    double *ew = (double*)malloc((size_t)m * sizeof(double));
    int *cursor = (int*)malloc((size_t)n * sizeof(int));
    memcpy(cursor, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cursor[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(cursor);
    *off_out = off; *edst_out = edst; *ew_out = ew;
}

/* ---------- Dense O(n^2) array-scan Dijkstra (small/dense graphs) ---------- */
static void dijkstra_dense(int n, const int *off, const int *edst, const double *ew,
                            int source, double *dist_out) {
    char *done = (char*)calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    for (int iter = 0; iter < n; iter++) {
        int u = -1;
        double best = INFINITY;
        for (int v = 0; v < n; v++) {
            double dv = dist_out[v];
            if (!done[v] && dv < best) { best = dv; u = v; }
        }
        if (u < 0) break; /* remaining nodes unreachable */
        done[u] = 1;
        double du = dist_out[u];
        int e0 = off[u], e1 = off[u + 1];
        for (int e = e0; e < e1; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) dist_out[v] = nd;
        }
    }
    free(done);
}

/* ---------- 4-ary heap based Dijkstra (sparse/large graphs) ---------- */
typedef struct { double d; int u; } HeapItem;

static inline void heap_sift_up(HeapItem *h, int i) {
    HeapItem it = h[i];
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= it.d) break;
        h[i] = h[p];
        i = p;
    }
    h[i] = it;
}

static inline void heap_sift_down(HeapItem *h, int hs, int i) {
    HeapItem it = h[i];
    for (;;) {
        int c0 = 4 * i + 1;
        if (c0 >= hs) break;
        int c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
        int s = c0;
        double sd = h[c0].d;
        if (c1 < hs && h[c1].d < sd) { s = c1; sd = h[c1].d; }
        if (c2 < hs && h[c2].d < sd) { s = c2; sd = h[c2].d; }
        if (c3 < hs && h[c3].d < sd) { s = c3; sd = h[c3].d; }
        if (sd >= it.d) break;
        h[i] = h[s];
        i = s;
    }
    h[i] = it;
}

static void dijkstra_sparse(int n, int m, const int *off, const int *edst, const double *ew,
                             int source, double *dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    char *done = (char*)calloc((size_t)n, 1);
    HeapItem *heap = (HeapItem*)malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0;

    heap[hs].d = 0.0; heap[hs].u = source; hs++;

    while (hs > 0) {
        HeapItem top = heap[0];
        hs--;
        heap[0] = heap[hs];
        if (hs > 0) heap_sift_down(heap, hs, 0);

        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;

        double du = dist_out[u];
        int e0 = off[u], e1 = off[u + 1];
        for (int e = e0; e < e1; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                heap[hs].d = nd; heap[hs].u = v;
                heap_sift_up(heap, hs);
                hs++;
            }
        }
    }

    free(done);
    free(heap);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;
    if (m == 0) {
        for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
        dist_out[source] = 0.0;
        return;
    }

    int *off, *edst;
    double *ew;
    build_csr(n, m, src, dst, weight, &off, &edst, &ew);

    /* Heuristic: dense/small graphs favor the O(n^2) scan (cache-friendly,
       vectorizable, no heap bookkeeping); sparse/large graphs favor the
       4-ary heap which does O((n+m) log n) work with fewer tree levels
       than a binary heap. */
    int use_dense = (n <= 4000) || ((double)m >= (double)n * (double)n / 20.0);

    if (use_dense) {
        dijkstra_dense(n, off, edst, ew, source, dist_out);
    } else {
        dijkstra_sparse(n, m, off, edst, ew, source, dist_out);
    }

    free(off); free(edst); free(ew);
}
