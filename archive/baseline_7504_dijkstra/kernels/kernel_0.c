#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { double d; int u; } HeapItem;

static inline void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    (*hs)--;
    h[0] = h[*hs];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n <= 0) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* build CSR (src -> dst,weight) */
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    {
        int *cursor = malloc((size_t)n * sizeof(int));
        memcpy(cursor, off, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = cursor[u]++;
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
        free(cursor);
    }

    char *done = calloc((size_t)n, 1);

    /* cost model: pick array-scan O(n^2+m) vs heap O((n+m)log n) */
    double log2n = log2((double)n + 2.0);
    double costArray = (double)n * (double)n + (double)m;
    double costHeap = 3.0 * ((double)n + (double)m) * log2n;
    int use_array = (n <= 64) || (costArray <= costHeap);

    if (use_array) {
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] < best) { best = dist_out[v]; u = v; }
            }
            if (u < 0) break;
            done[u] = 1;
            double du = dist_out[u];
            int e0 = off[u], e1 = off[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
    } else {
        HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        hpush(heap, &hs, 0.0, source);
        while (hs > 0) {
            HeapItem top = hpop(heap, &hs);
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
                    hpush(heap, &hs, nd, v);
                }
            }
        }
        free(heap);
    }

    free(done);
    free(off);
    free(edst);
    free(ew);
}
