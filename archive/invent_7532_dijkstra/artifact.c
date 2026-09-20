#include <stdlib.h>
#include <math.h>

typedef struct { double d; int u; } HeapItem;
typedef struct { int v; double w; } Edge;

static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static HeapItem hpop(HeapItem *h, int *hs) {
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
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;

    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];

    Edge *adj = malloc((size_t)(m > 0 ? m : 1) * sizeof(Edge));
    int *cursor = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) cursor[i] = off[i];
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cursor[u]++;
        adj[pos].v = dst[i];
        adj[pos].w = weight[i];
    }

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
        done[u] = 1;
        double du = dist_out[u];
        int start = off[u], end = off[u + 1];
        for (int e = start; e < end; e++) {
            int v = adj[e].v;
            double nd = du + adj[e].w;
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                hpush(heap, &hs, nd, v);
            }
        }
    }

    free(deg); free(off); free(adj); free(cursor); free(done); free(heap);
}
