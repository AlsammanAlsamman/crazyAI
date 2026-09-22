#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Binary heap (lazy-deletion) for sparse graphs ---- */
typedef struct { double d; int u; } HeapItem;

static inline void hpush(HeapItem *restrict h, int *restrict hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem hpop(HeapItem *restrict h, int *restrict hs) {
    HeapItem top = h[0];
    int ns = --(*hs);
    h[0] = h[ns];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < ns && h[l].d < h[s].d) s = l;
        if (r < ns && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;

    /* Build CSR via counting sort (O(n+m)) */
    int *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *deg = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];

    int *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    {
        int *cursor = (int*)malloc((size_t)n * sizeof(int));
        memcpy(cursor, off, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = cursor[u]++;
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
        free(cursor);
    }
    free(deg);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) { free(off); free(edst); free(ew); return; }
    dist_out[source] = 0.0;

    /* Density/size heuristic: dense or small -> array scan; else heap */
    int use_array = (n <= 1500) || ((long long)m * 6LL >= (long long)n * (long long)n);

    if (use_array) {
        char *visited = (char*)calloc((size_t)n, 1);
        double *key = (double*)malloc((size_t)n * sizeof(double));
        memcpy(key, dist_out, (size_t)n * sizeof(double));

        for (int iter = 0; iter < n; iter++) {
            double best = INFINITY;
            int bu = -1;
            for (int i = 0; i < n; i++) {
                double kv = key[i];
                if (kv < best) { best = kv; bu = i; }
            }
            if (bu < 0) break;
            visited[bu] = 1;
            key[bu] = INFINITY;
            double du = dist_out[bu];
            for (int e = off[bu]; e < off[bu + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    key[v] = nd; /* harmless if v already visited: nd can't be < dist_out[v] then */
                }
            }
        }
        free(visited);
        free(key);
    } else {
        char *done = (char*)calloc((size_t)n, 1);
        HeapItem *heap = (HeapItem*)malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        hpush(heap, &hs, 0.0, source);
        while (hs > 0) {
            HeapItem top = hpop(heap, &hs);
            int u = top.u;
            if (done[u]) continue;
            done[u] = 1;
            double du = dist_out[u];
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    hpush(heap, &hs, nd, v);
                }
            }
        }
        free(done);
        free(heap);
    }

    free(off); free(edst); free(ew);
}
