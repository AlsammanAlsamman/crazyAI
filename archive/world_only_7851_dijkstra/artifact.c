#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { double d; int u; } HeapItem;

static inline void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 1;
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
    int hs_local = *hs;
    while (1) {
        int l = 2*i+1, r = 2*i+2, s = i;
        if (l < hs_local && h[l].d < h[s].d) s = l;
        if (r < hs_local && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;

    /* Build CSR */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew  = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *cur = malloc((size_t)n * sizeof(int));
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cur[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(deg); free(cur);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    double logn = log2((double)n + 2.0);
    int use_array = (n <= 1500) || ((double)m * logn * 3.0 >= (double)n * (double)n);

    if (use_array) {
        double *sdist = malloc((size_t)n * sizeof(double));
        memcpy(sdist, dist_out, (size_t)n * sizeof(double));
        for (int iter = 0; iter < n; iter++) {
            double best = INFINITY;
            int bu = -1;
            for (int i = 0; i < n; i++) {
                double di = sdist[i];
                if (di < best) { best = di; bu = i; }
            }
            if (bu < 0) break;
            sdist[bu] = INFINITY;
            double du = dist_out[bu];
            for (int e = off[bu]; e < off[bu + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    sdist[v] = nd;
                }
            }
        }
        free(sdist);
    } else {
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
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    hpush(heap, &hs, nd, v);
                }
            }
        }
        free(done); free(heap);
    }

    free(off); free(edst); free(ew);
}
