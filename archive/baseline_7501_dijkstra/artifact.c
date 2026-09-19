#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { double d; int u; } HeapItem;

static inline void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}

static inline HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    int newsize = --(*hs);
    if (newsize > 0) {
        double d = h[newsize].d; int u = h[newsize].u;
        int i = 0;
        for (;;) {
            int l = 2 * i + 1;
            if (l >= newsize) break;
            int r = l + 1;
            int s = l;
            if (r < newsize && h[r].d < h[l].d) s = r;
            if (h[s].d >= d) break;
            h[i] = h[s];
            i = s;
        }
        h[i].d = d; h[i].u = u;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n <= 0) return;
    dist_out[source] = 0.0;
    if (n == 1) return;

    int *off  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *edst = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double *)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    {
        int *cnt = (int *)calloc((size_t)n, sizeof(int));
        for (int i = 0; i < m; i++) cnt[src[i]]++;
        off[0] = 0;
        for (int i = 0; i < n; i++) off[i + 1] = off[i] + cnt[i];
        memset(cnt, 0, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = off[u] + cnt[u]++;
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
        free(cnt);
    }

    double logn = log2((double)n + 2.0);
    int use_array = (n <= 2048) || ((double)m > 8.0 * (double)n * logn);

    if (use_array) {
        char *done = (char *)calloc((size_t)n, 1);
        for (int iter = 0; iter < n; iter++) {
            double best = INFINITY;
            int bu = -1;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] < best) { best = dist_out[v]; bu = v; }
            }
            if (bu < 0) break;
            done[bu] = 1;
            double du = dist_out[bu];
            for (int e = off[bu]; e < off[bu + 1]; e++) {
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
        free(heap); free(done);
    }

    free(off); free(edst); free(ew);
}
