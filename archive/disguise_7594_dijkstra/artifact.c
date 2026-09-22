#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { double d; int u; } HeapItem;

static inline void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem hpop(HeapItem *h, int *hs) {
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

    /* --- Build CSR adjacency (counting sort, no separate fill[] array) --- */
    int *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];

    int *cursor = (int*)malloc((size_t)n * sizeof(int));
    memcpy(cursor, off, (size_t)n * sizeof(int));

    int *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cursor[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(cursor);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    char *done = (char*)calloc((size_t)n, 1);

    /* Choose strategy: array-scan O(n^2 + m) vs binary-heap O((n+m) log n).
       Crossover roughly where n^2 ~ m*log2(n)  =>  avg_deg ~ n/log2(n).
       Small n is cheap and cache-friendly as array-scan regardless. */
    double avg_deg = (double)m / (double)n;
    double log2n = n > 1 ? log2((double)n) : 1.0;
    int use_dense = (n <= 4096) || (avg_deg > (double)n / log2n);

    if (use_dense) {
        for (int iter = 0; iter < n; iter++) {
            double best = INFINITY;
            for (int v = 0; v < n; v++) {
                double dv = done[v] ? INFINITY : dist_out[v];
                if (dv < best) best = dv;
            }
            if (best == INFINITY) break;
            int u = -1;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] == best) { u = v; break; }
            }
            done[u] = 1;
            double du = dist_out[u];
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
    } else {
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
        free(heap);
    }

    free(off); free(edst); free(ew); free(done);
}
