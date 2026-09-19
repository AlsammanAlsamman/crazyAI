#include <stdlib.h>
#include <math.h>
#include <string.h>

#define HD 4

static inline void heap_push(double * __restrict hd, int * __restrict hu, int *hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) / HD;
        if (hd[p] <= d) break;
        hd[i] = hd[p]; hu[i] = hu[p];
        i = p;
    }
    hd[i] = d; hu[i] = u;
}

static inline void heap_pop(double * __restrict hd, int * __restrict hu, int *hs, double *outd, int *outu) {
    *outd = hd[0]; *outu = hu[0];
    int n = --(*hs);
    double d = hd[n]; int u = hu[n];
    int i = 0;
    for (;;) {
        int first = HD * i + 1;
        if (first >= n) break;
        int s = first;
        double sval = hd[first];
        int last = first + HD; if (last > n) last = n;
        for (int c = first + 1; c < last; c++) {
            if (hd[c] < sval) { sval = hd[c]; s = c; }
        }
        if (sval >= d) break;
        hd[i] = hd[s]; hu[i] = hu[s];
        i = s;
    }
    hd[i] = d; hu[i] = u;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    int *off  = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *edst = (int*)malloc((size_t)m * sizeof(int));
    double *ew = (double*)malloc((size_t)m * sizeof(double));
    {
        int *cnt = (int*)calloc((size_t)n, sizeof(int));
        for (int i = 0; i < m; i++) cnt[src[i]]++;
        off[0] = 0;
        for (int i = 0; i < n; i++) off[i + 1] = off[i] + cnt[i];
        int *cur = (int*)malloc((size_t)n * sizeof(int));
        memcpy(cur, off, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = cur[u]++;
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
        free(cnt); free(cur);
    }

    double log2n = log2((double)n + 2.0);
    double cost_dense = (double)n * (double)n;
    double cost_heap  = ((double)n + (double)m) * log2n * 1.5;
    int use_dense = cost_dense < cost_heap;

    const int * __restrict Poff = off;
    const int * __restrict Pedst = edst;
    const double * __restrict Pew = ew;

    if (use_dense) {
        char *done = (char*)calloc((size_t)n, 1);
        double * __restrict D = dist_out;
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int i = 0; i < n; i++) {
                if (!done[i] && D[i] < best) { best = D[i]; u = i; }
            }
            if (u < 0) break;
            done[u] = 1;
            double du = D[u];
            int e0 = Poff[u], e1 = Poff[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = Pedst[e];
                double nd = du + Pew[e];
                if (nd < D[v]) D[v] = nd;
            }
        }
        free(done);
    } else {
        char *done = (char*)calloc((size_t)n, 1);
        double *hd = (double*)malloc((size_t)(m + 2) * sizeof(double));
        int *hu = (int*)malloc((size_t)(m + 2) * sizeof(int));
        int hs = 0;
        heap_push(hd, hu, &hs, 0.0, source);
        while (hs > 0) {
            double d; int u;
            heap_pop(hd, hu, &hs, &d, &u);
            if (done[u]) continue;
            done[u] = 1;
            double du = dist_out[u];
            int e0 = Poff[u], e1 = Poff[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = Pedst[e];
                if (done[v]) continue;
                double nd = du + Pew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    heap_push(hd, hu, &hs, nd, v);
                }
            }
        }
        free(done); free(hd); free(hu);
    }

    free(off); free(edst); free(ew);
}
