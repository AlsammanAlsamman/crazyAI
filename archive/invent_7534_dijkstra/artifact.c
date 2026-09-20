#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { double d; int u; } HeapItem;

static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) / 4;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p;
    }
}

static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    while (1) {
        int c0 = 4 * i + 1, s = i;
        for (int k = 0; k < 4; k++) { int c = c0 + k; if (c < *hs && h[c].d < h[s].d) s = c; }
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    const int PAR_THRESHOLD = 200000;

    int *deg = calloc((size_t)n, sizeof(int));
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));

    if (m >= PAR_THRESHOLD) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            #pragma omp atomic update
            deg[src[i]]++;
        }
#else
        for (int i = 0; i < m; i++) deg[src[i]]++;
#endif
    } else {
        for (int i = 0; i < m; i++) deg[src[i]]++;
    }

    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];

    int *cursor = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) cursor[i] = off[i];

    if (m >= PAR_THRESHOLD) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos;
            #pragma omp atomic capture
            { pos = cursor[u]; cursor[u]++; }
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
#else
        for (int i = 0; i < m; i++) {
            int u = src[i]; int pos = cursor[u]++;
            edst[pos] = dst[i]; ew[pos] = weight[i];
        }
#endif
    } else {
        for (int i = 0; i < m; i++) {
            int u = src[i]; int pos = cursor[u]++;
            edst[pos] = dst[i]; ew[pos] = weight[i];
        }
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
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }

    free(deg); free(off); free(edst); free(ew); free(cursor); free(done); free(heap);
}
