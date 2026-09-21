#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { double d; int u; } HeapItem;

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    const int * restrict rsrc = src;
    const int * restrict rdst = dst;
    const double * restrict rw = weight;

    /* Build CSR (unavoidable, O(n+m)) */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[rsrc[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    {
        int *fill = calloc((size_t)n, sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = rsrc[i];
            int pos = off[u] + fill[u]++;
            edst[pos] = rdst[i];
            ew[pos] = rw[i];
        }
        free(fill);
    }
    free(deg);

    const int * restrict roff = off;
    const int * restrict redst = edst;
    const double * restrict rew = ew;

    /* Adaptive choice: O(n^2) array scan for dense graphs, heap for sparse */
    double density = (double)m / ((double)n * (double)n + 1.0);
    int use_dense = (n > 1) && (density > 0.0625);

    if (use_dense) {
        double * restrict key = malloc((size_t)n * sizeof(double));
        memcpy(key, dist_out, (size_t)n * sizeof(double));

        for (int iter = 0; iter < n; iter++) {
            double best = INFINITY;
            int u = -1;
            for (int v = 0; v < n; v++) {
                double kv = key[v];
                if (kv < best) { best = kv; u = v; }
            }
            if (u < 0 || best == INFINITY) break;
            key[u] = INFINITY;
            double du = dist_out[u];
            int end = roff[u + 1];
            for (int e = roff[u]; e < end; e++) {
                int v = redst[e];
                double nd = du + rew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    key[v] = nd;
                }
            }
        }
        free(key);
    } else {
        HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        char *done = calloc((size_t)n, 1);

        heap[0].d = 0.0; heap[0].u = source; hs = 1;

        while (hs > 0) {
            HeapItem top = heap[0];
            hs--;
            heap[0] = heap[hs];
            {
                int i = 0;
                while (1) {
                    int l = 2 * i + 1, r = 2 * i + 2, s = i;
                    if (l < hs && heap[l].d < heap[s].d) s = l;
                    if (r < hs && heap[r].d < heap[s].d) s = r;
                    if (s == i) break;
                    HeapItem t = heap[s]; heap[s] = heap[i]; heap[i] = t;
                    i = s;
                }
            }
            int u = top.u;
            if (done[u]) continue;
            done[u] = 1;
            double du = dist_out[u];
            int end = roff[u + 1];
            for (int e = roff[u]; e < end; e++) {
                int v = redst[e];
                double nd = du + rew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    int i = hs++;
                    heap[i].d = nd; heap[i].u = v;
                    while (i > 0) {
                        int p = (i - 1) / 2;
                        if (heap[p].d <= heap[i].d) break;
                        HeapItem t = heap[p]; heap[p] = heap[i]; heap[i] = t;
                        i = p;
                    }
                }
            }
        }
        free(done);
        free(heap);
    }

    free(off); free(edst); free(ew);
}
