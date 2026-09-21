#include <stdlib.h>
#include <math.h>

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    /* Build CSR adjacency once; shared by both regimes. */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    {
        int *fill = calloc((size_t)n, sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = off[u] + fill[u]++;
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
        free(fill);
    }
    free(deg);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    /* Regime check mirroring the known_way's two named regimes:
       dense/small  -> "of every bird now perched and unclaimed I take the
                        one whose chant is shortest": sweep the whole field
                        each round, no priority structure at all (O(n^2)).
       sparse/large -> fall back to the validated binary-heap Dijkstra
                        (identical to the reference), guarding the risk
                        that an O(n^2) sweep is bad when the graph is big
                        and sparse. */
    double logn = log((double)n + 2.0) / log(2.0);
    double heap_cost = 3.0 * ((double)n + (double)m) * logn;
    int use_array_scan = ((double)n * (double)n) <= heap_cost;

    char *done = calloc((size_t)n, 1);

    if (use_array_scan) {
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] < best) { best = dist_out[v]; u = v; }
            }
            if (u < 0) break; /* rest of the field is the never-visited maze: leave blank, throw the thread away */
            done[u] = 1;
            double du = dist_out[u];
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
    } else {
        typedef struct { double d; int u; } HeapItem;
        HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        {
            int i = hs++; heap[i].d = 0.0; heap[i].u = source;
            while (i > 0) {
                int p = (i - 1) / 2;
                if (heap[p].d <= heap[i].d) break;
                HeapItem t = heap[p]; heap[p] = heap[i]; heap[i] = t; i = p;
            }
        }
        while (hs > 0) {
            HeapItem top = heap[0]; hs--; heap[0] = heap[hs];
            {
                int i = 0;
                while (1) {
                    int l = 2 * i + 1, r = 2 * i + 2, s = i;
                    if (l < hs && heap[l].d < heap[s].d) s = l;
                    if (r < hs && heap[r].d < heap[s].d) s = r;
                    if (s == i) break;
                    HeapItem t = heap[s]; heap[s] = heap[i]; heap[i] = t; i = s;
                }
            }
            int u = top.u;
            if (done[u]) continue;
            done[u] = 1;
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = dist_out[u] + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    int i = hs++; heap[i].d = nd; heap[i].u = v;
                    while (i > 0) {
                        int p = (i - 1) / 2;
                        if (heap[p].d <= heap[i].d) break;
                        HeapItem t = heap[p]; heap[p] = heap[i]; heap[i] = t; i = p;
                    }
                }
            }
        }
        free(heap);
    }

    free(done);
    free(off); free(edst); free(ew);
}
