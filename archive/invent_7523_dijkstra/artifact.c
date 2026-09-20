#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- heap-based Dijkstra: validated fallback for sparse/large graphs ---- */
typedef struct { double d; int u; } HeapItem;
static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) / 2; if (h[p].d <= h[i].d) break; HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}
static void heap_dijkstra(int n, int m, const int *off, const int *edst, const double *ew,
                           int source, double *dist_out) {
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
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = dist_out[u] + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }
    free(done); free(heap);
}

/* ---- round-synchronous "throw" (active-node Bellman-Ford, Seed 2) ---- */
void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) { int u = src[i]; int pos = off[u] + fill[u]++; edst[pos] = dst[i]; ew[pos] = weight[i]; }

    /* guard: round-based cost is O(rounds*m); rounds stays small only
       when the graph is dense or small (short hop-diameter). For
       sparse/large graphs, fall back to the validated heap method. */
    double avg_deg = (n > 0) ? (double)m / (double)n : 0.0;
    int use_rounds = (n <= 4000) || (avg_deg >= 8.0);

    if (!use_rounds) {
        heap_dijkstra(n, m, off, edst, ew, source, dist_out);
        free(deg); free(off); free(edst); free(ew); free(fill);
        return;
    }

    /* reverse CSR: row v = every incoming road (u, w) into heap v */
    int *rdeg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) rdeg[dst[i]]++;
    int *roff = malloc((size_t)(n + 1) * sizeof(int));
    roff[0] = 0;
    for (int i = 0; i < n; i++) roff[i + 1] = roff[i] + rdeg[i];
    int *rsrc = malloc((size_t)m * sizeof(int));
    double *rw = malloc((size_t)m * sizeof(double));
    int *rfill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) { int v = dst[i]; int pos = roff[v] + rfill[v]++; rsrc[pos] = src[i]; rw[pos] = weight[i]; }

    double *dist = malloc((size_t)n * sizeof(double));
    double *ndist = malloc((size_t)n * sizeof(double));
    char *changed = malloc((size_t)n);
    char *nchanged = malloc((size_t)n);
    for (int i = 0; i < n; i++) { dist[i] = INFINITY; changed[i] = 0; }
    dist[source] = 0.0;
    changed[source] = 1;

    int any_active = 1;
    int rounds_cap = n > 0 ? n - 1 : 0;
    for (int round = 0; round < rounds_cap && any_active; round++) {
        any_active = 0;
        #pragma omp parallel for schedule(static) reduction(||:any_active) if(n > 20000)
        for (int v = 0; v < n; v++) {
            double best = dist[v];
            int rs = roff[v], re = roff[v + 1];
            for (int e = rs; e < re; e++) {
                int u = rsrc[e];
                if (!changed[u]) continue;            /* runners stopped from a sealed source */
                double du = dist[u];
                if (du == INFINITY) continue;
                double cand = du + rw[e];
                if (cand < best) best = cand;
            }
            nchanged[v] = (char)(best < dist[v]);
            ndist[v] = best;
            if (nchanged[v]) any_active = 1;
        }
        double *td = dist; dist = ndist; ndist = td;
        char *tc = changed; changed = nchanged; nchanged = tc;
    }

    memcpy(dist_out, dist, (size_t)n * sizeof(double));

    free(deg); free(off); free(edst); free(ew); free(fill);
    free(rdeg); free(roff); free(rsrc); free(rw); free(rfill);
    free(dist); free(ndist); free(changed); free(nchanged);
}
