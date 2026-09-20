#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---------- Fallback: binary-heap Dijkstra (sparse / large graphs) ---------- */
typedef struct { double d; int u; } HeapItem;
static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) / 2; if (h[p].d <= h[i].d) break; HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    while (1) {
        int l = 2*i+1, r = 2*i+2, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}

static void dijkstra_heap(int n, int m, int source, double *dist_out,
                           const int *off, const int *edst, const double *ew) {
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

/* ---------- Main mechanism: the "lattice" Dijkstra, no priority structure ----------
   Lattice of old memory     -> dist_out[], one fixed slot per place (SEED 3)
   Cow's-head re-measurement -> direct array write relaxing roads out of the
                                 just-settled board, no heap ever touched (SEED 2)
   Quietest unsettled note   -> plain linear scan of the whole lattice, done once
                                 per settling, not once per relaxation (SEED 3)
   Settled boards never revisited; note==infinity forever is the blank desert. */
static void dijkstra_lattice(int n, int source, double *dist_out,
                              const int *off, const int *edst, const double *ew) {
    char *settled = calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    for (int step = 0; step < n; step++) {
        double best = INFINITY;
        int u = -1;
        const char * restrict st = settled;
        const double * restrict dp = dist_out;
        for (int i = 0; i < n; i++) {
            if (!st[i] && dp[i] < best) { best = dp[i]; u = i; }
        }
        if (u < 0 || best == INFINITY) break; /* no nightingale sings: rest is desert */
        settled[u] = 1;

        int lo = off[u], hi = off[u + 1];
        const int * restrict ed = edst;
        const double * restrict ewp = ew;
        double du = dist_out[u];
        for (int e = lo; e < hi; e++) {
            int v = ed[e];
            double cand = du + ewp[e];
            if (cand < dist_out[v]) dist_out[v] = cand;
        }
    }
    free(settled);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    /* Build CSR adjacency once, shared by whichever mechanism runs. */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    size_t msafe = (size_t)(m > 0 ? m : 1);
    int *edst = malloc(msafe * sizeof(int));
    double *ew = malloc(msafe * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }

    /* Guard against the mechanism's own named risk: the O(n^2) lattice scan
       loses to the heap on large, sparse graphs. Use it only where the task's
       own "known way" footnote says it wins -- small or dense graphs --
       and fall back to the heap otherwise. */
    long long m64 = (long long)m, n64 = (long long)n;
    int dense_or_small = (n <= 2000) || (m64 >= 8LL * n64);

    if (dense_or_small) dijkstra_lattice(n, source, dist_out, off, edst, ew);
    else                 dijkstra_heap(n, m, source, dist_out, off, edst, ew);

    free(deg); free(off); free(edst); free(ew); free(fill);
}
