#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct { double d; int u; } HeapItem;

static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p;
    }
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

typedef struct { double val; int idx; } MinPair;

#pragma omp declare reduction(minpair : MinPair : \
    omp_out = (omp_in.val < omp_out.val ? omp_in : omp_out)) \
    initializer(omp_priv = (MinPair){INFINITY, -1})

/* Regime A ("chant team", the taught seed): dist_out[] is every knot's
   currently held breath. The knot holding the shortest undone gasp is
   found by a direct massed comparison (vectorized linear scan) — no
   heap is ever consulted during relaxation itself. Best for small n or
   densely veined cities. */
static void dense_dijkstra(int n, const int * restrict off, const int * restrict edst,
                            const double * restrict ew, int source, double * restrict dist_out,
                            char * restrict done) {
    for (int i = 0; i < n; i++) { dist_out[i] = INFINITY; done[i] = 0; }
    dist_out[source] = 0.0;

    for (int iter = 0; iter < n; iter++) {
        MinPair mp; mp.val = INFINITY; mp.idx = -1;
        #pragma omp simd reduction(minpair:mp)
        for (int i = 0; i < n; i++) {
            double di = done[i] ? INFINITY : dist_out[i];
            if (di < mp.val) { mp.val = di; mp.idx = i; }
        }
        int u = mp.idx;
        if (u < 0) break;
        double best = mp.val;
        done[u] = 1;
        int lo = off[u], hi = off[u + 1];
        #pragma omp simd
        for (int e = lo; e < hi; e++) {
            int v = edst[e];
            double nd = best + ew[e];
            if (nd < dist_out[v]) dist_out[v] = nd;
        }
    }
}

/* Regime B: classic binary-heap Dijkstra — the validated known
   technique, kept as the fallback for the regime the massed-chant scan
   does not cover: sparse, large graphs ("veins that run out past the
   last house into valleys no king ever built a wall for"), where an
   O(n^2) scan would burn far more breath than a hierarchy does. */
static void heap_dijkstra(int n, const int * restrict off, const int * restrict edst,
                           const double * restrict ew, int source, double * restrict dist_out,
                           char * restrict done, HeapItem * restrict heap) {
    for (int i = 0; i < n; i++) { dist_out[i] = INFINITY; done[i] = 0; }
    dist_out[source] = 0.0;
    int hs = 0;
    hpush(heap, &hs, 0.0, source);
    while (hs > 0) {
        HeapItem top = hpop(heap, &hs);
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        int lo = off[u], hi = off[u + 1];
        for (int e = lo; e < hi; e++) {
            int v = edst[e];
            double nd = dist_out[u] + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;

    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }

    /* Regime detection, taken literally from the metaphor: the native
       only ever taught the massed "chant team" scan; it is cheap (and
       vectorizes to many comparisons/cycle) when the city is small or
       densely veined, but wasteful once the city sprawls out sparsely.
       Compare the two costs and pick the cheaper regime at runtime,
       with the validated heap as fallback -- this addresses the known
       risk of O(n^2) blowing up on large, sparse graphs instead of
       shipping the risky part unguarded. */
    double log2n = log2((double)(n > 1 ? n : 2));
    double dense_cost = (double)n * (double)n / 16.0;
    double heap_cost   = (double)(n + m) * log2n * 2.0;
    int use_dense = dense_cost <= heap_cost;

    char *done = malloc((size_t)n);
    if (use_dense) {
        dense_dijkstra(n, off, edst, ew, source, dist_out, done);
    } else {
        HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
        heap_dijkstra(n, off, edst, ew, source, dist_out, done, heap);
        free(heap);
    }

    free(deg); free(off); free(edst); free(ew); free(fill); free(done);
}
