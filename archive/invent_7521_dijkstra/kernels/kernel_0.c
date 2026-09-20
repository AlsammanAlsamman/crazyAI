#include <stdlib.h>
#include <math.h>

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

/* Garden-nightingale array-scan Dijkstra: O(n^2 + m).
   Literal mapping:
     stone       = node
     letter      = tentative distance, mirrored in key[] which is set to
                   +INF the instant a stone is locked (so the nightingale
                   only ever "circles the unlocked stones")
     thread      = directed edge out of the most-recently-locked stone
     thief price = weight[] of that edge, asked one direction at a time
     lock        = key[u] = INFINITY, dist_out[u] final, never touched again
                   (no visited[] array needed: non-negative weights mean a
                   locked letter can never be beaten later - "no thief
                   charges a negative toll")
     dropped threads = relaxations that fail the improvement test; skipped,
                   exactly like unreachable stones staying +INF forever
*/
static void dijkstra_array_scan(int n, const int *restrict off,
                                 const int *restrict edst,
                                 const double *restrict ew,
                                 int source, double *restrict dist_out,
                                 double *restrict key, int use_omp) {
    for (int i = 0; i < n; i++) { dist_out[i] = INFINITY; key[i] = INFINITY; }
    dist_out[source] = 0.0;
    key[source] = 0.0;

    for (int iter = 0; iter < n; iter++) {
        double best = INFINITY;
        if (use_omp) {
            #pragma omp parallel for reduction(min:best)
            for (int i = 0; i < n; i++) if (key[i] < best) best = key[i];
        } else {
            for (int i = 0; i < n; i++) if (key[i] < best) best = key[i];
        }
        if (best == INFINITY) break; /* rest is bramble/tide: unreachable */

        int u = -1;
        for (int i = 0; i < n; i++) { if (key[i] == best) { u = i; break; } }
        if (u < 0) break;

        key[u] = INFINITY; /* locked: rebuilt one last time, never again */

        double du = dist_out[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; key[v] = nd; }
            /* else: thread dropped and thrown away */
        }
    }
}

static void dijkstra_heap(int n, const int *restrict off,
                           const int *restrict edst, const double *restrict ew,
                           int m, int source, double *restrict dist_out) {
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

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;

    /* CSR build, shared by both paths (a road is only ever cast from the
       stone that owns it, one direction at a time). */
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

    /* Guard against the array-scan's own stated weakness (O(n^2) blows up
       on large sparse graphs): estimate both costs, pick the cheaper,
       falling back to the well-known binary-heap method when large+sparse. */
    double logn = log2((double)n + 2.0);
    double cost_heap  = ((double)n + (double)m) * logn;
    double cost_array = (double)n * (double)n + (double)m;

    if (cost_array <= cost_heap) {
        double *key = malloc((size_t)n * sizeof(double));
        /* Vectorization first: the scan is split so the min-value pass is a
           clean SIMD reduction. Thread parallelism only added, and only for
           that reduction, when n is large enough per scan (n>=20000) for it
           to pay for itself; serial fallback otherwise (guarded, per spec). */
        int use_omp = (n >= 20000);
        dijkstra_array_scan(n, off, edst, ew, source, dist_out, key, use_omp);
        free(key);
    } else {
        dijkstra_heap(n, off, edst, ew, m, source, dist_out);
    }

    free(deg); free(off); free(edst); free(ew); free(fill);
}
