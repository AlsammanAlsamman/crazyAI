#include <stdlib.h>
#include <math.h>

/* ---- binary-heap fallback: identical mechanism to the known/reference
   O((n+m) log n) solution, used when the graph is large AND sparse,
   the exact regime where the stone/scan mechanism is known to lose. ---- */
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

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;

    /* ---- shared CSR build ("roads out of each house") ---- */
    int *deg  = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off  = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int    *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew   = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i]; int pos = off[u] + fill[u]++;
        edst[pos] = dst[i]; ew[pos] = weight[i];
    }

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    /* Guard against the mechanism's own named weakness: O(n^2) only when
       small or dense; otherwise fall back to the validated heap path. */
    long long nn = (long long)n * (long long)n;
    int use_array = (n <= 3000) || ((long long)m * 16 >= nn);

    if (use_array) {
        /* ---- stones-and-ducks world, literally ---- */
        double *scan = malloc((size_t)n * sizeof(double)); /* "thread lengths of ducks not yet stoned" */
        for (int i = 0; i < n; i++) scan[i] = INFINITY;
        scan[source] = 0.0;

        const int    *restrict offp  = off;
        const int    *restrict edstp = edst;
        const double *restrict ewp   = ew;
        double *restrict distp = dist_out;
        double *restrict scanp = scan;

        for (int round = 0; round < n; round++) {
            /* "let the wandering ducks settle" -> value-only reduction, vectorizes cleanly */
            double best = INFINITY;
            for (int i = 0; i < n; i++) { double s = scanp[i]; if (s < best) best = s; }
            if (best == INFINITY) break; /* rest of the grid the ducks never reach */

            /* "trace every shot back to its landing" -> find which corner holds it */
            int u = -1;
            for (int i = 0; i < n; i++) { if (scanp[i] == best) { u = i; break; } }

            scanp[u] = INFINITY; /* black stone: never reconsidered */
            double du = distp[u];

            for (int e = offp[u]; e < offp[u + 1]; e++) {
                int v = edstp[e];
                double nd = du + ewp[e];
                if (nd < distp[v]) { distp[v] = nd; scanp[v] = nd; }
            }
        }
        free(scan);
    } else {
        /* ---- large & sparse: known heap-based path, unmodified ---- */
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

    free(deg); free(off); free(edst); free(ew); free(fill);
}
