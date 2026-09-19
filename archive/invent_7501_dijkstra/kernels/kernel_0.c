#include <stdlib.h>
#include <math.h>

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    /* Group every raspberry-cane road by the square it leaves from (CSR). */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }

    /* dist_out: the true, permanent pouch count once a boulder is pressed.
       scan_dist: the thumb's own view of the pouches -- identical values,
       except a boulder-ed square is pushed to infinity so the thumb's
       hand-comparison never lands on it again. */
    double *scan_dist = malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) { dist_out[i] = INFINITY; scan_dist[i] = INFINITY; }
    dist_out[source] = 0.0;   /* the traveler's own square: empty pouch, costs nothing to leave */
    scan_dist[source] = 0.0;

    int pawn = source;

    for (int round = 0; round < n; round++) {
        int u = pawn;
        double du = dist_out[u];

        /* Loose the snake down every cane leading from the pawn's square. */
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double cand = du + ew[e];        /* cane's eggshells + pawn's pouch */
            if (cand < dist_out[v]) {        /* only a lighter delivery counts */
                dist_out[v] = cand;
                scan_dist[v] = cand;
            }
        }

        /* Press the boulder on the pawn's own square: fixed forever. */
        scan_dist[u] = INFINITY;

        /* Thumb sweeps every unboulered pouch for the lightest one. */
        double best = INFINITY;
        int best_i = -1;
        for (int i = 0; i < n; i++) {
            double di = scan_dist[i];
            if (di < best) { best = di; best_i = i; }
        }
        if (best_i < 0) break;   /* no reachable pouch left -- rest stay empty, off the paper */
        pawn = best_i;
    }

    free(deg); free(off); free(edst); free(ew); free(fill); free(scan_dist);
}
