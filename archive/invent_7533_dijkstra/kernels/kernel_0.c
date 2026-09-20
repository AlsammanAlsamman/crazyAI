#include <stdlib.h>
#include <math.h>

typedef struct { double d; int u; } Tally;

static void lift_in(Tally *pile, int *sz, double d, int u) {
    int i = (*sz)++;
    pile[i].d = d; pile[i].u = u;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (pile[p].d <= pile[i].d) break;
        Tally t = pile[p]; pile[p] = pile[i]; pile[i] = t;
        i = p;
    }
}

static Tally lift_out(Tally *pile, int *sz) {
    Tally top = pile[0];
    (*sz)--;
    pile[0] = pile[*sz];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < *sz && pile[l].d < pile[s].d) s = l;
        if (r < *sz && pile[r].d < pile[s].d) s = r;
        if (s == i) break;
        Tally t = pile[s]; pile[s] = pile[i]; pile[i] = t;
        i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    /* lay the harbor-board: count roads leaving each place, then pack them
       contiguous (CSR) so walking a place's roads touches one clean run
       of memory instead of scattering across the whole hull */
    int *out_deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) out_deg[src[i]]++;

    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + out_deg[i];

    int *road_dst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *road_w = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fillpos = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fillpos[u]++;
        road_dst[pos] = dst[i];
        road_w[pos] = weight[i];
    }
    free(fillpos);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n > 0) dist_out[source] = 0.0;

    char *done = calloc((size_t)n, 1);
    Tally *pile = malloc((size_t)(m + 2) * sizeof(Tally));
    int sz = 0;
    lift_in(pile, &sz, 0.0, source);

    while (sz > 0) {
        Tally top = lift_out(pile, &sz);
        int u = top.u;
        if (done[u]) continue;      /* stale tally: over the side */
        done[u] = 1;
        double du = dist_out[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = road_dst[e];
            if (done[v]) continue;
            double nd = du + road_w[e];
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                lift_in(pile, &sz, nd, v);
            }
        }
    }

    free(out_deg); free(off); free(road_dst); free(road_w);
    free(done); free(pile);
}
