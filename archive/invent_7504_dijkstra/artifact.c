#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    /* --- lay the cords: build CSR adjacency, one bucket of cords per slab --- */
    int *deg = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fill = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(fill);
    free(deg);

    /* --- every slab gets a bowl; grey stone (unsettled) is the default --- */
    char *settled = (char*)calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;

    /* the traveler's own bowl gets its white stone at zero cost, immediately */
    dist_out[source] = 0.0;

    /* --- the shadow-stick: one tick per slab to settle --- */
    for (int round = 0; round < n; round++) {

        /* look over every slab still holding a grey stone; no priority
           structure consulted -- a plain scan for the smallest chalk-candidate */
        int best_u = -1;
        double best_d = INFINITY;

#ifdef _OPENMP
        #pragma omp parallel if(n > 20000)
        {
            int local_u = -1;
            double local_d = INFINITY;
            #pragma omp for nowait schedule(static)
            for (int v = 0; v < n; v++) {
                if (!settled[v] && dist_out[v] < local_d) {
                    local_d = dist_out[v];
                    local_u = v;
                }
            }
            #pragma omp critical
            {
                if (local_d < best_d) { best_d = local_d; best_u = local_u; }
            }
        }
#else
        for (int v = 0; v < n; v++) {
            if (!settled[v] && dist_out[v] < best_d) {
                best_d = dist_out[v];
                best_u = v;
            }
        }
#endif

        /* no cord ever reached the remaining slabs: their bowls stay empty */
        if (best_u == -1) break;

        /* set the white stone: this place's distance is now proven true */
        settled[best_u] = 1;

        /* follow every cord out of the newly white-stoned slab; a cord that
           would name a longer count than the chalk already there is burned */
        int rs = off[best_u], re = off[best_u + 1];
        for (int e = rs; e < re; e++) {
            int v = edst[e];
            if (!settled[v]) {
                double nd = best_d + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
    }

    free(off);
    free(edst);
    free(ew);
    free(settled);
}
