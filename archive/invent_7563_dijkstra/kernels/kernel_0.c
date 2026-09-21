#include <stdlib.h>
#include <math.h>
#include <string.h>

#define DARY 4  /* d-ary heap: fewer levels than binary, better cache reuse per pop */

typedef struct { double d; int u; } HeapItem;

static inline void dheap_push(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) / DARY;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem dheap_pop(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    int last = --(*hs);
    h[0] = h[last];
    int i = 0;
    for (;;) {
        int base = DARY * i + 1;
        int s = i;
        for (int k = 0; k < DARY; k++) {
            int c = base + k;
            if (c < *hs && h[c].d < h[s].d) s = c;
        }
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    /* --- build CSR: every place's own bundle of outgoing ribs --- */
    int *off  = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    {
        int *deg = (int*)calloc((size_t)n, sizeof(int));
        for (int i = 0; i < m; i++) deg[src[i]]++;
        off[0] = 0;
        for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
        int *cursor = (int*)malloc((size_t)n * sizeof(int));
        memcpy(cursor, off, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = cursor[u]++;
            edst[pos] = dst[i];
            ew[pos]   = weight[i];
        }
        free(deg); free(cursor);
    }

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n > 0) dist_out[source] = 0.0;

    /* Regime check (Seed 1's "ordinary foreign business of the roads
       settling into what is known"): if courts are so richly connected
       (dense) or so few (small) that maintaining a swarm archive costs
       more than just looking at every place directly, don't build one. */
    double avg_out = (n > 0) ? (double)m / (double)n : 0.0;
    int use_dense = (n <= 256) || (avg_out > (double)n / 32.0);

    char *done = (char*)calloc((size_t)n, 1);

    if (use_dense) {
        /* ---- dense / small regime: no priority structure at all ---- */
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] < best) { best = dist_out[v]; u = v; }
            }
            if (u < 0) break; /* remaining places sealed forever: unreachable */
            done[u] = 1;
            double du = dist_out[u];
            int e0 = off[u], e1 = off[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
    } else {
        /* ---- sparse regime: swarm race via a d-ary priority heap ----
           Each relaxation dispatches a fresh swarm (heap entry) carrying
           cumulative melt. We crack whichever shell the least-melt swarm
           among all still-crawling swarms reaches; a swarm that later
           reaches an already-cracked shell is thrown away (lazy deletion),
           and a cracked shell's value is never overwritten. */
        HeapItem *heap = (HeapItem*)malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        dheap_push(heap, &hs, 0.0, source);
        while (hs > 0) {
            HeapItem top = dheap_pop(heap, &hs);
            int u = top.u;
            if (done[u]) continue;   /* spent for no purpose */
            done[u] = 1;             /* fixed shell stays fixed */
            double du = dist_out[u];
            int e0 = off[u], e1 = off[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    dheap_push(heap, &hs, nd, v);
                }
            }
        }
        free(heap);
    }

    free(done);
    free(off); free(edst); free(ew);
}
