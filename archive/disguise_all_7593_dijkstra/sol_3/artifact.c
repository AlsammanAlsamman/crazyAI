#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct { double w; int v; int pad; } CEdge;   /* 16 B: one stream per adjacency */
typedef struct { double d; int u; int pad; } HItem;   /* 16 B: 4 siblings == 1 cache line */

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    #pragma omp parallel for schedule(static) if(n >= 262144)
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;

    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- CSR build: count -> prefix -> destructive scatter -> shift ---- */
    int *off = (int *)malloc((size_t)(n + 2) * sizeof(int));
    CEdge *E = (CEdge *)malloc((size_t)m * sizeof(CEdge));
    if (!off || !E) { free(off); free(E); return; }

    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int u = 0; u < n; u++) off[u + 1] += off[u];   /* off[u] == start(u), off[n] == m */
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int p = off[u]++;
        E[p].v = dst[i];
        E[p].w = weight[i];
    }
    for (int u = n; u > 0; u--) off[u] = off[u - 1];    /* undo the destructive ++ */
    off[0] = 0;

    /* ---- Squeaky's box: 4-ary min-heap, sibling groups cache-line aligned ---- */
    size_t cap = (size_t)m + 8;
    void *hraw = malloc(cap * sizeof(HItem) + 64);
    if (!hraw) { free(off); free(E); return; }
    HItem *hbase = (HItem *)(((uintptr_t)hraw + 63u) & ~(uintptr_t)63u);
    HItem *h = hbase + 3;   /* children of i are 4i+1..4i+4 -> exactly one 64 B line */

    int hs = 1;
    h[0].d = 0.0; h[0].u = source;

    while (hs) {
        const double du = h[0].d;
        const int u = h[0].u;

        /* pop front, sift the last card down through the hole */
        if (--hs) {
            const HItem last = h[hs];
            const double ld = last.d;
            int i = 0;
            for (;;) {
                int c = (i << 2) + 1;
                int best; double bd;
                if (c + 3 < hs) {
                    bd = h[c].d; best = c;
                    double t1 = h[c + 1].d, t2 = h[c + 2].d, t3 = h[c + 3].d;
                    if (t1 < bd) { bd = t1; best = c + 1; }
                    if (t2 < bd) { bd = t2; best = c + 2; }
                    if (t3 < bd) { bd = t3; best = c + 3; }
                } else if (c < hs) {
                    bd = h[c].d; best = c;
                    for (int k = c + 1; k < hs; k++) {
                        double tk = h[k].d;
                        if (tk < bd) { bd = tk; best = k; }
                    }
                } else break;
                if (bd >= ld) break;
                h[i] = h[best];
                i = best;
            }
            h[i] = last;
        }

        if (du > dist_out[u]) continue;   /* lingering card: a better one already won */

        const int s = off[u], t = off[u + 1];
        if (t - s >= 6)
            for (int e = s; e < t; e++) __builtin_prefetch(&dist_out[E[e].v], 0, 1);

        for (int e = s; e < t; e++) {
            const int v = E[e].v;
            const double nd = du + E[e].w;
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                int i = hs++;                      /* hand Squeaky a new card */
                while (i) {                        /* slide past a few neighbours only */
                    int p = (i - 1) >> 2;
                    if (h[p].d <= nd) break;
                    h[i] = h[p];
                    i = p;
                }
                h[i].d = nd; h[i].u = v;
            }
        }
    }

    free(hraw); free(E); free(off);
}
