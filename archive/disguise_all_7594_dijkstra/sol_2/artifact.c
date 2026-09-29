#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct { double d; int u; int pad; } HItem;

/* move item (key,u) up from slot i in a 4-ary heap */
static inline void hp_up(HItem *h, int *pos, int i, double key, int u)
{
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= key) break;
        h[i] = h[p];
        pos[h[i].u] = i;
        i = p;
    }
    h[i].d = key; h[i].u = u; pos[u] = i;
}

/* move item (key,u) down from slot i in a 4-ary heap of given size */
static inline void hp_down(HItem *h, int *pos, int size, int i, double key, int u)
{
    for (;;) {
        int c = (i << 2) + 1;
        if (c >= size) break;
        int lim = c + 4; if (lim > size) lim = size;
        int best = c; double bd = h[c].d;          /* the 4 siblings share one cache line */
        for (int j = c + 1; j < lim; j++) {
            double t = h[j].d;
            if (t < bd) { bd = t; best = j; }
        }
        if (bd >= key) break;
        h[i] = h[best];
        pos[h[i].u] = i;
        i = best;
    }
    h[i].d = key; h[i].u = u; pos[u] = i;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- CSR: count -> prefix -> scatter (cursor) -> shift back ---- */
    int *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !edst || !ew) { free(off); free(edst); free(ew); return; }

    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    for (int i = 0; i < m; i++) {
        int p = off[src[i]]++;
        edst[p] = dst[i];
        ew[p]  = weight[i];
    }
    for (int i = n; i > 0; i--) off[i] = off[i - 1];
    off[0] = 0;

    /* ---- indexed 4-ary heap, sibling groups on one 64B line ---- */
    int *pos = (int *)malloc((size_t)n * sizeof(int));
    void *hraw = malloc((size_t)(n + 4) * sizeof(HItem) + 64);
    if (!pos || !hraw) { free(off); free(edst); free(ew); free(pos); free(hraw); return; }
    HItem *harr = (HItem *)(((uintptr_t)hraw + 63u) & ~(uintptr_t)63u);
    HItem *h = harr + 3;   /* children 4i+1..4i+4 -> harr[4(i+1)..4(i+1)+3] : one cache line */

    for (int i = 0; i < n; i++) pos[i] = -1;

    int size = 1;
    h[0].d = 0.0; h[0].u = source; pos[source] = 0;

    while (size > 0) {
        double du = h[0].d;
        int u = h[0].u;
        pos[u] = -1;                       /* settled: can never re-enter (w >= 0) */
        size--;
        if (size > 0) hp_down(h, pos, size, 0, h[size].d, h[size].u);

        int e = off[u], e1 = off[u + 1];
        for (; e < e1; e++) {
            if (e + 4 < e1) __builtin_prefetch(&dist_out[edst[e + 4]], 1, 1);
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                int p = pos[v];
                if (p >= 0) hp_up(h, pos, p, nd, v);      /* decrease-key in place */
                else        hp_up(h, pos, size++, nd, v); /* first sighting: insert */
            }
        }
    }

    free(off); free(edst); free(ew); free(pos); free(hraw);
}
