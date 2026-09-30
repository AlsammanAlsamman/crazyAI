#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================================================================
   THE NET, literally:
     jewel               -> node
     cord                -> directed edge (CSR)
     length of the cord  -> weight = the delay the glint suffers on it
     glint               -> one relaxation in flight, carrying an arrival time
     chalk square        -> dist_out[] slot
     chalked number      -> the FIRST (hence shortest) arrival time
     one chalk tick      -> a time-slice of width delta (a bucket)
     blank + scratch     -> INFINITY
   A jewel re-emits the instant it is first lit.  Nothing is ever
   "settled" first; there is no priority structure anywhere.
   ================================================================== */

/* ---------- REGIME A: the grid is small / almost fully corded ----------
   The child sweeps her eyes over the WHOLE grid and takes the palest
   unlit square.  No cords waiting, no ticks: a plain O(n^2) glance,
   whose inner sweep is a flat, cache-resident, unrolled min-reduction. */
static void sweep_whole_grid(int n,
                            const int *restrict off,
                            const int *restrict edst,
                            const double *restrict ew,
                            int source,
                            double *restrict dist)
{
    double *restrict cand = (double *)malloc((size_t)n * sizeof(double));
    if (!cand) return;
    for (int i = 0; i < n; i++) { dist[i] = INFINITY; cand[i] = INFINITY; }
    dist[source] = 0.0;
    cand[source] = 0.0;

    for (int it = 0; it < n; it++) {
        double b0 = INFINITY, b1 = INFINITY, b2 = INFINITY, b3 = INFINITY;
        int i = 0;
        for (; i + 4 <= n; i += 4) {                 /* one glance, 4 lanes */
            double x0 = cand[i], x1 = cand[i + 1], x2 = cand[i + 2], x3 = cand[i + 3];
            if (x0 < b0) b0 = x0;
            if (x1 < b1) b1 = x1;
            if (x2 < b2) b2 = x2;
            if (x3 < b3) b3 = x3;
        }
        for (; i < n; i++) { double x = cand[i]; if (x < b0) b0 = x; }
        if (b1 < b0) b0 = b1;
        if (b3 < b2) b2 = b3;
        if (b2 < b0) b0 = b2;
        if (!(b0 < INFINITY)) break;                 /* only dark squares left */

        int u = 0;
        while (cand[u] != b0) u++;                   /* the palest square */
        cand[u] = INFINITY;

        double du = dist[u];
        int e1 = off[u + 1];
        for (int e = off[u]; e < e1; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; cand[v] = nd; }
        }
    }
    free(cand);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (m < 0) m = 0;
    if (source < 0 || source >= n) {
        for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
        return;
    }

    /* ---- thread every cord into the net, and learn the cords' lengths ---- */
    size_t ma = (size_t)(m > 0 ? m : 1);
    int    *pos  = (int *)calloc((size_t)n + 1, sizeof(int));
    int    *off  = (int *)malloc(((size_t)n + 1) * sizeof(int));
    int    *edst = (int *)malloc(ma * sizeof(int));
    double *ew   = (double *)malloc(ma * sizeof(double));
    if (!pos || !off || !edst || !ew) {
        free(pos); free(off); free(edst); free(ew);
        for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
        if (source >= 0 && source < n) dist_out[source] = 0.0;
        return;
    }

    double wmax = 0.0, wsum = 0.0;
    for (int i = 0; i < m; i++) {
        pos[src[i]]++;
        double w = weight[i];
        wsum += w;
        if (w > wmax) wmax = w;
    }
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + pos[i];
    for (int i = 0; i < n; i++) pos[i] = off[i];
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int p = pos[u]++;
        edst[p] = dst[i];
        ew[p]   = weight[i];
    }

    /* ---- which regime is the chalk grid in?  (runtime, in-metaphor) ---- */
    {
        double dnn = (double)n * (double)n;
        if (n <= 8192 && dnn <= 8.0 * (double)m + 8192.0) {
            sweep_whole_grid(n, off, edst, ew, source, dist_out);
            free(pos); free(off); free(edst); free(ew);
            return;
        }
    }

    /* ---- REGIME B: the ticking ring of time-slices ----
       delta = how much time one chalk tick covers.  Chosen so only about
       1/K of the cords are "short" (fire inside the tick): that keeps the
       in-tick re-lighting subgraph subcritical, so re-emission is rare. */
    double avgdeg = (m > 0) ? (double)m / (double)n : 1.0;
    int K = (int)(2.0 * avgdeg + 0.5);
    if (K < 8)   K = 8;
    if (K > 128) K = 128;

    double delta;
    if (wmax > 0.0) {
        delta = wmax / (double)K;
        double wmean = wsum / (double)(m > 0 ? m : 1);
        /* skewed cords (one enormous slack cord) must not widen the tick */
        if (wmean > 0.0 && delta > 4.0 * wmean) delta = 4.0 * wmean;
    } else {
        delta = 1.0;                 /* every cord taut to nothing */
    }

    long long need = (wmax > 0.0) ? (long long)(wmax / delta) + 4 : 2;
    int NB = 4;
    while ((long long)NB < need && NB < (1 << 14)) NB <<= 1;
    if ((long long)NB < need) { NB = 1 << 14; delta = wmax / (double)(NB - 4); }
    int    mask = NB - 1;
    double invd = 1.0 / delta;

    int **bd  = (int **)calloc((size_t)NB, sizeof(int *));   /* tick contents */
    int  *bs  = (int  *)calloc((size_t)NB, sizeof(int));     /* tick size     */
    int  *bc  = (int  *)calloc((size_t)NB, sizeof(int));     /* tick capacity */
    int  *qs  = (int  *)malloc((size_t)n * sizeof(int));     /* tick a jewel waits in, -1 = none */
    int   fcap = 256;
    int  *front = (int *)malloc((size_t)fcap * sizeof(int));

    if (!bd || !bs || !bc || !qs || !front) goto cleanup_fail;

    {
        double *restrict dist = dist_out;
        for (int i = 0; i < n; i++) { dist[i] = INFINITY; qs[i] = -1; }
        dist[source] = 0.0;

        long long pending = 0;

/* park a glint for jewel `nd_node` in tick slot `sl` */
#define PARK(nd_node, sl)                                                      \
        do {                                                                   \
            int _s = (sl);                                                     \
            if (bs[_s] == bc[_s]) {                                            \
                int _nc = bc[_s] ? bc[_s] * 2 : 64;                            \
                int *_p = (int *)realloc(bd[_s], (size_t)_nc * sizeof(int));    \
                if (!_p) goto cleanup;                                         \
                bd[_s] = _p; bc[_s] = _nc;                                     \
            }                                                                  \
            bd[_s][bs[_s]++] = (nd_node);                                      \
            qs[(nd_node)] = _s;                                                \
            pending++;                                                         \
        } while (0)

        PARK(source, 0);

        long long tick = 0;
        while (pending > 0) {
            int c = (int)(tick & (long long)mask);
            if (bs[c] == 0) { tick++; continue; }    /* nothing arrives this instant */

            /* drain this tick until it is empty, INCLUDING jewels re-lit
               inside it by short cords -- a jewel never waits to be settled */
            while (bs[c] > 0) {
                int cnt = bs[c];
                if (cnt > fcap) {
                    int nf = cnt + cnt / 2;
                    int *p = (int *)realloc(front, (size_t)nf * sizeof(int));
                    if (!p) goto cleanup;
                    front = p; fcap = nf;
                }
                memcpy(front, bd[c], (size_t)cnt * sizeof(int));
                bs[c] = 0;
                pending -= cnt;

                for (int t = 0; t < cnt; t++) {
                    int u = front[t];
                    if (qs[u] != c) continue;        /* a late flare: thrown away */
                    qs[u] = -1;
                    double du = dist[u];
                    int e1 = off[u + 1];
                    for (int e = off[u]; e < e1; e++) {
                        int v = edst[e];
                        double nd = du + ew[e];
                        if (nd < dist[v]) {          /* first/better light only */
                            dist[v] = nd;
                            int s = (int)((long long)(nd * invd) & (long long)mask);
                            if (qs[v] != s) PARK(v, s);
                        }
                    }
                }
            }
            tick++;
        }
#undef PARK
    }

cleanup:
    for (int i = 0; i < NB; i++) free(bd[i]);
    free(bd); free(bs); free(bc); free(qs); free(front);
    free(pos); free(off); free(edst); free(ew);
    return;

cleanup_fail:
    if (bd) { for (int i = 0; i < NB; i++) free(bd[i]); free(bd); }
    free(bs); free(bc); free(qs); free(front);
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    free(pos); free(off); free(edst); free(ew);
}
