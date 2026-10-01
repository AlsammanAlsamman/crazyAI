#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================= abandoned-ritual fallback A: binary-heap Dijkstra =========
   "call in the hooded figure with the ranked ledger" - sparse regime.        */
typedef struct { double d; int u; } HItem;

static void dij_heap(int n, int m, const int *restrict off, const int *restrict edst,
                     const double *restrict ew, int source, double *restrict dist)
{
    unsigned char *done;
    HItem *h;
    int hs;
    int i;
    for (i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    done = (unsigned char *)calloc((size_t)n, 1);
    h = (HItem *)malloc(((size_t)m + 2) * sizeof(HItem));
    if (!done || !h) { free(done); free(h); return; }
    h[0].d = 0.0; h[0].u = source; hs = 1;
    while (hs > 0) {
        HItem top = h[0];
        int u, e, e1;
        double du;
        --hs;
        if (hs) {                       /* sift the last item down from the root */
            HItem x = h[hs];
            int j = 0;
            for (;;) {
                int l = 2 * j + 1, r, s;
                if (l >= hs) break;
                r = l + 1;
                s = (r < hs && h[r].d < h[l].d) ? r : l;
                if (h[s].d >= x.d) break;
                h[j] = h[s]; j = s;
            }
            h[j] = x;
        }
        u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        du = top.d;
        e1 = off[u + 1];
        for (e = off[u]; e < e1; e++) {
            int v = edst[e];
            double nd;
            if (done[v]) continue;
            nd = du + ew[e];
            if (nd < dist[v]) {
                HItem x; int j;
                dist[v] = nd;
                x.d = nd; x.u = v;
                j = hs++;
                while (j > 0) { int p = (j - 1) >> 1; if (h[p].d <= x.d) break; h[j] = h[p]; j = p; }
                h[j] = x;
            }
        }
    }
    free(done); free(h);
}

/* ================= abandoned-ritual fallback B: O(n^2) array-scan Dijkstra ===
   dense regime: no heap at all, the min sweep vectorizes.                    */
static void dij_scan(int n, const int *restrict off, const int *restrict edst,
                     const double *restrict ew, int source, double *restrict dist)
{
    unsigned char *done = (unsigned char *)calloc((size_t)n, 1);
    int it, i;
    if (!done) return;
    for (i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    for (it = 0; it < n; it++) {
        double best = INFINITY;
        int u = -1, e, e1;
        for (i = 0; i < n; i++) {                   /* branchless min-reduction */
            double d = done[i] ? INFINITY : dist[i];
            best = d < best ? d : best;
        }
        if (!(best < INFINITY)) break;
        for (i = 0; i < n; i++) if (!done[i] && dist[i] == best) { u = i; break; }
        if (u < 0) break;
        done[u] = 1;
        e1 = off[u + 1];
        for (e = off[u]; e < e1; e++) {
            int v = edst[e];
            double nd = best + ew[e];
            if (nd < dist[v]) dist[v] = nd;
        }
    }
    free(done);
}

/* ============================ THE RITUAL ====================================
   one ember per place; runners down every road of every lit, unsealed heap;
   one land-wide throw per round resolving each heap's own shortest sliver;
   a garrison on every heap whose ember did not move.                         */
void kernel(int n, int m, const int *restrict src, const int *restrict dst,
            const double *restrict weight, int source, double *restrict dist_out)
{
    int *ibuf, *off, *edst, *fro, *nxt, *stamp, *cur;
    double *ew;
    int i, fn, round, bail, lg, t;
    unsigned long long work, budget;

    if (n <= 0) return;
    for (i = 0; i < n; i++) dist_out[i] = INFINITY;   /* cold ash everywhere   */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;                           /* the banked ember      */
    if (m <= 0) return;

    /* one arena for every integer structure, one for the notches */
    ibuf = (int *)malloc(((size_t)m + 4u * (size_t)n + 8u) * sizeof(int));
    ew   = (double *)malloc((size_t)m * sizeof(double));
    if (!ibuf || !ew) { free(ibuf); free(ew); return; }
    off   = ibuf;
    edst  = off  + ((size_t)n + 1);
    fro   = edst + (size_t)m;
    nxt   = fro  + (size_t)n;
    stamp = nxt  + (size_t)n;

    /* roads laid out by their home place (CSR) */
    memset(off, 0, ((size_t)n + 1) * sizeof(int));
    for (i = 0; i < m; i++) off[src[i] + 1]++;
    for (i = 0; i < n; i++) off[i + 1] += off[i];
    cur = fro;                                        /* scratch cursor, n ints */
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (i = 0; i < m; i++) {
        int p = cur[src[i]]++;
        edst[p] = dst[i];
        ew[p]   = weight[i];
    }
    for (i = 0; i < n; i++) stamp[i] = -1;

    /* "count the roads against the places": how long may the ritual run before
       it is abandoned?  ~ the cost of the ranked-ledger way, so we can never
       lose by more than the restart.                                         */
    lg = 1; t = n; while (t > 1) { t >>= 1; lg++; }
    if (lg > 24) lg = 24;
    budget = ((unsigned long long)m + (unsigned long long)n) * (unsigned long long)(lg + 1);
    work = 0;

    fn = 0; fro[fn++] = source;
    round = 0; bail = 0;

    while (fn > 0) {
        int nn = 0, k;
        for (k = 0; k < fn; k++) {                    /* runners down every road */
            int u = fro[k];
            double du = dist_out[u];
            int e0 = off[u], e1 = off[u + 1], e;
            work += (unsigned long long)(unsigned int)(e1 - e0);
            for (e = e0; e < e1; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                double dv;
                if (e + 8 < e1) __builtin_prefetch(&dist_out[edst[e + 8]], 1, 1);
                dv = dist_out[v];
                /* the throw, resolved locally per heap: shortest sliver wins,
                   every longer sliver is burned.  A heap already holding a
                   shorter ember is garrisoned and refuses entry.             */
                if (nd < dv) {
                    dist_out[v] = nd;
                    if (stamp[v] != round) { stamp[v] = round; nxt[nn++] = v; }
                }
            }
        }
        if (work > budget) { bail = 1; break; }
        { int *tmp = fro; fro = nxt; nxt = tmp; }
        fn = nn;                 /* heaps whose ember did not move are sealed:
                                    their runners do not go out next round.   */
        round++;
    }

    if (bail) {                  /* the ash never caught: abandon the ritual   */
        double dn = (double)n;
        if ((double)m > 0.0625 * dn * dn) dij_scan(n, off, edst, ew, source, dist_out);
        else                              dij_heap(n, m, off, edst, ew, source, dist_out);
    }

    free(ibuf); free(ew);
}
