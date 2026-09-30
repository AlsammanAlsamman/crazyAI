#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================================================================== *
 *  "the old road, walked jewel by jewel"                               *
 *  Kept ONLY as a guarded fallback for the regime the chalk row        *
 *  cannot hold (see wheel_ok / the patience budget below).             *
 * ==================================================================== */
typedef struct { double d; int u; } HItem;

static void hp_push(HItem *restrict h, int *restrict hs, double d, int u)
{
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d;
    h[i].u = u;
}

static void heap_dijkstra(int n, int m,
                          const int *restrict off, const int *restrict edst,
                          const double *restrict ew, int source,
                          double *restrict dist)
{
    HItem *h    = (HItem *)malloc((size_t)(m + 2) * sizeof(HItem));
    char  *done = (char  *)calloc((size_t)n, 1);
    int hs = 0;
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    if (!h || !done) { free(h); free(done); return; }
    hp_push(h, &hs, 0.0, source);
    while (hs > 0) {
        HItem top = h[0];
        int last = --hs;
        if (last > 0) {
            HItem mv = h[last];
            int i = 0;
            for (;;) {
                int l = 2 * i + 1;
                if (l >= last) break;
                int r = l + 1, s = l;
                if (r < last && h[r].d < h[l].d) s = r;
                if (h[s].d >= mv.d) break;
                h[i] = h[s];
                i = s;
            }
            h[i] = mv;
        }
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        double du = dist[u];
        int s0 = off[u], t0 = off[u + 1];
        for (int e = s0; e < t0; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; hp_push(h, &hs, nd, v); }
        }
    }
    free(h);
    free(done);
}

/* ==================================================================== *
 *  "if the grid is small enough to take in with one glance"             *
 *  Dense / tiny regime: no chalk row at all, sweep the whole grid.      *
 *  Contiguous double array + sentinel => auto-vectorizable min scan.    *
 * ==================================================================== */
static void glance_scan(int n,
                        const int *restrict off, const int *restrict edst,
                        const double *restrict ew, int source,
                        double *restrict dist)
{
    double *key = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    if (!key) { heap_dijkstra(n, off[n], off, edst, ew, source, dist); return; }
    for (int i = 0; i < n; i++) key[i] = INFINITY;
    key[source] = 0.0;
    for (int it = 0; it < n; it++) {
        double best = INFINITY;
        int u = -1;
        for (int i = 0; i < n; i++) {           /* one glance across the grid */
            double kv = key[i];
            if (kv < best) { best = kv; u = i; }
        }
        if (u < 0) break;                        /* sky empty: rest stays blank */
        key[u] = INFINITY;
        double du = dist[u];
        int s0 = off[u], t0 = off[u + 1];
        for (int e = s0; e < t0; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; key[v] = nd; }
        }
    }
    free(key);
}

/* ==================================================================== *
 *  THE NATIVE'S MECHANISM: glints in flight, a circular chalk row.      *
 *  Returns 1 if the crouch completed, 0 if patience ran out.           *
 * ==================================================================== */
static int wheel_run(int n, int m,
                     const int *restrict off, const int *restrict edst,
                     const double *restrict ew,
                     double wmax, double delta, int B,
                     int source, double *restrict dist)
{
    int    *bhead = (int    *)malloc((size_t)B * sizeof(int));     /* chalk row  */
    double *proc  = (double *)malloc((size_t)n * sizeof(double));  /* flared-at  */
    int ecap = m + 64;
    int *enode = (int *)malloc((size_t)ecap * sizeof(int));        /* glints     */
    int *enext = (int *)malloc((size_t)ecap * sizeof(int));
    if (!bhead || !proc || !enode || !enext) {
        free(bhead); free(proc); free(enode); free(enext);
        return 0;
    }
    for (int i = 0; i < B; i++) bhead[i] = -1;
    for (int i = 0; i < n; i++) proc[i] = INFINITY;

    const int  Bm   = B - 1;                 /* B is a power of two */
    const double dinv = 1.0 / delta;
    long pending = 1, k = 0, scans = 0;
    long budget  = 8L * (long)m + 32L * (long)n + 4096L;   /* patience */
    int  ecnt = 1, aborted = 0;

    enode[0] = source; enext[0] = -1; bhead[0] = 0;   /* touch the one jewel */

    while (pending > 0) {                 /* "I crouch, and I wait"          */
        int slot = (int)(k & (long)Bm);    /* the square the clock stands on  */
        int e = bhead[slot];
        while (e >= 0) {                  /* drain this instant to fixpoint  */
            bhead[slot] = enext[e];
            pending--;
            int u = enode[e];
            double du = dist[u];
            if (du < proc[u]) {           /* SEED 2: a late flare is dropped */
                proc[u] = du;
                int s0 = off[u], t0 = off[u + 1];
                scans += (long)(t0 - s0);
                for (int j = s0; j < t0; j++) {
                    if (j + 8 < t0) __builtin_prefetch(&dist[edst[j + 8]], 1, 1);
                    int v = edst[j];
                    double nd = du + ew[j];          /* arrival = flare + cord */
                    if (nd < dist[v]) {              /* first light only       */
                        dist[v] = nd;
                        long kb = (long)(nd * dinv); /* which future square    */
                        if (kb < k) kb = k;
                        int sl = (int)(kb & (long)Bm);
                        if (ecnt >= ecap) {
                            int nc = ecap + (ecap >> 1) + 64;
                            int *a = (int *)realloc(enode, (size_t)nc * sizeof(int));
                            if (a) enode = a;
                            int *b = (int *)realloc(enext, (size_t)nc * sizeof(int));
                            if (b) enext = b;
                            if (!a || !b) { aborted = 1; goto finish; }
                            ecap = nc;
                        }
                        enode[ecnt] = v;
                        enext[ecnt] = bhead[sl];
                        bhead[sl] = ecnt;            /* park the glint in time */
                        ecnt++;
                        pending++;
                    }
                }
                if (scans > budget) { aborted = 1; goto finish; }
            }
            e = bhead[slot];              /* glints landing on *this* instant */
        }
        k++;                              /* the clock moves one square on    */
    }
finish:
    free(bhead); free(proc); free(enode); free(enext);
    return aborted ? 0 : 1;
}

/* ==================================================================== */
void kernel(int n, int m, const int *restrict src, const int *restrict dst,
            const double *restrict weight, int source, double *restrict dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;   /* SEED 3: all blank */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- thread the net: CSR ---- */
    int    *off  = (int    *)calloc((size_t)n + 1, sizeof(int));
    int    *edst = (int    *)malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    int    *cur  = (int    *)malloc((size_t)n * sizeof(int));
    if (!off || !edst || !ew || !cur) {
        free(off); free(edst); free(ew); free(cur);
        return;
    }
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)n * sizeof(int));

    double wmax = 0.0, wsum = 0.0;
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int p = cur[u]++;
        double w = weight[i];
        edst[p] = dst[i];
        ew[p]   = w;
        wsum += w;
        if (w > wmax) wmax = w;
    }
    free(cur);

    /* ---- regime 1: a grid small enough for one glance ---- */
    if (n <= 512) {
        glance_scan(n, off, edst, ew, source, dist_out);
        free(off); free(edst); free(ew);
        return;
    }

    /* ---- choose the spacing of the chalk row ---- */
    double davg = (double)m / (double)n;
    if (davg < 1.0) davg = 1.0;
    double wmean = wsum / (double)m;
    double delta = (wmax > 0.0) ? (wmax / (2.0 * davg)) : 1.0;
    if (!(delta > 0.0) || !isfinite(delta)) delta = 1.0;

    long nbneed = (long)(wmax / delta) + 3;
    int  B = 8;
    while ((long)B < nbneed && B < (1 << 18)) B <<= 1;

    /* ---- regime 2 test: can the row hold these cords? ----
       If the longest cord dwarfs its kin, one chalk square would span many
       typical cords, the drains degenerate, and the old road is better.    */
    int wheel_ok = ((long)B >= nbneed) && (wmax <= 4.0 * davg * wmean);

    if (!wheel_ok || !wheel_run(n, m, off, edst, ew, wmax, delta, B,
                                source, dist_out))
        heap_dijkstra(n, m, off, edst, ew, source, dist_out);

    free(off); free(edst); free(ew);
}
