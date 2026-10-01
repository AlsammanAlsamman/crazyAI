#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------------------
   The native's two ways of reading the chalk grid.

   (A) "the whole grid fits under my hand" / "the net is drawn so tight that
       nearly every jewel touches nearly every other" -> sweep the eye across
       every square and pick the dimmest unlit one. Plain O(n^2) array scan,
       no drawers, no heap. Chosen for tiny or dense graphs.

   (B) otherwise: file each glint in the drawer of the instant it will arrive.
       Short cords flash at once (light edges, resolved inside the current
       drawer); long cords keep the glint waiting on the way (heavy edges,
       deferred to a later drawer). No priority structure is ever consulted;
       a jewel fires the moment it first lights, settled or not.
   --------------------------------------------------------------------------- */

/* (A) sweep the grid by eye -- dense / tiny regime */
static void chalk_sweep(int n, const int *restrict off, const int *restrict edst,
                        const double *restrict ew, double *restrict dist)
{
    double *dtmp = (double *)malloc((size_t)n * sizeof(double));
    unsigned char *lit = (unsigned char *)calloc((size_t)n, 1);
    if (!dtmp || !lit) { free(dtmp); free(lit); return; }
    for (int j = 0; j < n; j++) dtmp[j] = dist[j];
    for (int it = 0; it < n; it++) {
        double best = INFINITY;
        for (int j = 0; j < n; j++) { double d = dtmp[j]; best = d < best ? d : best; }
        if (!(best < INFINITY)) break;              /* only dark squares remain */
        int u = -1;
        for (int j = 0; j < n; j++) if (dtmp[j] == best) { u = j; break; }
        if (u < 0) break;
        dtmp[u] = INFINITY; lit[u] = 1;
        double du = dist[u];
        const int e1 = off[u + 1];
        for (int e = off[u]; e < e1; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (!lit[v] && nd < dist[v]) { dist[v] = nd; dtmp[v] = nd; }
        }
    }
    free(dtmp); free(lit);
}

/* one glint filed into a drawer */
static inline int bpush(int **ba, int *bs, int *bc, int bi, int v)
{
    if (bs[bi] == bc[bi]) {
        int nc = bc[bi] ? bc[bi] * 2 : 16;
        int *p = (int *)realloc(ba[bi], (size_t)nc * sizeof(int));
        if (!p) return 0;
        ba[bi] = p; bc[bi] = nc;
    }
    ba[bi][bs[bi]++] = v;
    return 1;
}

void kernel(int n, int m, const int *restrict src, const int *restrict dst,
            const double *restrict weight, int source, double *restrict dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;   /* every square blank & crossed */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;                               /* the one touched jewel */
    if (m <= 0) return;

    /* ---- the pace of the glint: how long a cord must be before it is "slack" ---- */
    double wmax = 0.0, wsum = 0.0;
    for (int i = 0; i < m; i++) { double w = weight[i]; wsum += w; if (w > wmax) wmax = w; }
    double dlt;
    if (!(wmax > 0.0)) dlt = 1.0;
    else {
        double wmean  = wsum / (double)m;
        double avgdeg = (double)m / (double)n; if (avgdeg < 1.0) avgdeg = 1.0;
        dlt = 2.0 * wmean / avgdeg;                       /* ~ one cord-length per jewel */
        double lo1 = wmean / 64.0, lo2 = wmax / 1024.0;   /* keep the drawer-stack short */
        if (dlt < lo1) dlt = lo1;
        if (dlt < lo2) dlt = lo2;
        if (dlt > wmax) dlt = wmax;
        if (!(dlt > 0.0)) dlt = 1.0;
    }
    /* how many drawers must stay open at once: a glint can wait at most wmax/dlt */
    int nbneed = (int)(wmax / dlt) + 3;
    int NB = 8; while (NB < nbneed && NB < 4096) NB <<= 1;
    if (nbneed > NB) { dlt = wmax / (double)(NB - 3); if (!(dlt > 0.0)) dlt = 1.0; }
    const long long NBm = (long long)NB - 1;
    const double inv = 1.0 / dlt;

    /* ---- the net, threaded once: CSR with the short cords of each jewel first ---- */
    int    *off  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int    *lcnt = (int *)calloc((size_t)n, sizeof(int));
    int    *fl   = (int *)calloc((size_t)n, sizeof(int));
    int    *fh   = (int *)malloc((size_t)n * sizeof(int));
    int    *lend = (int *)malloc((size_t)n * sizeof(int));
    int    *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !lcnt || !fl || !fh || !lend || !edst || !ew) {
        free(off); free(lcnt); free(fl); free(fh); free(lend); free(edst); free(ew);
        return;
    }
    for (int i = 0; i < m; i++) { int u = src[i]; fl[u]++; if (weight[i] < dlt) lcnt[u]++; }
    off[0] = 0;
    for (int u = 0; u < n; u++) off[u + 1] = off[u] + fl[u];
    for (int u = 0; u < n; u++) { lend[u] = off[u] + lcnt[u]; fl[u] = off[u]; fh[u] = lend[u]; }
    for (int i = 0; i < m; i++) {
        int u = src[i]; double w = weight[i];
        int pos = (w < dlt) ? fl[u]++ : fh[u]++;
        edst[pos] = dst[i]; ew[pos] = w;
    }
    free(lcnt); free(fl); free(fh);

    /* ---- which reading of the grid? ---- */
    int use_sweep = (n <= 256) ||
                    ((double)m >= 0.10 * (double)n * (double)n && n <= 8192);
    if (use_sweep) {
        chalk_sweep(n, off, edst, ew, dist_out);
        free(off); free(lend); free(edst); free(ew);
        return;
    }

    /* ---- the drawers of waiting glints ---- */
    int **ba = (int **)calloc((size_t)NB, sizeof(int *));
    int  *bs = (int *)calloc((size_t)NB, sizeof(int));
    int  *bc = (int *)calloc((size_t)NB, sizeof(int));
    int  *inb = (int *)calloc((size_t)n, sizeof(int));   /* drawer+1 of latest glint, 0 = none */
    int  *heavy = (int *)malloc((size_t)n * sizeof(int));
    int  *hst = (int *)malloc((size_t)n * sizeof(int));
    if (!ba || !bs || !bc || !inb || !heavy || !hst) {
        if (ba) { for (int b = 0; b < NB; b++) free(ba[b]); free(ba); }
        free(bs); free(bc); free(inb); free(heavy); free(hst);
        chalk_sweep(n, off, edst, ew, dist_out);          /* safe fallback */
        free(off); free(lend); free(edst); free(ew);
        return;
    }
    for (int j = 0; j < n; j++) hst[j] = -1;

    long long queued = 0, i = 0;
    int phase = 0;
    if (bpush(ba, bs, bc, 0, source)) { inb[source] = 1; queued = 1; }

#define FLARE(Y, ND)                                                          \
    do {                                                                      \
        int y_ = (Y); double nd_ = (ND);                                      \
        if (nd_ < dist_out[y_]) {                                             \
            dist_out[y_] = nd_;                     /* chalk the square */     \
            long long g = (long long)(nd_ * inv);   /* drawer of arrival */    \
            if (g < i) g = i;                                                 \
            if (g > i + NBm) g = i + NBm;                                     \
            int bj = (int)(g & NBm);                                          \
            if (inb[y_] != bj + 1) {                                          \
                if (bpush(ba, bs, bc, bj, y_)) { inb[y_] = bj + 1; queued++; }\
            }                                                                 \
        }                                                                     \
    } while (0)

    while (queued > 0) {
        int bi = (int)(i & NBm);
        if (bs[bi] == 0) { i++; continue; }              /* no glint due this instant */
        phase++;
        int hcnt = 0, p = 0;

        /* short cords flash at once: keep draining this drawer, including what
           the draining itself puts back into it. No settlement test anywhere. */
        while (p < bs[bi]) {
            int v = ba[bi][p++];
            queued--;
            if (inb[v] != bi + 1) continue;              /* a later, slacker flare: thrown away */
            inb[v] = 0;
            if (hst[v] != phase) { hst[v] = phase; heavy[hcnt++] = v; }
            double dv = dist_out[v];
            const int le = lend[v];
            for (int e = off[v]; e < le; e++) FLARE(edst[e], dv + ew[e]);
        }
        bs[bi] = 0;

        /* long cords kept their glints waiting; now they are let go */
        for (int k = 0; k < hcnt; k++) {
            int v = heavy[k];
            double dv = dist_out[v];
            const int e1 = off[v + 1];
            for (int e = lend[v]; e < e1; e++) FLARE(edst[e], dv + ew[e]);
        }

        if (bs[bi] == 0) i++;    /* only leave the instant when nothing is due in it */
    }
#undef FLARE

    for (int b = 0; b < NB; b++) free(ba[b]);
    free(ba); free(bs); free(bc); free(inb); free(heavy); free(hst);
    free(off); free(lend); free(edst); free(ew);
}
