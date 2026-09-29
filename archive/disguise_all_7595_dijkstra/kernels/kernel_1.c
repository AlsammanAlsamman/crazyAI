#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Fused edge record: one random 16B store on build, one stream on scan. */
typedef struct { double w; int v; int pad_; } DJEdge;
/* Fused node record: distance + shelf position share one cache line. */
typedef struct { double d; int pos; int pad_; } DJNode;

#define DJ_NEW  (-1)
#define DJ_DONE (-2)

/* "slide the house forward on the shelf" : decrease-key / insert, 4-ary sift-up */
static inline void dj_up(double *restrict hk, int *restrict hv,
                         DJNode *restrict ni, int i, double k, int u)
{
    while (i > 0) {
        int p = (i - 1) >> 2;
        double kp = hk[p];
        if (kp <= k) break;
        int vp = hv[p];
        hk[i] = kp; hv[i] = vp; ni[vp].pos = i;
        i = p;
    }
    hk[i] = k; hv[i] = u; ni[u].pos = i;
}

/* restore the shelf after grabbing the front item : 4-ary sift-down */
static inline void dj_down(double *restrict hk, int *restrict hv,
                           DJNode *restrict ni, int hs, double k, int u)
{
    int i = 0;
    for (;;) {
        int c = (i << 2) + 1;
        if (c >= hs) break;
        int best; double bk;
        if (c + 3 < hs) {                     /* full 4 children: branchless tournament */
            double k0 = hk[c], k1 = hk[c + 1], k2 = hk[c + 2], k3 = hk[c + 3];
            int s1 = (k1 < k0); double a = s1 ? k1 : k0; int ia = c + s1;
            int s2 = (k3 < k2); double b = s2 ? k3 : k2; int ib = c + 2 + s2;
            int s3 = (b < a);   bk = s3 ? b : a;         best = s3 ? ib : ia;
        } else {
            best = c; bk = hk[c];
            for (int j = c + 1; j < hs; j++) {
                double t = hk[j];
                if (t < bk) { bk = t; best = j; }
            }
        }
        if (!(bk < k)) break;
        int vb = hv[best];
        hk[i] = bk; hv[i] = vb; ni[vb].pos = i;
        i = best;
    }
    hk[i] = k; hv[i] = u; ni[u].pos = i;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (m < 0) m = 0;

    size_t nn = (size_t)n;
    size_t mm = (size_t)(m > 0 ? m : 1);

    int    *off = (int *)   malloc((nn + 1) * sizeof(int));
    DJNode *ni  = (DJNode *)malloc(nn * sizeof(DJNode));
    double *hk  = (double *)malloc(nn * sizeof(double));
    int    *hv  = (int *)   malloc(nn * sizeof(int));
    DJEdge *adj = (DJEdge *)malloc(mm * sizeof(DJEdge));

    if (!off || !ni || !hk || !hv || !adj) {          /* degenerate safety net */
        for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
        if (source >= 0 && source < n) dist_out[source] = 0.0;
        free(off); free(ni); free(hk); free(hv); free(adj);
        return;
    }

    int nthr = 1;
#ifdef _OPENMP
    nthr = omp_get_max_threads();
    if (nthr > 8) nthr = 8;
    if (nthr < 1) nthr = 1;
#endif

    int par_build = 0;
#ifdef _OPENMP
    if (nthr > 1 && m >= (1 << 21) &&
        (double)nthr * (double)nn * 4.0 <= 2.5e8) par_build = 1;
#endif

    /* ---------------- CSR construction ---------------- */
#ifdef _OPENMP
    if (par_build) {
        int *hist = (int *)malloc((size_t)nthr * nn * sizeof(int));
        if (!hist) {
            par_build = 0;
        } else {
            #pragma omp parallel num_threads(nthr)
            {
                int t = omp_get_thread_num();
                int *h = hist + (size_t)t * nn;
                memset(h, 0, nn * sizeof(int));
                long long lo = (long long)m * t / nthr;
                long long hi = (long long)m * (t + 1) / nthr;
                for (long long i = lo; i < hi; i++) h[src[i]]++;
            }
            /* blocked cross-thread exclusive prefix: keeps T*4096 ints L2-resident.
               thread 0 owns the lowest edge indices => adjacency order == input order */
            {
                int run = 0;
                const int B = 4096;
                for (int b = 0; b < n; b += B) {
                    int be = b + B; if (be > n) be = n;
                    for (int u = b; u < be; u++) {
                        off[u] = run;
                        for (int t = 0; t < nthr; t++) {
                            int *p = hist + (size_t)t * nn + (size_t)u;
                            int c = *p; *p = run; run += c;
                        }
                    }
                }
                off[n] = run;
            }
            #pragma omp parallel num_threads(nthr)
            {
                int t = omp_get_thread_num();
                int *h = hist + (size_t)t * nn;
                long long lo = (long long)m * t / nthr;
                long long hi = (long long)m * (t + 1) / nthr;
                for (long long i = lo; i < hi; i++) {
                    int u = src[i];
                    int p = h[u]++;
                    adj[p].w = weight[i];
                    adj[p].v = dst[i];
                }
            }
            free(hist);
        }
    }
#endif
    if (!par_build) {
        memset(off, 0, (nn + 1) * sizeof(int));
        for (int i = 0; i < m; i++) off[src[i] + 1]++;
        for (int u = 0; u < n; u++) off[u + 1] += off[u];   /* off[u] = start(u) */
        for (int i = 0; i < m; i++) {                        /* off[] doubles as cursor */
            int p = off[src[i]]++;
            adj[p].w = weight[i];
            adj[p].v = dst[i];
        }
        for (int u = n; u >= 1; u--) off[u] = off[u - 1];    /* shift cursors back */
        off[0] = 0;
    }

    /* ---------------- scratch-guesses: everyone "no idea yet" ---------------- */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nthr) if(n >= 65536)
#endif
    for (int i = 0; i < n; i++) { ni[i].d = INFINITY; ni[i].pos = DJ_NEW; }

    if (source >= 0 && source < n) {
        ni[source].d = 0.0;
        ni[source].pos = 0;
        hk[0] = 0.0; hv[0] = source;
        int hs = 1;

        /* ---------------- nearest-to-farthest, one house at a time ---------------- */
        while (hs > 0) {
            int u = hv[0];
            double du = hk[0];
            ni[u].pos = DJ_DONE;                 /* carved in stone */
            hs--;
            if (hs > 0) dj_down(hk, hv, ni, hs, hk[hs], hv[hs]);

            int e  = off[u];
            int ee = off[u + 1];
            for (; e < ee; e++) {
                int pfi = e + 4; if (pfi >= ee) pfi = ee - 1;
                __builtin_prefetch(&ni[adj[pfi].v], 1, 1);

                int v = adj[e].v;
                double nd = du + adj[e].w;
                DJNode *V = ni + v;
                if (nd < V->d) {                 /* false for DONE nodes: weights >= 0 */
                    V->d = nd;
                    int p = V->pos;
                    if (p >= 0) {
                        dj_up(hk, hv, ni, p, nd, v);      /* slide forward on the shelf */
                    } else {
                        int idx = hs++;
                        dj_up(hk, hv, ni, idx, nd, v);    /* first time on the shelf */
                    }
                }
            }
        }
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nthr) if(n >= 65536)
#endif
    for (int i = 0; i < n; i++) dist_out[i] = ni[i].d;

    free(off); free(ni); free(hk); free(hv); free(adj);
}
