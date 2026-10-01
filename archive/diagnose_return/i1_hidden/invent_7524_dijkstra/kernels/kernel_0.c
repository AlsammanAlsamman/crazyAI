#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define DJ_NBMAX 16384   /* slab-table size, multiple of 64 */

/* circular search for the next lit slab, 64 slabs per word test */
static int dj_scan(const uint64_t *bm, int nw, int p)
{
    int w = p >> 6, b = p & 63;
    uint64_t x = bm[w] >> b;
    if (x) return (int)__builtin_ctzll(x);
    int sk = 64 - b;
    for (int i = 1; i <= nw; ++i) {
        int ww = w + i; if (ww >= nw) ww -= nw;
        if (bm[ww]) return sk + (int)__builtin_ctzll(bm[ww]);
        sk += 64;
    }
    return -1;
}

/* guarded fallback: textbook lazy binary-heap Dijkstra on the same CSR */
static void dj_heap(int n, const int *off, const int *eto, const double *ew,
                    int source, double *dist)
{
    int cap = n + 64, hs = 0;
    double *hk = (double*)malloc((size_t)cap * sizeof(double));
    int    *hv = (int*)   malloc((size_t)cap * sizeof(int));
    if (!hk || !hv) { free(hk); free(hv); return; }
    for (int i = 0; i < n; ++i) dist[i] = INFINITY;
    dist[source] = 0.0;
    hk[hs] = 0.0; hv[hs] = source; ++hs;
    while (hs > 0) {
        double bk = hk[0]; int bu = hv[0];
        --hs;
        if (hs > 0) {
            double k = hk[hs]; int v = hv[hs]; int i = 0;
            for (;;) {
                int c = 2*i + 1; if (c >= hs) break;
                if (c + 1 < hs && hk[c+1] < hk[c]) ++c;
                if (hk[c] >= k) break;
                hk[i] = hk[c]; hv[i] = hv[c]; i = c;
            }
            hk[i] = k; hv[i] = v;
        }
        if (bk > dist[bu]) continue;
        int e1 = off[bu+1];
        for (int e = off[bu]; e < e1; ++e) {
            int v = eto[e];
            double nd = bk + ew[e];
            if (nd < dist[v]) {
                dist[v] = nd;
                if (hs == cap) {
                    int nc = cap + (cap >> 1) + 64;
                    double *t1 = (double*)realloc(hk, (size_t)nc * sizeof(double));
                    if (!t1) { free(hk); free(hv); return; }
                    hk = t1;
                    int *t2 = (int*)realloc(hv, (size_t)nc * sizeof(int));
                    if (!t2) { free(hk); free(hv); return; }
                    hv = t2; cap = nc;
                }
                int i = hs++;
                while (i > 0) {
                    int par = (i - 1) >> 1;
                    if (hk[par] <= nd) break;
                    hk[i] = hk[par]; hv[i] = hv[par]; i = par;
                }
                hk[i] = nd; hv[i] = v;
            }
        }
    }
    free(hk); free(hv);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;   /* every square crossed out */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;                                /* the one touched jewel */
    if (m <= 0) return;

    /* ---- the net: CSR, plus the cord-length statistics, in one pass ---- */
    int    *off = (int*)   calloc((size_t)n + 1, sizeof(int));
    int    *cur = (int*)   malloc((size_t)n * sizeof(int));
    int    *eto = (int*)   malloc((size_t)m * sizeof(int));
    double *ew  = (double*)malloc((size_t)m * sizeof(double));
    if (!off || !cur || !eto || !ew) { free(off); free(cur); free(eto); free(ew); return; }

    double wmax = 0.0, wsum = 0.0, wminp = INFINITY;
    for (int i = 0; i < m; ++i) {
        unsigned s = (unsigned)src[i], t = (unsigned)dst[i];
        if (s < (unsigned)n && t < (unsigned)n) off[s + 1]++;
        double w = weight[i];
        if (w > wmax) wmax = w;
        wsum += w;
        if (w > 0.0 && w < wminp) wminp = w;
    }
    for (int i = 0; i < n; ++i) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; ++i) {
        unsigned s = (unsigned)src[i], t = (unsigned)dst[i];
        if (s < (unsigned)n && t < (unsigned)n) {
            int p = cur[s]++; eto[p] = (int)t; ew[p] = weight[i];
        }
    }

    /* every cord of length zero: one flash lights all reachable jewels at 0 */
    if (!(wmax > 0.0)) {
        int *stk = cur, top = 0;
        stk[top++] = source;
        while (top) {
            int u = stk[--top], e1 = off[u+1];
            for (int e = off[u]; e < e1; ++e) {
                int v = eto[e];
                if (dist_out[v] != 0.0) { dist_out[v] = 0.0; stk[top++] = v; }
            }
        }
        free(off); free(cur); free(eto); free(ew);
        return;
    }
    free(cur);

    /* ---- slab width: how finely the clock is read ---- */
    double dneed = wmax / (double)(DJ_NBMAX - 128);       /* coarsest window that spans wmax */
    double dheur = (wsum / (double)m) * ((double)n / (double)m); /* mean cord / mean degree */
    double delta = wminp;                                  /* ideal: exact order, no re-flare */
    if (dneed > delta) {
        delta = (dneed < dheur) ? dneed : dheur;
        if (delta < wminp) delta = wminp;
    }
    if (!(delta > 0.0) || !isfinite(delta)) delta = 1.0;

    long long need = (long long)(wmax / delta) + 2;
    if (need > DJ_NBMAX - 128) need = DJ_NBMAX - 128;
    int nb = (int)(((need + 63) / 64 + 2) * 64);
    if (nb < 192) nb = 192;
    if (nb > DJ_NBMAX) nb = DJ_NBMAX;
    int nw = nb >> 6;
    long long lim = (long long)nb - 65;
    double invd = 1.0 / delta;

    int      *bhead = (int*)     malloc((size_t)nb * sizeof(int));
    uint64_t *bm    = (uint64_t*)calloc((size_t)nw, sizeof(uint64_t));
    int      *nxt   = (int*)     malloc((size_t)n * sizeof(int));
    int      *prv   = (int*)     malloc((size_t)n * sizeof(int));
    int      *inb   = (int*)     malloc((size_t)n * sizeof(int));
    if (!bhead || !bm || !nxt || !prv || !inb) {
        if (bhead && bm && nxt && prv && inb) { /*unreachable*/ }
        dj_heap(n, off, eto, ew, source, dist_out);
        free(bhead); free(bm); free(nxt); free(prv); free(inb);
        free(off); free(eto); free(ew);
        return;
    }
    for (int i = 0; i < nb; ++i) bhead[i] = -1;
    memset(inb, 0xFF, (size_t)n * sizeof(int));            /* -1 : in no slab */

#define DJ_INS(V,B) do { int _v=(V), _b=(B); int _h=bhead[_b];                 \
        nxt[_v]=_h; prv[_v]=-1; if(_h>=0) prv[_h]=_v; bhead[_b]=_v;            \
        inb[_v]=_b; bm[_b>>6] |= (uint64_t)1<<(_b&63); } while(0)
#define DJ_REM(V) do { int _v=(V), _b=inb[_v], _p=prv[_v], _q=nxt[_v];         \
        if(_p>=0) nxt[_p]=_q;                                                  \
        else { bhead[_b]=_q; if(_q<0) bm[_b>>6] &= ~((uint64_t)1<<(_b&63)); }  \
        if(_q>=0) prv[_q]=_p; inb[_v]=-1; } while(0)

    long long kcur = 0, pops = 0, budget = 4*(long long)n + 65536;
    int nwin = 1, bail = 0;
    DJ_INS(source, 0);

    while (nwin > 0) {
        int p = (int)(kcur % nb);
        if (bhead[p] < 0) {                                /* crouch: skip dark slabs */
            int sk = dj_scan(bm, nw, p);
            if (sk < 0) break;
            kcur += sk;
            p = (int)(kcur % nb);
        }
        while (bhead[p] >= 0) {                            /* drain this instant */
            int u = bhead[p], q = nxt[u];
            bhead[p] = q;
            if (q >= 0) prv[q] = -1;
            else bm[p>>6] &= ~((uint64_t)1<<(p&63));
            inb[u] = -1; --nwin;
            if (++pops > budget) { bail = 1; break; }      /* guard: re-flare storm */
            double du = dist_out[u];
            int e1 = off[u+1];
            for (int e = off[u]; e < e1; ++e) {            /* fire every cord at once */
                int v = eto[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {                    /* first/earlier light only */
                    dist_out[v] = nd;
                    long long bi = (long long)(nd * invd);
                    if (bi < kcur) bi = kcur;
                    else if (bi > kcur + lim) bi = kcur + lim;
                    int nbk = (int)(bi % nb), ob = inb[v];
                    if (ob != nbk) {
                        if (ob >= 0) { DJ_REM(v); } else ++nwin;
                        DJ_INS(v, nbk);
                    }
                }
            }
        }
        if (bail) break;
        ++kcur;
    }
#undef DJ_INS
#undef DJ_REM

    if (bail) dj_heap(n, off, eto, ew, source, dist_out);

    free(bhead); free(bm); free(nxt); free(prv); free(inb);
    free(off); free(eto); free(ew);
}
