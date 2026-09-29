#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef struct { double d; int v; } HItem;

static void hpush(HItem **hp, int *hcap, int *hs, double d, int v)
{
    if (*hs == *hcap) {
        int nc = (*hcap) * 2;
        HItem *t = (HItem *)realloc(*hp, (size_t)nc * sizeof(HItem));
        *hp = t; *hcap = nc;
    }
    {
        HItem *h = *hp;
        int i = (*hs)++;
        while (i > 0) {
            int p = (i - 1) >> 1;
            if (h[p].d <= d) break;
            h[i] = h[p]; i = p;
        }
        h[i].d = d; h[i].v = v;
    }
}

static HItem hpop(HItem *h, int *hs)
{
    HItem top = h[0];
    int sz = --(*hs);
    if (sz > 0) {
        HItem last = h[sz];
        int i = 0;
        for (;;) {
            int l = 2 * i + 1, r;
            if (l >= sz) break;
            r = l + 1;
            if (r < sz && h[r].d < h[l].d) l = r;
            if (h[l].d >= last.d) break;
            h[i] = h[l]; i = l;
        }
        h[i] = last;
    }
    return top;
}

/* smallest index q in [p,B) whose bit is set, else -1  (B = nw0*64) */
static int nextset(const uint64_t *l0, const uint64_t *l1, const uint64_t *l2,
                   int nw0, int nw1, int nw2, int p)
{
    int w0 = p >> 6;
    uint64_t m = l0[w0] & (~(uint64_t)0 << (p & 63));
    if (m) return (w0 << 6) + __builtin_ctzll(m);
    {
        int i = w0 + 1;
        if (i < nw0) {
            int w1 = i >> 6;
            uint64_t m1 = l1[w1] & (~(uint64_t)0 << (i & 63));
            if (m1) {
                int wi = (w1 << 6) + __builtin_ctzll(m1);
                return (wi << 6) + __builtin_ctzll(l0[wi]);
            }
            {
                int j = w1 + 1;
                if (j < nw1) {
                    int w2 = j >> 6;
                    uint64_t m2 = l2[w2] & (~(uint64_t)0 << (j & 63));
                    while (!m2) { if (++w2 >= nw2) return -1; m2 = l2[w2]; }
                    {
                        int wj = (w2 << 6) + __builtin_ctzll(m2);
                        int wi = (wj << 6) + __builtin_ctzll(l1[wj]);
                        return (wi << 6) + __builtin_ctzll(l0[wi]);
                    }
                }
            }
        }
    }
    return -1;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    int i;
    if (n <= 0) return;
    for (i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---------- CSR (counting sort by src), weight stats folded in ---------- */
    {
    int *off = (int *)calloc((size_t)n + 1, sizeof(int));
    int *pos, *eto;
    double *ew;
    double wmax = 0.0, wmin = INFINITY;
    int haszero = 0;

    for (i = 0; i < m; i++) off[src[i] + 1]++;
    for (i = 0; i < n; i++) off[i + 1] += off[i];
    pos = (int *)malloc((size_t)n * sizeof(int));
    memcpy(pos, off, (size_t)n * sizeof(int));
    eto = (int *)malloc((size_t)m * sizeof(int));
    ew  = (double *)malloc((size_t)m * sizeof(double));
    for (i = 0; i < m; i++) {
        int u = src[i];
        int p = pos[u]++;
        double w = weight[i];
        eto[p] = dst[i];
        ew[p]  = w;
        if (w > wmax) wmax = w;
        if (w > 0.0) { if (w < wmin) wmin = w; } else haszero = 1;
    }
    free(pos);

    /* ---------- degenerate: every bridge is instantaneous ---------- */
    if (!(wmax > 0.0)) {
        int *stk = (int *)malloc((size_t)n * sizeof(int));
        unsigned char *vis = (unsigned char *)calloc((size_t)n, 1);
        int sp = 0;
        stk[sp++] = source; vis[source] = 1;
        while (sp > 0) {
            int u = stk[--sp], e;
            for (e = off[u]; e < off[u + 1]; e++) {
                int y = eto[e];
                if (!vis[y]) { vis[y] = 1; dist_out[y] = 0.0; stk[sp++] = y; }
            }
        }
        free(stk); free(vis); free(off); free(eto); free(ew);
        return;
    }

    /* ---------- basket geometry ---------- */
    {
    long long tgt = 2LL * (long long)n + 64;
    int B0 = 64, B = 64, nw0, nw1, nw2, orderfree;
    double dall, delta, inv;
    long long span, curv = 0, qsize = 0, maskl;
    int *bhead, *bnext, *bprev, *inb, *cb;
    unsigned char *done;
    uint64_t *l0, *l1, *l2;
    HItem *heap; int hcap = 1024, hs = 0;

    if (tgt < 1024) tgt = 1024;
    if (tgt > (1 << 21)) tgt = (1 << 21);
    while ((long long)B0 < tgt) B0 <<= 1;

    dall = wmax / (double)(B0 - 2);
    if (wmin >= dall) { delta = wmin; orderfree = haszero ? 0 : 1; }
    else              { delta = dall; orderfree = 0; }
    inv = 1.0 / delta;

    span = (long long)(wmax * inv) + 2;
    while ((long long)B < span) B <<= 1;
    if (B > B0) B = B0;
    maskl = (long long)(B - 1);

    nw0 = B >> 6;
    nw1 = (nw0 + 63) >> 6;
    nw2 = (nw1 + 63) >> 6;

    bhead = (int *)malloc((size_t)B * sizeof(int));
    memset(bhead, 0xFF, (size_t)B * sizeof(int));
    bnext = (int *)malloc((size_t)n * sizeof(int));
    bprev = (int *)malloc((size_t)n * sizeof(int));
    inb   = (int *)malloc((size_t)n * sizeof(int));
    for (i = 0; i < n; i++) inb[i] = -1;
    done  = (unsigned char *)calloc((size_t)n, 1);
    cb    = (int *)malloc((size_t)n * sizeof(int));
    l0 = (uint64_t *)calloc((size_t)nw0, 8);
    l1 = (uint64_t *)calloc((size_t)nw1, 8);
    l2 = (uint64_t *)calloc((size_t)nw2, 8);
    heap = (HItem *)malloc((size_t)hcap * sizeof(HItem));

#define BSET(ix) do { int i_ = (ix); \
        l0[i_ >> 6]  |= (uint64_t)1 << (i_ & 63); \
        l1[i_ >> 12] |= (uint64_t)1 << ((i_ >> 6) & 63); \
        l2[i_ >> 18] |= (uint64_t)1 << ((i_ >> 12) & 63); } while (0)

#define BCLR(ix) do { int i_ = (ix), w_ = i_ >> 6; \
        l0[w_] &= ~((uint64_t)1 << (i_ & 63)); \
        if (!l0[w_]) { int x_ = w_ >> 6; \
            l1[x_] &= ~((uint64_t)1 << (w_ & 63)); \
            if (!l1[x_]) l2[x_ >> 6] &= ~((uint64_t)1 << (x_ & 63)); } } while (0)

    /* move node y's tag to the basket matching key nd (already stored in dist) */
#define PLACE(y, nd) do { \
        long long vi_ = (long long)((nd) * inv); \
        int nb_, ob_; \
        if (vi_ < curv) vi_ = curv; \
        nb_ = (int)(vi_ & maskl); \
        ob_ = inb[y]; \
        if (!(ob_ == nb_ && !(!orderfree && vi_ == curv))) { \
            if (ob_ >= 0) { \
                int pp_ = bprev[y], qq_ = bnext[y]; \
                if (pp_ >= 0) bnext[pp_] = qq_; \
                else { bhead[ob_] = qq_; if (qq_ < 0) BCLR(ob_); } \
                if (qq_ >= 0) bprev[qq_] = pp_; \
                inb[y] = -1; qsize--; \
            } \
            if (!orderfree && vi_ == curv) { \
                hpush(&heap, &hcap, &hs, (nd), (y)); \
            } else { \
                int hh_ = bhead[nb_]; \
                bnext[y] = hh_; bprev[y] = -1; \
                if (hh_ >= 0) bprev[hh_] = (y); else BSET(nb_); \
                bhead[nb_] = (y); inb[y] = nb_; qsize++; \
            } \
        } } while (0)

    bhead[0] = source; bnext[source] = -1; bprev[source] = -1; inb[source] = 0;
    qsize = 1; BSET(0);

    while (qsize > 0) {
        int b = (int)(curv & maskl);
        int k = 0, v;
        if (bhead[b] < 0) {                       /* walk to the next full basket */
            int q = nextset(l0, l1, l2, nw0, nw1, nw2, b);
            if (q < 0) q = nextset(l0, l1, l2, nw0, nw1, nw2, 0);
            if (q < 0) break;
            curv += (long long)((unsigned)(q - b) & (unsigned)maskl);
            b = q;
        }
        for (v = bhead[b]; v != -1; v = bnext[v]) { cb[k++] = v; inb[v] = -1; }
        bhead[b] = -1; BCLR(b); qsize -= k;

        if (orderfree) {
            int t;
            for (t = 0; t < k; t++) {
                int u = cb[t], e, ee;
                double du;
                if (done[u] || inb[u] >= 0) continue;
                done[u] = 1;
                du = dist_out[u];
                e = off[u]; ee = off[u + 1];
                for (; e < ee; e++) {
                    int y = eto[e];
                    double nd;
                    if (e + 2 < ee) __builtin_prefetch(&dist_out[eto[e + 2]], 1, 1);
                    nd = du + ew[e];
                    if (nd < dist_out[y] && !done[y]) { dist_out[y] = nd; PLACE(y, nd); }
                }
            }
        } else {
            int t;
            hs = 0;
            for (t = 0; t < k; t++) { int v2 = cb[t]; hpush(&heap, &hcap, &hs, dist_out[v2], v2); }
            while (hs > 0) {
                HItem top = hpop(heap, &hs);
                int u = top.v, e, ee;
                double du = top.d;
                if (done[u] || du != dist_out[u]) continue;
                done[u] = 1;
                e = off[u]; ee = off[u + 1];
                for (; e < ee; e++) {
                    int y = eto[e];
                    double nd;
                    if (e + 2 < ee) __builtin_prefetch(&dist_out[eto[e + 2]], 1, 1);
                    nd = du + ew[e];
                    if (nd < dist_out[y] && !done[y]) { dist_out[y] = nd; PLACE(y, nd); }
                }
            }
        }
    }

#undef PLACE
#undef BSET
#undef BCLR
    free(bhead); free(bnext); free(bprev); free(inb); free(done); free(cb);
    free(l0); free(l1); free(l2); free(heap);
    }
    free(off); free(eto); free(ew);
    }
}
