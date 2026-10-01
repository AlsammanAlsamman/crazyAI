#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------------------------------------------------------------- *
 *  THE POND  : unordered bag of landed ducks; O(1) insert and
 *              decrease-key, extract-min by one SIMD sweep.
 *  THE NEST   : 4-ary heap ("long cousins", four to a cache line),
 *              used only if the pond swells past the break-even T.
 * ---------------------------------------------------------------- */

typedef struct { double d; int u; } Duck;

static inline void nest_push(Duck *h, int *hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) {                       /* hole-shifting: one write/level */
        int p = (i - 1) >> 2;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}

static inline Duck nest_pop(Duck *h, int *hs) {
    Duck top = h[0];
    int sz = --(*hs);
    if (sz <= 0) return top;
    Duck last = h[sz];
    int i = 0;
    for (;;) {
        int c = (i << 2) + 1;
        if (c >= sz) break;
        int e = c + 4; if (e > sz) e = sz;
        int b = c; double bd = h[c].d;
        for (int j = c + 1; j < e; j++) { double dj = h[j].d; if (dj < bd) { bd = dj; b = j; } }
        if (bd >= last.d) break;
        h[i] = h[b];
        i = b;
    }
    h[i] = last;
    return top;
}

/* the hand hovering over the pond: argmin of a flat double array */
static inline int pond_argmin(const double *restrict k, int n) {
#if defined(__AVX2__)
    if (n >= 8) {
        __m256d c0 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        __m256d c1 = _mm256_set_pd(7.0, 6.0, 5.0, 4.0);
        __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0;
        __m256d p0 = c0, p1 = c1;
        const __m256d st = _mm256_set1_pd(8.0);
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256d v0 = _mm256_loadu_pd(k + i);
            __m256d v1 = _mm256_loadu_pd(k + i + 4);
            __m256d l0 = _mm256_cmp_pd(v0, m0, _CMP_LT_OQ);
            __m256d l1 = _mm256_cmp_pd(v1, m1, _CMP_LT_OQ);
            m0 = _mm256_blendv_pd(m0, v0, l0);
            m1 = _mm256_blendv_pd(m1, v1, l1);
            p0 = _mm256_blendv_pd(p0, c0, l0);
            p1 = _mm256_blendv_pd(p1, c1, l1);
            c0 = _mm256_add_pd(c0, st);
            c1 = _mm256_add_pd(c1, st);
        }
        double mv[8], mp[8];
        _mm256_storeu_pd(mv,     m0); _mm256_storeu_pd(mv + 4, m1);
        _mm256_storeu_pd(mp,     p0); _mm256_storeu_pd(mp + 4, p1);
        double best = mv[0]; int bi = (int)mp[0];
        for (int j = 1; j < 8; j++) if (mv[j] < best) { best = mv[j]; bi = (int)mp[j]; }
        for (; i < n; i++) if (k[i] < best) { best = k[i]; bi = i; }
        return bi;
    }
#endif
    {
        double best = k[0]; int bi = 0;
        for (int i = 1; i < n; i++) if (k[i] < best) { best = k[i]; bi = i; }
        return bi;
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;      /* bare corners */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;                                  /* the white stone, written first */
    if (m <= 0) return;

    /* ---- the roads, laid out corner by corner (CSR) ---- */
    int *cnt = (int *)calloc((size_t)n + 1, sizeof(int));
    int *off = (int *)malloc(((size_t)n + 1) * sizeof(int));
    int *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew = (double *)malloc((size_t)m * sizeof(double));
    char *hasout = (char *)malloc((size_t)n);
    double *pkey = (double *)malloc((size_t)n * sizeof(double));
    int *pnode = (int *)malloc((size_t)n * sizeof(int));
    int *ppos = (int *)malloc((size_t)n * sizeof(int));
    if (!cnt || !off || !edst || !ew || !hasout || !pkey || !pnode || !ppos) {
        free(cnt); free(off); free(edst); free(ew);
        free(hasout); free(pkey); free(pnode); free(ppos);
        return;
    }
    for (int i = 0; i < m; i++) cnt[src[i]]++;
    {
        int s = 0;
        for (int i = 0; i < n; i++) {
            off[i] = s; s += cnt[i];
            hasout[i] = (char)(cnt[i] != 0);   /* "a house with a far side" */
            ppos[i] = -1;
        }
        off[n] = s;
    }
    memcpy(cnt, off, (size_t)n * sizeof(int));               /* reuse as fill cursor */
    for (int i = 0; i < m; i++) {
        int u = src[i], p = cnt[u]++;
        edst[p] = dst[i]; ew[p] = weight[i];
    }

    const int *restrict EO = off;
    const int *restrict ED = edst;
    const double *restrict EW = ew;
    double *restrict D = dist_out;

    /* ---- how wide may the pond grow before the nest is cheaper? ---- */
    int T;
    {
        double avg = (double)m / (double)n;
        double lg  = log2((double)n + 2.0);
        double td  = 10.0 * (avg + 1.0) * lg;
        if (td < 512.0) td = 512.0;
        if (td > (double)n + 1.0) td = (double)n + 1.0;       /* dense/small: never switch */
        T = (int)td;
    }

    int psz = 0;
    if (hasout[source]) { pkey[0] = 0.0; pnode[0] = source; ppos[source] = 0; psz = 1; }

    Duck *nest = NULL; char *done = NULL; int hs = 0, migrated = 0;

    /* ================= POND PHASE ================= */
    while (psz > 0) {
        if (psz > T) {                                       /* tip the pond into the nest */
            nest = (Duck *)malloc(((size_t)m + (size_t)n + 2) * sizeof(Duck));
            done = (char *)calloc((size_t)n, 1);
            if (!nest || !done) { free(nest); free(done); nest = NULL; done = NULL; T = n + 1; }
            else {
                for (int i = 0; i < psz; i++) nest_push(nest, &hs, pkey[i], pnode[i]);
                migrated = 1;
                break;
            }
        }
        int bi = pond_argmin(pkey, psz);                      /* the leader duck */
        int u = pnode[bi];
        double du = pkey[bi];
        ppos[u] = -1;                                         /* the black stone */
        {
            int last = --psz;
            if (bi != last) {
                pkey[bi] = pkey[last];
                int lu = pnode[last];
                pnode[bi] = lu; ppos[lu] = bi;
            }
        }
        {
            int e = EO[u], ee = EO[u + 1];
            for (; e < ee; e++) {
                int v = ED[e];
                double nd = du + EW[e];
                if (nd < D[v]) {
                    D[v] = nd;
                    if (hasout[v]) {                          /* slack roads never spool a duck */
                        int p = ppos[v];
                        if (p >= 0) pkey[p] = nd;             /* O(1) decrease-key */
                        else { pkey[psz] = nd; pnode[psz] = v; ppos[v] = psz; psz++; }
                    }
                }
            }
        }
    }

    /* ================= NEST PHASE (large sparse fallback) ================= */
    if (migrated) {
        while (hs > 0) {
            Duck top = nest_pop(nest, &hs);
            int u = top.u;
            if (done[u]) continue;
            if (top.d > D[u]) continue;                       /* stale duck */
            done[u] = 1;
            double du = D[u];
            int e = EO[u], ee = EO[u + 1];
            for (; e < ee; e++) {
                int v = ED[e];
                double nd = du + EW[e];
                if (nd < D[v]) {
                    D[v] = nd;
                    if (hasout[v]) nest_push(nest, &hs, nd, v);
                }
            }
        }
    }

    free(nest); free(done);
    free(cnt); free(off); free(edst); free(ew);
    free(hasout); free(pkey); free(pnode); free(ppos);
}
