#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ROW      32
#define ROWSHIFT 5

/* ---------------- the cairn: 4-ary heap, faintest stone on top ---------------- */
typedef struct { double d; int u; int pad; } Stone;

static inline void cairn_push(Stone *restrict h, int *restrict hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}

static inline Stone cairn_pop(Stone *restrict h, int *restrict hs) {
    Stone top = h[0];
    int sz = --(*hs);
    if (sz > 0) {
        Stone last = h[sz];
        int i = 0;
        for (;;) {
            int c = (i << 2) + 1;
            if (c >= sz) break;
            int e = c + 4; if (e > sz) e = sz;
            int b = c; double bd = h[c].d;
            for (int k = c + 1; k < e; k++) { double dk = h[k].d; if (dk < bd) { bd = dk; b = k; } }
            if (!(bd < last.d)) break;
            h[i] = h[b];
            i = b;
        }
        h[i] = last;
    }
    return top;
}

/* ------------- the faintest note in one row of the lattice (blanks ignored) ---- */
static inline double row_min(const double *restrict key, int b) {
    const double *p = key + (size_t)b * ROW;
#if defined(__AVX2__)
    __m256d a0 = _mm256_set1_pd(INFINITY), a1 = a0, a2 = a0, a3 = a0;
    for (int k = 0; k < ROW; k += 16) {
        a0 = _mm256_min_pd(_mm256_loadu_pd(p + k),      a0);
        a1 = _mm256_min_pd(_mm256_loadu_pd(p + k + 4),  a1);
        a2 = _mm256_min_pd(_mm256_loadu_pd(p + k + 8),  a2);
        a3 = _mm256_min_pd(_mm256_loadu_pd(p + k + 12), a3);
    }
    a0 = _mm256_min_pd(a0, a1); a2 = _mm256_min_pd(a2, a3); a0 = _mm256_min_pd(a0, a2);
    __m128d lo = _mm256_castpd256_pd128(a0);
    __m128d hi = _mm256_extractf128_pd(a0, 1);
    __m128d mm = _mm_min_pd(lo, hi);
    mm = _mm_min_sd(mm, _mm_unpackhi_pd(mm, mm));
    return _mm_cvtsd_f64(mm);
#else
    double best = INFINITY;
    for (int k = 0; k < ROW; k++) { double v = p[k]; if (v < best) best = v; }
    return best;
#endif
}

/* ------------------- the glance: quietest nick along the rim ------------------- */
static inline int rim_argmin(const double *restrict mark, int nbpad, double *restrict outbest) {
#if defined(__AVX2__)
    __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0, m2 = m0, m3 = m0;
    __m256d j0 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
    __m256d j1 = _mm256_add_pd(j0, _mm256_set1_pd(4.0));
    __m256d j2 = _mm256_add_pd(j0, _mm256_set1_pd(8.0));
    __m256d j3 = _mm256_add_pd(j0, _mm256_set1_pd(12.0));
    __m256d k0 = _mm256_set1_pd(-1.0), k1 = k0, k2 = k0, k3 = k0;
    const __m256d st = _mm256_set1_pd(16.0);
    for (int i = 0; i < nbpad; i += 16) {
        __m256d v0 = _mm256_loadu_pd(mark + i);
        __m256d v1 = _mm256_loadu_pd(mark + i + 4);
        __m256d v2 = _mm256_loadu_pd(mark + i + 8);
        __m256d v3 = _mm256_loadu_pd(mark + i + 12);
        __m256d c0 = _mm256_cmp_pd(v0, m0, _CMP_LT_OQ);
        __m256d c1 = _mm256_cmp_pd(v1, m1, _CMP_LT_OQ);
        __m256d c2 = _mm256_cmp_pd(v2, m2, _CMP_LT_OQ);
        __m256d c3 = _mm256_cmp_pd(v3, m3, _CMP_LT_OQ);
        m0 = _mm256_blendv_pd(m0, v0, c0); k0 = _mm256_blendv_pd(k0, j0, c0);
        m1 = _mm256_blendv_pd(m1, v1, c1); k1 = _mm256_blendv_pd(k1, j1, c1);
        m2 = _mm256_blendv_pd(m2, v2, c2); k2 = _mm256_blendv_pd(k2, j2, c2);
        m3 = _mm256_blendv_pd(m3, v3, c3); k3 = _mm256_blendv_pd(k3, j3, c3);
        j0 = _mm256_add_pd(j0, st); j1 = _mm256_add_pd(j1, st);
        j2 = _mm256_add_pd(j2, st); j3 = _mm256_add_pd(j3, st);
    }
    double mv[16], iv[16];
    _mm256_storeu_pd(mv,      m0); _mm256_storeu_pd(mv + 4,  m1);
    _mm256_storeu_pd(mv + 8,  m2); _mm256_storeu_pd(mv + 12, m3);
    _mm256_storeu_pd(iv,      k0); _mm256_storeu_pd(iv + 4,  k1);
    _mm256_storeu_pd(iv + 8,  k2); _mm256_storeu_pd(iv + 12, k3);
    double best = INFINITY; int bi = -1;
    for (int t = 0; t < 16; t++) if (mv[t] < best) { best = mv[t]; bi = (int)iv[t]; }
    *outbest = best;
    return bi;
#else
    double best = INFINITY; int bi = -1;
    for (int i = 0; i < nbpad; i++) { double v = mark[i]; if (v < best) { best = v; bi = i; } }
    *outbest = best; return bi;
#endif
}

/* ------------- which slot of the winning row sings that note ------------------- */
static inline int row_slot(const double *restrict key, int b, double best) {
    const double *p = key + (size_t)b * ROW;
#if defined(__AVX2__)
    __m256d vb = _mm256_set1_pd(best);
    for (int k = 0; k < ROW; k += 4) {
        int msk = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(p + k), vb, _CMP_EQ_OQ));
        if (msk) return k + (int)__builtin_ctz((unsigned)msk);
    }
    return -1;
#else
    for (int k = 0; k < ROW; k++) if (p[k] == best) return k;
    return -1;
#endif
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;      /* the desert, until carved */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- the roads, grouped by the place they leave from ---- */
    int    *off  = (int *)   malloc((size_t)(n + 2) * sizeof(int));
    int    *edst = (int *)   malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !edst || !ew) { free(off); free(edst); free(ew); return; }
    memset(off, 0, (size_t)(n + 2) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    for (int i = 0; i < m; i++) { int u = src[i]; int p = off[u]++; edst[p] = dst[i]; ew[p] = weight[i]; }
    for (int u = n; u > 0; u--) off[u] = off[u - 1];
    off[0] = 0;

    /* ---- count places against roads: which regime is this land in? ---- */
    double dn = (double)n, dm = (double)m;
    double lg = log2(dn > 2.0 ? dn : 2.0);
    double lattice_cost = 0.012 * dn * dn + 26.0 * dn;
    double cairn_cost   = 13.0 * (dm + dn) * lg;
    int use_lattice = (n <= 4096) || (lattice_cost <= cairn_cost);

    if (use_lattice) {
        /* =============== the native's own way: glance at the lattice =============== */
        int nb    = (n + ROW - 1) / ROW;
        int npad  = nb * ROW;
        int nbpad = ((nb + 15) / 16) * 16;
        void *rawk = malloc((size_t)npad  * sizeof(double) + 32);
        void *rawm = malloc((size_t)nbpad * sizeof(double) + 32);
        if (rawk && rawm) {
            double *restrict key  = (double *)(((uintptr_t)rawk + 31u) & ~(uintptr_t)31u);
            double *restrict mark = (double *)(((uintptr_t)rawm + 31u) & ~(uintptr_t)31u);
            const double BLANK = NAN;                 /* blank as the pink-brown desert */
            for (int i = 0; i < n; i++)     key[i]  = INFINITY;
            for (int i = n; i < npad; i++)  key[i]  = BLANK;
            for (int i = 0; i < nbpad; i++) mark[i] = INFINITY;
            key[source] = 0.0;
            mark[source >> ROWSHIFT] = 0.0;

            for (;;) {
                double best;
                int b = rim_argmin(mark, nbpad, &best);
                if (b < 0 || !(best < INFINITY)) break;      /* last bird silent; rest is desert */
                int s = row_slot(key, b, best);
                if (s < 0) { mark[b] = row_min(key, b); continue; }
                int u = (b << ROWSHIFT) + s;

                dist_out[u] = best;                          /* the permanent carving */
                key[u]      = BLANK;                         /* scrape the slot clean */
                mark[b]     = row_min(key, b);               /* re-nick that row */

                int e = off[u], e2 = off[u + 1];
                for (; e < e2; e++) {                        /* loose every bird at once */
                    int v = edst[e];
                    double c = best + ew[e];
                    if (c < key[v]) {                        /* NaN slot => false => untouched */
                        key[v] = c;                          /* cut the thinner milk in */
                        int bv = v >> ROWSHIFT;
                        if (c < mark[bv]) mark[bv] = c;
                    }
                }
            }
        }
        free(rawk); free(rawm);
    } else {
        /* ====== vast land, thin roads: same birds, same cow, choosing from a cairn ===== */
        char  *done  = (char *) calloc((size_t)n, 1);
        Stone *cairn = (Stone *)malloc((size_t)(m + 2) * sizeof(Stone));
        if (done && cairn) {
            int hs = 0;
            cairn_push(cairn, &hs, 0.0, source);
            while (hs > 0) {
                Stone top = cairn_pop(cairn, &hs);
                int u = top.u;
                if (done[u]) continue;                       /* thrown away stays thrown away */
                done[u] = 1;
                double du = top.d;
                dist_out[u] = du;
                int e = off[u], e2 = off[u + 1];
                for (; e < e2; e++) {
                    int v = edst[e];
                    if (done[v]) continue;                   /* dead road, no bird again */
                    double c = du + ew[e];
                    if (c < dist_out[v]) { dist_out[v] = c; cairn_push(cairn, &hs, c, v); }
                }
            }
        }
        free(done); free(cairn);
    }

    free(off); free(edst); free(ew);
}
