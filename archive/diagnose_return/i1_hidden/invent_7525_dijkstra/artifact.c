/* Dijkstra as the native tells it:
 *   - a rigid lattice with one fixed slot per place (no key ever migrates)
 *   - inside each lattice, a smaller lattice of the same kind (8-ary block minima)
 *   - settle the quietest unsettled slot; never touch a settled board twice
 *   - a failed relaxation is thrown away and stays thrown away (no duplicate entries)
 *   - unreached places are never visited; they stay blank (INFINITY)
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#if defined(__AVX__)
#include <immintrin.h>
#endif

/* one road: how hoarse the bird gets, and which board it dies on */
typedef struct { double w; int d; int pad; } Road;   /* 16 B, never straddles a line */

/* the quietest of eight notes in one stretch of lattice, and where it sits */
static inline int qmin8(const double *p, double *out)
{
#if defined(__AVX__)
    __m256d a  = _mm256_loadu_pd(p);
    __m256d b  = _mm256_loadu_pd(p + 4);
    __m256d mv = _mm256_min_pd(a, b);
    __m128d lo = _mm256_castpd256_pd128(mv);
    __m128d hi = _mm256_extractf128_pd(mv, 1);
    __m128d m2 = _mm_min_pd(lo, hi);
    __m128d m3 = _mm_min_sd(m2, _mm_unpackhi_pd(m2, m2));
    double  mn = _mm_cvtsd_f64(m3);
    *out = mn;
    __m256d bc = _mm256_set1_pd(mn);
    int ma = _mm256_movemask_pd(_mm256_cmp_pd(a, bc, _CMP_EQ_OQ));
    if (ma) return (int)__builtin_ctz((unsigned)ma);
    int mb = _mm256_movemask_pd(_mm256_cmp_pd(b, bc, _CMP_EQ_OQ));
    return 4 + (int)__builtin_ctz((unsigned)mb);
#else
    double mn = p[0]; int bi = 0;
    for (int t = 1; t < 8; ++t) { double x = p[t]; if (x < mn) { mn = x; bi = t; } }
    *out = mn; return bi;
#endif
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    /* every place blank as the desert until a bird reaches it */
    if (n >= (1 << 18)) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;
    } else {
        for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;
    }
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- the roads out of every place, grouped once ---- */
    int  *rp  = (int  *)calloc((size_t)n + 2, sizeof(int));
    int  *pos = (int  *)malloc((size_t)n * sizeof(int));
    Road *ed  = (Road *)malloc(((size_t)m + 8) * sizeof(Road));
    if (!rp || !pos || !ed) { free(rp); free(pos); free(ed); return; }
    for (int i = 0; i < m; ++i) rp[src[i] + 1]++;
    for (int i = 0; i < n; ++i) rp[i + 1] += rp[i];
    memcpy(pos, rp, (size_t)n * sizeof(int));
    for (int i = 0; i < m; ++i) {
        int u = src[i];
        int p = pos[u]++;
        ed[p].d = dst[i];
        ed[p].w = weight[i];
    }
    memset(ed + m, 0, 8 * sizeof(Road));   /* harmless targets for prefetch */
    free(pos);

    /* GUARD (see VERDICT): below this size one straight look at the whole
       lattice is cheaper than walking the nest of smaller lattices, and it
       needs no allocation at all.  This is the native's own one-level reading. */
    if (n <= 192) {
        double lat[192];
        for (int i = 0; i < n; ++i) lat[i] = INFINITY;
        lat[source] = 0.0;
        for (;;) {
            double best = INFINITY; int v = -1;
            for (int i = 0; i < n; ++i) if (lat[i] < best) { best = lat[i]; v = i; }
            if (v < 0) break;
            lat[v] = INFINITY;                     /* settled: out of the lattice */
            for (int e = rp[v], ee = rp[v + 1]; e < ee; ++e) {
                int w = ed[e].d; double nd = best + ed[e].w;
                if (nd < dist_out[w]) { dist_out[w] = nd; lat[w] = nd; }
            }
        }
        free(rp); free(ed); return;
    }

    /* ---- the lattice, and inside it the smaller lattices ----
       level 0 IS dist_out (the carvings).  level k holds, for each block of
       8 slots of level k-1, the quietest *unsettled* note beneath it and
       which place it belongs to.  Slots never move: id is position, always. */
    int n0 = (n + 7) & ~7;
    int sz[32]; int L;
    {
        int c = n0 >> 3; if (c < 1) c = 1;
        sz[1] = (c + 7) & ~7; if (sz[1] < 8) sz[1] = 8;
        L = 1;
        while (sz[L] > 8) {
            int cc = sz[L] >> 3;
            int s2 = (cc + 7) & ~7; if (s2 < 8) s2 = 8;
            sz[L + 1] = s2; ++L;
        }
    }
    size_t tot = 0;
    for (int k = 1; k <= L; ++k) tot += (size_t)sz[k];

    void          *kraw = malloc(tot * sizeof(double) + 64);
    int           *ibuf = (int *)malloc(tot * sizeof(int));
    unsigned char *sb   = (unsigned char *)calloc((size_t)(n0 >> 3) + 16, 1);
    if (!kraw || !ibuf || !sb) { free(kraw); free(ibuf); free(sb); free(rp); free(ed); return; }
    double *kbuf = (double *)((((uintptr_t)kraw) + 63) & ~(uintptr_t)63);
    double *lv[32]; int *id[32];
    { size_t off = 0;
      for (int k = 1; k <= L; ++k) { lv[k] = kbuf + off; id[k] = ibuf + off; off += (size_t)sz[k]; } }
    if (tot >= (1u << 18)) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < tot; ++i) kbuf[i] = INFINITY;
    } else {
        for (size_t i = 0; i < tot; ++i) kbuf[i] = INFINITY;
    }
    memset(ibuf, 0, tot * sizeof(int));

    /* the traveler's own place enters the lattice at zero hoarseness */
    {
        int w = source;
        lv[1][w >> 3] = 0.0; id[1][w >> 3] = w;
        for (int k = 2; k <= L; ++k) {
            int jj = w >> (3 * k);
            if (0.0 < lv[k][jj]) { lv[k][jj] = 0.0; id[k][jj] = w; } else break;
        }
    }

    for (;;) {
        /* the quietest unsettled note in the whole land: one look at the top */
        double mn; int p = qmin8(lv[L], &mn);
        if (mn >= INFINITY) break;              /* no bird sings; the rest stays blank */
        int    v  = id[L][p];
        double dv = mn;                          /* == dist_out[v], now permanent */

        /* settle: this board is never touched again */
        int j = v >> 3;
        sb[j] |= (unsigned char)(1u << (v & 7));
        {   /* recut the smallest lattice over v's block, masking the settled */
            int base = j << 3;
            int hi   = (base + 8 <= n) ? 8 : (n - base);
            unsigned s = sb[j];
            double bm = INFINITY; int bi = base;
            for (int t = 0; t < hi; ++t)
                if (!((s >> t) & 1u)) {
                    double x = dist_out[base + t];
                    if (x < bm) { bm = x; bi = base + t; }
                }
            lv[1][j] = bm; id[1][j] = bi;
        }
        for (int k = 2; k <= L; ++k) {           /* the cows outward, one per level */
            int jj = v >> (3 * k), b2 = jj << 3;
            double m2; int q = qmin8(lv[k - 1] + b2, &m2);
            lv[k][jj] = m2; id[k][jj] = id[k - 1][b2 + q];
        }

        /* loose one nightingale down every road out of v; each dies at once */
        for (int e = rp[v], ee = rp[v + 1]; e < ee; ++e) {
            __builtin_prefetch(&dist_out[ed[e + 4].d], 1, 1);
            int    w  = ed[e].d;
            double nd = dv + ed[e].w;
            if (nd < dist_out[w]) {              /* thinner milk: recut the slot */
                dist_out[w] = nd;
                int jw = w >> 3;
                if (nd < lv[1][jw]) {
                    lv[1][jw] = nd; id[1][jw] = w;
                    for (int k = 2; k <= L; ++k) {
                        int jj = w >> (3 * k);
                        if (nd < lv[k][jj]) { lv[k][jj] = nd; id[k][jj] = w; } else break;
                    }
                }
            }
            /* louder note: thrown away, and it stays thrown away */
        }
    }

    free(sb); free(ibuf); free(kraw); free(ed); free(rp);
}
