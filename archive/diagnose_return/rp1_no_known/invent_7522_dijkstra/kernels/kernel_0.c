#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------------- the hovering hand: argmin over the contiguous thread array ----------------
   Pass 1 reduces to the true shortest thread length (pure vminpd, 4 accumulators).
   Pass 2 "shuttles back to its landing": finds the first corner carrying exactly that
   length.  Two passes, both branch-light, both contiguous; no gather, no index blends. */
static int leader_scan(const double *restrict key, int k, double *best_out)
{
    double best = INFINITY;
    int i = 0;
#if defined(__AVX2__)
    if (k >= 16) {
        __m256d v0 = _mm256_set1_pd(INFINITY), v1 = v0, v2 = v0, v3 = v0;
        for (; i + 16 <= k; i += 16) {
            v0 = _mm256_min_pd(v0, _mm256_loadu_pd(key + i));
            v1 = _mm256_min_pd(v1, _mm256_loadu_pd(key + i + 4));
            v2 = _mm256_min_pd(v2, _mm256_loadu_pd(key + i + 8));
            v3 = _mm256_min_pd(v3, _mm256_loadu_pd(key + i + 12));
        }
        v0 = _mm256_min_pd(_mm256_min_pd(v0, v1), _mm256_min_pd(v2, v3));
        __m128d lo = _mm256_castpd256_pd128(v0);
        __m128d hi = _mm256_extractf128_pd(v0, 1);
        __m128d mn = _mm_min_pd(lo, hi);
        mn = _mm_min_sd(mn, _mm_unpackhi_pd(mn, mn));
        best = _mm_cvtsd_f64(mn);
    }
#endif
    for (; i < k; i++) if (key[i] < best) best = key[i];
    *best_out = best;
    if (!(best < INFINITY)) return -1;          /* no duck left anywhere */

    i = 0;
#if defined(__AVX2__)
    {
        __m256d vm = _mm256_set1_pd(best);
        for (; i + 4 <= k; i += 4) {
            int msk = _mm256_movemask_pd(
                        _mm256_cmp_pd(_mm256_loadu_pd(key + i), vm, _CMP_EQ_OQ));
            if (msk) return i + __builtin_ctz((unsigned)msk);
        }
    }
#endif
    for (; i < k; i++) if (key[i] == best) return i;
    return -1;
}

/* ---------------- the sorted flock: 4-ary lazy heap (sprawl fallback) ---------------- */
typedef struct { double d; int u; } HItem;

static inline void h4_push(HItem *restrict h, int *hs, double d, int u)
{
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}

static inline HItem h4_pop(HItem *restrict h, int *hs)
{
    HItem top = h[0];
    int nn = --(*hs);
    if (nn == 0) return top;
    HItem last = h[nn];
    int i = 0;
    for (;;) {
        int c = (i << 2) + 1;
        if (c >= nn) break;
        int e = c + 4; if (e > nn) e = nn;
        int b = c; double bd = h[c].d;
        for (int j = c + 1; j < e; j++) if (h[j].d < bd) { bd = h[j].d; b = j; }
        if (bd >= last.d) break;
        h[i] = h[b];
        i = b;
    }
    h[i] = last;
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;                      /* the white stone, written first */
    if (m <= 0) return;

    int *deg = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;

    /* houses that have a far side */
    int k = 0;
    for (int i = 0; i < n; i++) if (deg[i] != 0) k++;

    /* ---- count houses against roads: weave or sprawl? ---- */
    double lg = 1.0;
    for (int t = n; t > 1; t >>= 1) lg += 1.0;
    double weave  = (double)k * (double)k;                        /* hand-walk work  */
    double sprawl = 20.0 * ((double)m + (double)n) * lg;          /* sorted-flock work */

    if (weave <= sprawl) {
        /* ================= WEAVE: no priority structure at all ================= */
        int *order = (int *)malloc((size_t)n * sizeof(int));   /* new -> old */
        int *pos   = (int *)malloc((size_t)n * sizeof(int));   /* old -> new */
        int a = 0, b = k;
        for (int i = 0; i < n; i++) {
            if (deg[i] != 0) { order[a] = i; pos[i] = a; a++; }
            else             { order[b] = i; pos[i] = b; b++; } /* slack roads: tail */
        }
        int *off = (int *)malloc((size_t)(k + 1) * sizeof(int));
        off[0] = 0;
        for (int i = 0; i < k; i++) off[i + 1] = off[i] + deg[order[i]];
        int *cur = (int *)malloc((size_t)(k + 1) * sizeof(int));
        memcpy(cur, off, (size_t)(k + 1) * sizeof(int));
        int *edst    = (int *)malloc((size_t)m * sizeof(int));
        double *ew   = (double *)malloc((size_t)m * sizeof(double));
        for (int i = 0; i < m; i++) {
            int u = pos[src[i]];                 /* deg>0, so u < k */
            int p = cur[u]++;
            edst[p] = pos[dst[i]];
            ew[p]   = weight[i];
        }

        double *d   = (double *)malloc((size_t)n * sizeof(double));
        double *key = (double *)malloc((size_t)(n + 16) * sizeof(double));
        for (int i = 0; i < n; i++) { d[i] = INFINITY; key[i] = INFINITY; }
        int s = pos[source];
        d[s] = 0.0;
        if (s < k) key[s] = 0.0;

        const int    *restrict E = edst;
        const double *restrict W = ew;
        const int    *restrict O = off;
        double *restrict D = d;
        double *restrict K = key;

        for (;;) {
            double best;
            int u = leader_scan(K, k, &best);    /* stillness, then the leader duck */
            if (u < 0) break;
            K[u] = INFINITY;                     /* the black stone */
            int e0 = O[u], e1 = O[u + 1];
            for (int e = e0; e < e1; e++) {      /* loose fresh ducks -- no queue */
                int v = E[e];
                double nd = best + W[e];
                if (nd < D[v]) { D[v] = nd; K[v] = nd; }
            }
        }
        for (int i = 0; i < n; i++) dist_out[order[i]] = d[i];

        free(order); free(pos); free(off); free(cur);
        free(edst); free(ew); free(d); free(key);
    } else {
        /* ================= SPRAWL: keep the ducks in a sorted flock ================= */
        int *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
        off[0] = 0;
        for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
        int *cur = (int *)malloc((size_t)(n + 1) * sizeof(int));
        memcpy(cur, off, (size_t)(n + 1) * sizeof(int));
        int *edst  = (int *)malloc((size_t)m * sizeof(int));
        double *ew = (double *)malloc((size_t)m * sizeof(double));
        for (int i = 0; i < m; i++) {
            int u = src[i]; int p = cur[u]++;
            edst[p] = dst[i]; ew[p] = weight[i];
        }

        /* only bother testing for slack roads if slack roads are actually common */
        int prune = ((double)(n - k) * 4.0 > (double)n);

        HItem *heap = (HItem *)malloc((size_t)(m + 2) * sizeof(HItem));
        int hs = 0;
        const int    *restrict E = edst;
        const double *restrict W = ew;
        const int    *restrict O = off;
        const int    *restrict G = deg;
        double *restrict D = dist_out;

        if (deg[source] != 0) h4_push(heap, &hs, 0.0, source);
        while (hs > 0) {
            HItem t = h4_pop(heap, &hs);
            int u = t.u;
            double du = t.d;
            if (du > D[u]) continue;             /* a stale duck: no 'done' array needed */
            int e0 = O[u], e1 = O[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = E[e];
                double nd = du + W[e];
                if (nd < D[v]) {
                    D[v] = nd;
                    if (!prune || G[v] != 0) h4_push(heap, &hs, nd, v);
                }
            }
        }
        free(off); free(cur); free(edst); free(ew); free(heap);
    }
    free(deg);
}
