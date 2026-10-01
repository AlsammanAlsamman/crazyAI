/* Dijkstra by ducks and stones:
 *   - no priority structure is consulted before any relaxation (SEED 2)
 *   - the leader is found by a hovering hand over block summaries (SEED 3)
 *   - houses with no far side are thrown away unspooled, never measured again
 *   - the grid is read first; a wide, thin grid gets the heap-ladder instead
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ===================== the heap-ladder (fallback path) ===================== */

typedef struct { double d; int u; } HeapItem;

static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) { int p = (i - 1) / 2; if (h[p].d <= d) break; h[i] = h[p]; i = p; }
    h[i].d = d; h[i].u = u;
}

static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    int sz = --(*hs);
    if (sz > 0) {
        HeapItem last = h[sz];
        int i = 0;
        for (;;) {
            int l = 2 * i + 1, r = l + 1, s = -1;
            double bd = last.d;
            if (l < sz && h[l].d < bd) { s = l; bd = h[l].d; }
            if (r < sz && h[r].d < bd) { s = r; }
            if (s < 0) break;
            h[i] = h[s]; i = s;
        }
        h[i] = last;
    }
    return top;
}

static void dij_ladder(int n, int m, const int *src, const int *dst,
                       const double *weight, int source, double *dist_out) {
    int *deg = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = (int *)malloc((size_t)(m ? m : 1) * sizeof(int));
    double *ew = (double *)malloc((size_t)(m ? m : 1) * sizeof(double));
    int *fill = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i]; int p = off[u] + fill[u]++;
        edst[p] = dst[i]; ew[p] = weight[i];
    }
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    char *done = (char *)calloc((size_t)n, 1);
    HeapItem *heap = (HeapItem *)malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0;
    hpush(heap, &hs, 0.0, source);
    while (hs > 0) {
        HeapItem top = hpop(heap, &hs);
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        double du = dist_out[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }
    free(deg); free(off); free(edst); free(ew); free(fill); free(done); free(heap);
}

/* ===================== the hovering hand (SIMD min / find) ================= */

static double hand_min(const double *__restrict a, int n) {
#if defined(__AVX2__)
    if (n >= 8) {
        __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0;
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            m0 = _mm256_min_pd(m0, _mm256_loadu_pd(a + i));
            m1 = _mm256_min_pd(m1, _mm256_loadu_pd(a + i + 4));
        }
        m0 = _mm256_min_pd(m0, m1);
        for (; i + 4 <= n; i += 4) m0 = _mm256_min_pd(m0, _mm256_loadu_pd(a + i));
        __m128d lo = _mm256_castpd256_pd128(m0);
        __m128d hi = _mm256_extractf128_pd(m0, 1);
        lo = _mm_min_pd(lo, hi);
        lo = _mm_min_sd(lo, _mm_unpackhi_pd(lo, lo));
        double best = _mm_cvtsd_f64(lo);
        for (; i < n; i++) if (a[i] < best) best = a[i];
        return best;
    }
#endif
    { double best = INFINITY;
      for (int i = 0; i < n; i++) if (a[i] < best) best = a[i];
      return best; }
}

static int hand_find(const double *__restrict a, int n, double val) {
#if defined(__AVX2__)
    {
        __m256d vv = _mm256_set1_pd(val);
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d x = _mm256_loadu_pd(a + i);
            int mk = _mm256_movemask_pd(_mm256_cmp_pd(x, vv, _CMP_EQ_OQ));
            if (mk) return i + __builtin_ctz((unsigned)mk);
        }
        for (; i < n; i++) if (a[i] == val) return i;
        return -1;
    }
#else
    for (int i = 0; i < n; i++) if (a[i] == val) return i;
    return -1;
#endif
}

/* ========================== the ducks (main path) ========================== */

static void dij_ducks(int n, int m, const int *src, const int *dst,
                      const double *weight, int source, double *dist_out) {
    /* --- read the grid: which houses have a far side? --- */
    int *deg  = (int *)calloc((size_t)n, sizeof(int));
    int *perm = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;

    int ns = 0;
    for (int v = 0; v < n; v++) if (deg[v] > 0) ns++;
    /* houses with a far side get the low names (they will be walked over);
       dead ends get the high names and are never measured again           */
    { int a = 0, b = ns;
      for (int v = 0; v < n; v++) perm[v] = (deg[v] > 0) ? a++ : b++; }

    int *off = (int *)malloc((size_t)(ns + 1) * sizeof(int));
    off[0] = 0;
    { int c = 0;
      for (int v = 0; v < n; v++) if (deg[v] > 0) { c += deg[v]; off[perm[v] + 1] = c; } }

    int    *edst = (int *)   malloc((size_t)(m ? m : 1) * sizeof(int));
    double *ew   = (double *)malloc((size_t)(m ? m : 1) * sizeof(double));
    int    *fill = (int *)   calloc((size_t)(ns ? ns : 1), sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = perm[src[i]];
        int p = off[u] + fill[u]++;
        edst[p] = perm[dst[i]];
        ew[p]   = weight[i];
    }

    double *__restrict dist = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    int s = perm[source];
    dist[s] = 0.0;

    /* --- block size: hand hovers over ~sqrt(ns/5) summaries --- */
    int shift = 5;
    while (shift < 13 && ((size_t)1 << (2 * shift)) * 5u < (size_t)ns) shift++;
    int B  = 1 << shift;
    int nb = (ns + B - 1) >> shift;
    if (nb < 1) nb = 1;

    double *__restrict key  = (double *)malloc(((size_t)ns + 8) * sizeof(double));
    double *__restrict bmin = (double *)malloc(((size_t)nb + 8) * sizeof(double));
    for (int i = 0; i < ns; i++) key[i]  = INFINITY;
    for (int j = 0; j < nb; j++) bmin[j] = INFINITY;
    if (s < ns) { key[s] = 0.0; bmin[s >> shift] = 0.0; }

    const int    *__restrict cedst = edst;
    const double *__restrict cew   = ew;
    const int    *__restrict coff  = off;

    while (ns > 0) {
        /* stillness: hover over the summaries */
        double best = hand_min(bmin, nb);
        if (!(best < INFINITY)) break;               /* nothing reachable left */
        int bj = hand_find(bmin, nb, best);
        if (bj < 0) break;                           /* defensive; cannot happen */
        int lo = bj << shift;
        int hi = lo + B; if (hi > ns) hi = ns;

        /* it may hide among its long cousins: shuttle back to its landing */
        int p = hand_find(key + lo, hi - lo, best);
        if (p < 0) { bmin[bj] = hand_min(key + lo, hi - lo); continue; }
        int u = lo + p;
        double du = best;

        /* black stone */
        key[u] = INFINITY;
        bmin[bj] = hand_min(key + lo, hi - lo);

        /* loose fresh ducks: no ladder is consulted, ever */
        int e = coff[u], ee = coff[u + 1];
        for (; e < ee; e++) {
            if (e + 8 < ee) __builtin_prefetch(&dist[cedst[e + 8]], 1, 1);
            int v = cedst[e];
            double nd = du + cew[e];
            if (nd < dist[v]) {
                dist[v] = nd;
                if (v < ns) {                        /* dead ends stay unspooled */
                    key[v] = nd;
                    int jb = v >> shift;
                    if (nd < bmin[jb]) bmin[jb] = nd;
                }
            }
        }
    }

    for (int v = 0; v < n; v++) dist_out[v] = dist[perm[v]];

    free(deg); free(perm); free(off); free(edst); free(ew); free(fill);
    free(dist); free(key); free(bmin);
}

/* ============================== the contract ============================== */

void kernel(int n, int m, const int *src, const int *dst,
            const double *weight, int source, double *dist_out) {
    if (n <= 0) return;
    if (m <= 0) {
        for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
        if (source >= 0 && source < n) dist_out[source] = 0.0;
        return;
    }
    /* read the grid before loosing anything: tight-and-thick -> walk,
       wide-and-thin -> climb the ladder instead                        */
    double dn = (double)n, dm = (double)m;
    double lg = log2(dn + 2.0);
    double est_walk   = 0.60 * dn * sqrt(dn) + 8.0 * dm;
    double est_ladder = 5.0 * (dn + dm) * lg;
    if (est_walk <= est_ladder) dij_ducks (n, m, src, dst, weight, source, dist_out);
    else                        dij_ladder(n, m, src, dst, weight, source, dist_out);
}
