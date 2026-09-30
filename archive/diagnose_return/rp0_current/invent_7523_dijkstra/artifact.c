/* THROW-AND-SEAL  ---  single-source shortest distances.
 *
 * place            -> node index
 * road             -> CSR edge (dst, weight)
 * notch            -> weight
 * lit ember        -> finite tentative dist_out[v]
 * cold ash         -> INFINITY
 * runner           -> one relaxation, launched from a merely-tentative ember
 * stick on a heap  -> candidate value cand[v]
 * garrison refuses -> the strict relax test (nd < dist && nd < cand)
 * land-wide throw  -> one synchronized round: dist read-only, then applied
 * row read across  -> per-node local min; SIMD, 4 heaps per breath
 * burned for fuel  -> candidate dropped, never stored
 * garrison seals   -> node leaves the frontier; its runners stop
 * hooded figure    -> the O(n^2) row-scan for the crisscrossed land
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ---------------- THE HOODED FIGURE: dense / tiny land ----------------
 * O(n^2) array-scan Dijkstra.  Each dawn he reads straight across one
 * row of embers in a single SIMD breath and names the dimmest unsealed
 * heap.  Sealed heaps carry key = INFINITY, so the scan is a pure
 * vector min with index tracking -- no branchy mask bookkeeping.      */
static void hooded_scan(int n,
                        const int    * restrict off,
                        const int    * restrict edst,
                        const double * restrict ew,
                        int source,
                        double * restrict dist,
                        double * restrict key)
{
    for (int i = 0; i < n; i++) { dist[i] = INFINITY; key[i] = INFINITY; }
    dist[source] = 0.0; key[source] = 0.0;

    for (int it = 0; it < n; it++) {
        int    best = -1;
        double bd   = INFINITY;
        int    i    = 0;
#ifdef __AVX2__
        if (n >= 8) {
            __m256d vb   = _mm256_set1_pd(INFINITY);
            __m256d vi   = _mm256_set1_pd(-1.0);
            __m256d step = _mm256_set1_pd(4.0);
            __m256d idx  = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
            for (; i + 4 <= n; i += 4) {
                __m256d d  = _mm256_loadu_pd(key + i);
                __m256d lt = _mm256_cmp_pd(d, vb, _CMP_LT_OQ);
                vb  = _mm256_blendv_pd(vb, d,   lt);
                vi  = _mm256_blendv_pd(vi, idx, lt);
                idx = _mm256_add_pd(idx, step);
            }
            double bl[4], il[4];
            _mm256_storeu_pd(bl, vb);
            _mm256_storeu_pd(il, vi);
            for (int j = 0; j < 4; j++)
                if (bl[j] < bd) { bd = bl[j]; best = (int)il[j]; }
        }
#endif
        for (; i < n; i++)
            if (key[i] < bd) { bd = key[i]; best = i; }

        if (best < 0 || !(bd < INFINITY)) break;   /* only cold ash left */
        key[best] = INFINITY;                      /* garrison seals it  */

        const int e1 = off[best + 1];
        for (int e = off[best]; e < e1; e++) {
            const int v = edst[e];
            const double nd = bd + ew[e];
            /* a sealed v has dist[v] <= bd <= nd, so it refuses entry */
            if (nd < dist[v]) { dist[v] = nd; key[v] = nd; }
        }
    }
}

/* -------- THE HOODED FIGURE WITH HIS HEAP: pathological-land bail ---- */
typedef struct { double d; int u; } HItem;

static void hpush(HItem * restrict h, int *hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= d) break;
        h[i] = h[p]; i = p;
    }
    h[i].d = d; h[i].u = u;
}

static void heap_dijkstra(int n, int m,
                          const int    * restrict off,
                          const int    * restrict edst,
                          const double * restrict ew,
                          int source,
                          double * restrict dist,
                          unsigned char * restrict done,
                          HItem * restrict heap)
{
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    memset(done, 0, (size_t)n);
    dist[source] = 0.0;
    int hs = 0;
    hpush(heap, &hs, 0.0, source);
    while (hs > 0) {
        HItem top = heap[0];
        int sz = --hs;
        HItem last = heap[sz];
        if (sz > 0) {
            int i = 0;
            for (;;) {
                int l = 2 * i + 1;
                if (l >= sz) break;
                int r = l + 1;
                int s = (r < sz && heap[r].d < heap[l].d) ? r : l;
                if (heap[s].d >= last.d) break;
                heap[i] = heap[s];
                i = s;
            }
            heap[i] = last;
        }
        const int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        const double du = dist[u];
        const int e1 = off[u + 1];
        for (int e = off[u]; e < e1; e++) {
            const int v = edst[e];
            const double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }
    (void)m;
}

/* ============================== KERNEL ============================== */
void kernel(int n, int m, const int *src, const int *dst,
            const double *weight, int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- file every road under its home place (CSR) ---- */
    int    *off  = (int *)calloc((size_t)n + 1, sizeof(int));
    int    *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !edst || !ew) { free(off); free(edst); free(ew); return; }

    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    {
        int *fill = (int *)malloc((size_t)n * sizeof(int));
        if (!fill) { free(off); free(edst); free(ew); return; }
        memcpy(fill, off, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            const int u = src[i];
            const int p = fill[u]++;
            edst[p] = dst[i];
            ew[p]   = weight[i];
        }
        free(fill);
    }

    /* ---- REGIME 1: crisscrossed or tiny land -> the hooded figure ----
     * row-scan cost ~ n^2/4 (SIMD) + m ; throw-and-seal ~ 3m.
     * scan wins once m > n^2/12.                                     */
    if (n < 32 || (double)m * 12.0 > (double)n * (double)n) {
        double *key = (double *)malloc((size_t)n * sizeof(double));
        if (key) {
            hooded_scan(n, off, edst, ew, source, dist_out, key);
            free(key);
            free(off); free(edst); free(ew);
            return;
        }
        /* no scratch: fall through to the runners */
    }

    /* ---- REGIME 2: open land -> THROW AND SEAL ---- */
    double        *cand     = (double *)malloc((size_t)n * sizeof(double));
    int           *frontier = (int *)malloc((size_t)n * sizeof(int));
    int           *next     = (int *)malloc((size_t)n * sizeof(int));
    unsigned char *innext   = (unsigned char *)calloc((size_t)n, 1);
    if (!cand || !frontier || !next || !innext) {
        free(cand); free(frontier); free(next); free(innext);
        free(off); free(edst); free(ew);
        return;
    }
    for (int i = 0; i < n; i++) cand[i] = INFINITY;

    int fcount = 1;
    frontier[0] = source;

    /* the native's own thrift: if the runners burn more sticks than the
     * hooded figure would have counted, dismiss them.                */
    const size_t budget = (size_t)8 * (size_t)m + (size_t)16 * (size_t)n + 4096;
    size_t spent = 0;
    int bailed = 0;

    while (fcount > 0) {
        if (spent > budget) { bailed = 1; break; }
        int ncount = 0;

        /* --- the runners: dist_out is READ-ONLY for the whole round --- */
        for (int fi = 0; fi < fcount; fi++) {
            const int u = frontier[fi];
            const double du = dist_out[u];
            const int e0 = off[u], e1 = off[u + 1];
            spent += (size_t)(e1 - e0) + 1u;
            int e = e0;
#ifdef __AVX2__
            if (e1 - e0 >= 16) {
                const __m256d vdu = _mm256_set1_pd(du);
                for (; e + 4 <= e1; e += 4) {
                    __m128i vv = _mm_loadu_si128((const __m128i *)(edst + e));
                    __m256d vw = _mm256_loadu_pd(ew + e);
                    __m256d nd = _mm256_add_pd(vdu, vw);
                    __m256d dv = _mm256_i32gather_pd(dist_out, vv, 8);
                    __m256d cv = _mm256_i32gather_pd(cand,     vv, 8);
                    __m256d lt = _mm256_and_pd(
                        _mm256_cmp_pd(nd, dv, _CMP_LT_OQ),
                        _mm256_cmp_pd(nd, cv, _CMP_LT_OQ));
                    int msk = _mm256_movemask_pd(lt);
                    if (msk) {
                        double nds[4];
                        _mm256_storeu_pd(nds, nd);
                        for (int j = 0; j < 4; j++) if (msk & (1 << j)) {
                            const int v = edst[e + j];
                            /* re-test: two lanes may share the same heap */
                            if (nds[j] < dist_out[v] && nds[j] < cand[v]) {
                                cand[v] = nds[j];
                                if (!innext[v]) { innext[v] = 1; next[ncount++] = v; }
                            }
                        }
                    }
                }
            }
#endif
            for (; e < e1; e++) {
                const int v = edst[e];
                const double nd = du + ew[e];
                /* a heap already ringed by its garrison refuses the stick,
                 * and the stick is burned for fuel on the spot           */
                if (nd < dist_out[v] && nd < cand[v]) {
                    cand[v] = nd;
                    if (!innext[v]) { innext[v] = 1; next[ncount++] = v; }
                }
            }
        }

        /* --- THE THROW: one row read straight across, all heaps in one
         * breath.  Each heap keeps only its shortest sliver as the new
         * ember; the rest were already burned.  No heap is compared to
         * any other heap.  Heaps that caught nothing are sealed: their
         * runners stop going out.                                    */
        for (int k = 0; k < ncount; k++) {
            const int v = next[k];
            dist_out[v] = cand[v];
            cand[v]     = INFINITY;
            innext[v]   = 0;
            frontier[k] = v;
        }
        spent += (size_t)ncount;
        fcount = ncount;
    }

    /* ---- REGIME 3: the land defeated the runners -> heap Dijkstra ---- */
    if (bailed) {
        HItem *heap = (HItem *)malloc(((size_t)m + 2) * sizeof(HItem));
        if (heap) {
            heap_dijkstra(n, m, off, edst, ew, source, dist_out, innext, heap);
            free(heap);
        } else {
            double *key = (double *)malloc((size_t)n * sizeof(double));
            if (key) {
                hooded_scan(n, off, edst, ew, source, dist_out, key);
                free(key);
            }
        }
    }

    free(cand); free(frontier); free(next); free(innext);
    free(off);  free(edst);     free(ew);
}
