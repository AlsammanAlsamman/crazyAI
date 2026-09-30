/* Nightingales-and-lattice shortest paths.
 *
 *   dist_out          = the rigid lattice: one fixed slot per place in the land
 *   NRoad             = a road: its stretch of ground, and the board it ends at
 *   a bird's death    = one compare-and-store into one fixed slot (no queue touched)
 *   the cow's head    = the nested relax loop: re-measure by way of the last settled board
 *   faintest()        = walk the carved slots for the quietest note (never walk the desert)
 *   the ranked cairn  = 4-ary heap, used only when the land is vast and thinly roaded
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* a road: stretch of ground + the cross-marked board it ends at (16B, one cache-line quarter) */
typedef struct { double w; int v; int pad; } NRoad;
/* a dying note held in the cairn (16B: four notes to a 64B course) */
typedef struct { double d; int u; int pad; } NNote;

/* ---- the ranked cairn: a 4-ary heap of dying notes -------------------- */
static inline void cairn_push(NNote *restrict h, int *restrict hs, double d, int u)
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

static inline NNote cairn_pop(NNote *restrict h, int *restrict hs)
{
    NNote top = h[0];
    int last = --(*hs);
    if (last > 0) {
        double d = h[last].d;
        int u = h[last].u;
        int i = 0;
        for (;;) {
            int c = (i << 2) + 1;
            if (c >= last) break;
            int e = c + 4; if (e > last) e = last;
            int best = c; double bd = h[c].d;
            for (int k = c + 1; k < e; k++) if (h[k].d < bd) { bd = h[k].d; best = k; }
            if (bd >= d) break;
            h[i] = h[best];
            i = best;
        }
        h[i].d = d; h[i].u = u;
    }
    return top;
}

/* ---- walk the carved slots for the faintest note ---------------------- *
 * The branch inside is taken only when the running minimum improves, which
 * on a frontier of size k happens ~O(log k) times: the hot path is
 * load / compare / movemask / test over 8 doubles at a time.             */
static int faintest(const double *restrict a, int na)
{
    int bi = 0;
    double bv = a[0];
#if defined(__AVX2__)
    int i = 0;
    __m256d bvv = _mm256_set1_pd(bv);
    for (; i + 8 <= na; i += 8) {
        __m256d x0 = _mm256_loadu_pd(a + i);
        __m256d x1 = _mm256_loadu_pd(a + i + 4);
        __m256d m0 = _mm256_cmp_pd(x0, bvv, _CMP_LT_OQ);
        __m256d m1 = _mm256_cmp_pd(x1, bvv, _CMP_LT_OQ);
        if (_mm256_movemask_pd(_mm256_or_pd(m0, m1))) {
            for (int k = 0; k < 8; k++)
                if (a[i + k] < bv) { bv = a[i + k]; bi = i + k; }
            bvv = _mm256_set1_pd(bv);
        }
    }
    for (; i < na; i++) if (a[i] < bv) { bv = a[i]; bi = i; }
#else
    for (int i = 1; i < na; i++) if (a[i] < bv) { bv = a[i]; bi = i; }
#endif
    return bi;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    double *restrict dist = dist_out;
    const int    *restrict es = src;
    const int    *restrict ez = dst;
    const double *restrict ew = weight;

    for (int i = 0; i < n; i++) dist[i] = INFINITY;      /* blank as the desert */
    if (source < 0 || source >= n) return;
    dist[source] = 0.0;
    if (m <= 0) return;

    /* ---- lay out the roads: one contiguous run of roads per board ----
     * Counts are written shifted by two, so after the prefix sum off[u+1]
     * is u's cursor and, once every road is placed, off[0..n] is exactly
     * the finished offset table. One 16B random write per road, not two. */
    int   *off = (int   *)malloc((size_t)(n + 2) * sizeof(int));
    NRoad *rd  = (NRoad *)malloc((size_t)m * sizeof(NRoad));
    if (!off || !rd) { free(off); free(rd); return; }
    memset(off, 0, (size_t)(n + 2) * sizeof(int));
    for (int i = 0; i < m; i++) off[es[i] + 2]++;
    for (int k = 2; k <= n + 1; k++) off[k] += off[k - 1];
    for (int i = 0; i < m; i++) {
        int p = off[es[i] + 1]++;
        rd[p].v = ez[i];
        rd[p].w = ew[i];
    }

    /* ---- count boards and roads: which land is this? ---------------- */
    double dn = (double)n, dm = (double)m;
    double l2 = log2(dn + 2.0);
    double cairn_cost = 6.0 * (dm + dn) * l2 + 4.0 * (dm + dn);
    double walk_cost  = 0.05 * dn * dn + 20.0 * dn;
    int walk_lattice  = (n <= 1024) || (walk_cost <= cairn_cost);
    double visit_budget = cairn_cost / 0.18;   /* slot-visits the walk may spend */

    NNote *heap = NULL;
    int hs = 0;
    int   *act  = NULL;  /* boards bearing a carving, not yet settled */
    double *actd = NULL; /* their notes, kept contiguous for the walk   */
    int   *apos = NULL;  /* board -> place in act, or -1                */
    int na = 0;

    if (walk_lattice) {
        act  = (int    *)malloc((size_t)n * sizeof(int));
        actd = (double *)malloc((size_t)n * sizeof(double));
        apos = (int    *)malloc((size_t)n * sizeof(int));
        if (!act || !actd || !apos) { walk_lattice = 0; }
        else {
            memset(apos, 0xFF, (size_t)n * sizeof(int));   /* all -1 */
            act[0] = source; actd[0] = 0.0; apos[source] = 0; na = 1;
        }
    }

    if (walk_lattice) {
        double visits = 0.0;
        while (na > 0) {
            visits += (double)na;
            int bi = faintest(actd, na);
            int u  = act[bi];
            double du = actd[bi];
            na--;
            if (bi != na) {                  /* swap-remove: settled, never revisited */
                int mv = act[na];
                act[bi] = mv; actd[bi] = actd[na]; apos[mv] = bi;
            }
            apos[u] = -1;

            /* open the cow's head: re-measure by way of this settled board */
            int e = off[u], ee = off[u + 1];
            for (; e < ee; e++) {
                int v = rd[e].v;
                double nd = du + rd[e].w;
                if (nd < dist[v]) {          /* thinner milk: recut the slot */
                    dist[v] = nd;
                    int p = apos[v];
                    if (p >= 0) actd[p] = nd;
                    else { apos[v] = na; act[na] = v; actd[na] = nd; na++; }
                }
            }

            /* if walking the lattice outgrows the birds, stack the notes instead */
            if (visits > visit_budget) {
                heap = (NNote *)malloc((size_t)(m + n + 2) * sizeof(NNote));
                if (heap) {
                    for (int i = 0; i < na; i++) cairn_push(heap, &hs, actd[i], act[i]);
                    na = 0;
                }
            }
        }
    }

    if (!walk_lattice || hs > 0) {
        if (!heap) {
            heap = (NNote *)malloc((size_t)(m + n + 2) * sizeof(NNote));
            if (!heap) { free(off); free(rd); free(act); free(actd); free(apos); return; }
            cairn_push(heap, &hs, 0.0, source);
        }
        while (hs > 0) {
            NNote t = cairn_pop(heap, &hs);
            int u = t.u;
            if (t.d > dist[u]) continue;     /* a spent bird is never sent again */
            int e = off[u], ee = off[u + 1];
            for (; e < ee; e++) {
                int v = rd[e].v;
                double nd = t.d + rd[e].w;
                if (nd < dist[v]) { dist[v] = nd; cairn_push(heap, &hs, nd, v); }
            }
        }
    }

    free(off); free(rd); free(act); free(actd); free(apos); free(heap);
}
