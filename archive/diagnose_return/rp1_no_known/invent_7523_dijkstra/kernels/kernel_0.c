#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif

/* A road as the runner carries it: the far heap and the notch in ONE 16-byte
   object, so both arrive together on a single cache line, one stream. */
typedef struct { double w; int v; int pad; } Road;

/* The hooded figure's ordered walk -- only for the thicket aid and the
   long-thin-land handover, never for the throws themselves. */
typedef struct { double d; int u; } HeapItem;

static void hpush(HeapItem *h, int *hs, double d, int u)
{
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}

static HeapItem hpop(HeapItem *h, int *hs)
{
    HeapItem top = h[0];
    int sz = --(*hs);
    if (sz > 0) {
        HeapItem last = h[sz];
        int i = 0;
        for (;;) {
            int l = 2 * i + 1;
            if (l >= sz) break;
            int r = l + 1;
            int s = (r < sz && h[r].d < h[l].d) ? r : l;
            if (h[s].d >= last.d) break;
            h[i] = h[s];
            i = s;
        }
        h[i] = last;
    }
    return top;
}

/* "read straight across a single row, one sliver from every heap in the same
   breath" -- applied to the embers: the dimmest unsealed ember in the land,
   four lanes per instruction. Returns -1 when the whole land is cold ash. */
static int land_argmin(const double * restrict key, int n)
{
    int best = -1;
    double bv = INFINITY;
#if defined(__AVX2__)
    if (n >= 8) {
        const __m256i lane = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256d vmin = _mm256_set1_pd(INFINITY);
        __m256i vidx = _mm256_set1_epi64x(-1);
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d v  = _mm256_loadu_pd(key + i);
            __m256d lt = _mm256_cmp_pd(v, vmin, _CMP_LT_OQ);
            __m256i id = _mm256_add_epi64(_mm256_set1_epi64x((long long)i), lane);
            vmin = _mm256_min_pd(vmin, v);
            vidx = _mm256_blendv_epi8(vidx, id, _mm256_castpd_si256(lt));
        }
        {
            double mv[4]; long long mi[4];
            int k;
            _mm256_storeu_pd(mv, vmin);
            _mm256_storeu_si256((__m256i *)mi, vidx);
            for (k = 0; k < 4; k++)
                if (mv[k] < bv) { bv = mv[k]; best = (int)mi[k]; }
        }
        for (; i < n; i++)
            if (key[i] < bv) { bv = key[i]; best = i; }
        return best;
    }
#endif
    {
        int i;
        for (i = 0; i < n; i++)
            if (key[i] < bv) { bv = key[i]; best = i; }
    }
    return best;
}

/* ONE LAND-WIDE THROW.
   Every heap whose ember moved last throw sends its runners. Each stick is
   compared against the far heap's own ember as it lands (the per-heap minimum,
   fused into the landing -- losers are burned, never stored). A heap enters the
   next throw at most once, stamped. Heaps whose embers did not move send
   nothing: their garrison stands, dismissible. */
static int one_throw(const int * restrict off, const Road * restrict R,
                     double * restrict dist, int * restrict stamp,
                     const int * restrict frontier, int fn,
                     int * restrict nextf, int round, long long * restrict work)
{
    int nn = 0;
    int k;
    for (k = 0; k < fn; k++) {
        int u  = frontier[k];
        double du = dist[u];
        int e0 = off[u], e1 = off[u + 1], e;
        *work += (long long)(e1 - e0);
        for (e = e0; e < e1; e++) {
            int v = R[e].v;
            double nd = du + R[e].w;
            if (UNLIKELY(nd < dist[v])) {
                dist[v] = nd;
                if (stamp[v] != round) { stamp[v] = round; nextf[nn++] = v; }
            }
        }
    }
    return nn;
}

void kernel(int n, int m, const int *src, const int *dst,
            const double *weight, int source, double *dist_out)
{
    const int    * restrict SRC = src;
    const int    * restrict DST = dst;
    const double * restrict W   = weight;
    double       * restrict dist = dist_out;
    int *off, *cur;
    Road *R;
    int i, thicket;

    if (n <= 0) return;
    for (i = 0; i < n; i++) dist[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist[source] = 0.0;                    /* the ember banked nearly out, never out */
    if (m <= 0) return;

    /* ---- lay out the roads leaving each place (CSR, counting sort) ---- */
    off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    cur = (int *)malloc((size_t)(n + 1) * sizeof(int));
    R   = (Road *)malloc((size_t)m * sizeof(Road));
    if (!off || !cur || !R) { free(off); free(cur); free(R); return; }
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (i = 0; i < m; i++) off[SRC[i] + 1]++;
    for (i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)(n + 1) * sizeof(int));
    for (i = 0; i < m; i++) {
        int u = SRC[i];
        int p = cur[u]++;
        R[p].v = DST[i];
        R[p].w = W[i];
        R[p].pad = 0;
    }
    free(cur);

    /* ---- the native counts the roads against the places ----
       a thicket (or a land one breath wide): runners would trample each other,
       so the hooded figure reads the whole land each throw instead. */
    thicket = (n <= 256) || ((double)m * 24.0 >= (double)n * (double)n);

    if (thicket) {
        double *key    = (double *)malloc((size_t)n * sizeof(double));
        char   *sealed = (char *)calloc((size_t)n, 1);
        int it;
        if (!key || !sealed) { free(key); free(sealed); free(off); free(R); return; }
        for (i = 0; i < n; i++) key[i] = dist[i];
        for (it = 0; it < n; it++) {
            int u = land_argmin(key, n);
            int e, e1;
            double du;
            if (u < 0) break;              /* only cold ash left: forever unlit */
            key[u] = INFINITY;             /* the garrison seals this heap */
            sealed[u] = 1;
            du = dist[u];
            e1 = off[u + 1];
            for (e = off[u]; e < e1; e++) {
                int v = R[e].v;
                double nd = du + R[e].w;
                if (nd < dist[v] && !sealed[v]) { dist[v] = nd; key[v] = nd; }
            }
        }
        free(key); free(sealed); free(off); free(R);
        return;
    }

    /* ---- the throws ---- */
    {
        int *frontier = (int *)malloc((size_t)n * sizeof(int));
        int *nextf    = (int *)malloc((size_t)n * sizeof(int));
        int *stamp    = (int *)malloc((size_t)n * sizeof(int));
        int fn, round = 0, handover = 0;
        long long work = 0;
        long long budget = 4LL * (long long)m + 8LL * (long long)n;

        if (!frontier || !nextf || !stamp) {
            free(frontier); free(nextf); free(stamp); free(off); free(R); return;
        }
        for (i = 0; i < n; i++) stamp[i] = 0;
        fn = 0;
        frontier[fn++] = source;

        while (fn > 0) {
            int nn;
            int *t;
            round++;
            nn = one_throw(off, R, dist, stamp, frontier, fn, nextf, round, &work);
            t = frontier; frontier = nextf; nextf = t;
            fn = nn;
            /* more sandals worn than four times the roads, heaps still unsealed:
               a long thin land. Call the runners home. */
            if (work > budget && fn > 0) { handover = 1; break; }
        }

        if (handover) {
            char *sealed  = (char *)calloc((size_t)n, 1);
            HeapItem *heap = (HeapItem *)malloc(((size_t)n + (size_t)m + 2) * sizeof(HeapItem));
            if (sealed && heap) {
                int hs = 0;
                for (i = 0; i < n; i++)
                    if (dist[i] < INFINITY) hpush(heap, &hs, dist[i], i);
                while (hs > 0) {
                    HeapItem top = hpop(heap, &hs);
                    int u = top.u, e, e1;
                    double du;
                    if (sealed[u]) continue;
                    if (top.d > dist[u]) continue;
                    sealed[u] = 1;
                    du = dist[u];
                    e1 = off[u + 1];
                    for (e = off[u]; e < e1; e++) {
                        int v = R[e].v;
                        double nd = du + R[e].w;
                        if (nd < dist[v]) { dist[v] = nd; hpush(heap, &hs, nd, v); }
                    }
                }
            } else {
                /* no memory for the ordered walk: keep throwing to convergence.
                   Slower, still exact. */
                while (fn > 0) {
                    int nn, *t;
                    round++;
                    nn = one_throw(off, R, dist, stamp, frontier, fn, nextf, round, &work);
                    t = frontier; frontier = nextf; nextf = t;
                    fn = nn;
                }
            }
            free(sealed); free(heap);
        }

        free(frontier); free(nextf); free(stamp);
    }

    free(off); free(R);
    /* dist_out already holds the whole land's embers, read off in one pass. */
}
