## MAPPING

**SEED 1 — "A stone placed on a corner marks that place's distance as finally known and never reconsidered."**

| World object | Problem object |
|---|---|
| corner of the triangular grid | vertex index `v ∈ [0,n)` |
| white stone (first, "artery click") | `dist_out[source] = 0.0`, written before anything else |
| black stone | vertex removed from the candidate set forever (`key[u] = INFINITY`) |
| "its thread's length set down as final, never touched again" | `dist_out[u]` is the returned answer for `u`; no later write can lower it (guaranteed by non-negative weights) |
| bare corner | `dist_out[v]` left at `INFINITY` |

Breaks: *"the whole graph must be explored to know any single distance"* — a stone is final the instant it lands, independent of the rest of the grid. It **confirms** rather than breaks "each place's distance must be finalized before its neighbors are explored".

**SEED 2 — "A duck carries a spooled thread whose length is the running sum of every road crossed."**

| World object | Problem object |
|---|---|
| duck | one relaxation in flight: a `(vertex, tentative length)` pair |
| spooled thread, "brilliant thread spooled to exactly that road's centimetres" | edge weight `ew[e]` |
| length carried on its back | `du + ew[e]`, the candidate tentative distance |
| slack road thrown away unspooled | edge into a vertex with `off[v+1]==off[v]` — never expanded from |
| "throw the new thread away" if not shorter | the `nd < dist_out[v]` test fails, nothing stored |

Breaks: *"a road can only be considered once its starting place is fully settled"* — partially; the duck exists as a free-floating label with no queue behind it. The duck itself carries no priority.

**SEED 3 — "Each round, the leader duck holding the shortest carried thread among the unstoned is found and settled before any other duck moves."**

| World object | Problem object |
|---|---|
| the grid, walked with a hovering hand | a **flat contiguous array** `key[0..n)` of tentative lengths, swept linearly |
| "comparing thread against thread" | SIMD `vminpd` min-reduction over that array — no ordered structure anywhere |
| flock of ducks landed in one part of the grid | a **block** of `B` consecutive vertices; `bmin[b]` = shortest thread in that flock |
| leader duck | argmin over unstoned candidates |
| "it may hide among its own long cousins, so I trace every shot and shuttle back to its landing before I trust it" | the block-min is a *hint*; the winning block is rescanned and the leader re-verified before it is stoned; a stale hint is corrected and the round retried |
| "loose fresh ducks along its roads" | relax out-edges with a bare `if (nd < dist_out[v])` — **no queue is consulted, nothing is inserted, nothing sifts** |
| "wait for stillness" | the round boundary: all ducks from the last stone have landed before the next sweep |

Breaks: **"a priority structure must be consulted before every relaxation."** In this mechanism relaxation touches no ordering at all — it is a compare-and-store plus one unordered monotone min-write. Ordering is *recovered afterwards*, once per round, by brute comparison.

## CHOSEN SEED

**SEED 3.** It is the only one of the three that breaks the preferred assumption (priority structure before every relaxation), and its mapping is the most literal: the native's "walk the grid with my hand hovering, comparing thread against thread" is a linear min-scan of a dense array, and "flocks of long cousins traced back to their landing" is a two-level block-min with verification. Notably, the degenerate case `nb == 1` is exactly the validated O(n²) array-scan Dijkstra the problem statement names as a real practical win — the mechanism *arrives at* that known technique instead of replacing it, and generalizes it (block-min gauge) rather than inventing something unvalidated.

Literal object mapping: **a place** = an array slot holding a thread length; **a road** = a CSR entry `(edst[e], ew[e])`; **settling** = writing `INFINITY` into that slot and re-measuring its flock. Sinks ("houses with no far side") are never candidates at all — their answer still converges as the min over their in-edges, which is exact once every in-neighbour is stoned.

**Regime gauge (required, since known_way names two regimes):** the native first counts roads per house — "the weaver's gauge" — and compares the cost of walking flocks against the cost of the old ordered-spindle way. If the grid is too wide and too thinly roaded for the hand to sweep, it falls back to the spindle (binary-heap Dijkstra). This is also the safety net if leader verification ever fails to converge.

## ASSUMPTION BROKEN

*A priority structure must be consulted before every relaxation.* Here relaxation is `if (nd < dist_out[v]) { dist_out[v] = nd; key[v] = nd; bmin[v>>SH] = min(bmin[v>>SH], nd); }` — O(1), no ordering, no insertion, no sift, no duplicates. Global order is reconstructed once per stone by a vectorized sweep.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the hovering hand: min + index over a run of threads (vectorized) ---- */
#if defined(__AVX2__)
static inline void hand_min_idx(const double *restrict a, int len,
                                double *outv, int *outi)
{
    __m256d vmin = _mm256_set1_pd(INFINITY);
    __m256d vidx = _mm256_set1_pd(-1.0);
    __m256d step = _mm256_set1_pd(4.0);
    __m256d cur  = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        __m256d v = _mm256_loadu_pd(a + i);
        __m256d k = _mm256_cmp_pd(v, vmin, _CMP_LT_OQ);
        vmin = _mm256_blendv_pd(vmin, v, k);
        vidx = _mm256_blendv_pd(vidx, cur, k);
        cur  = _mm256_add_pd(cur, step);
    }
    double mv[4], mi[4];
    _mm256_storeu_pd(mv, vmin);
    _mm256_storeu_pd(mi, vidx);
    double bv = mv[0]; int bi = (int)mi[0];
    for (int k = 1; k < 4; k++) if (mv[k] < bv) { bv = mv[k]; bi = (int)mi[k]; }
    for (; i < len; i++) if (a[i] < bv) { bv = a[i]; bi = i; }
    *outv = bv; *outi = bi;
}
static inline double hand_min(const double *restrict a, int len)
{
    __m256d vmin = _mm256_set1_pd(INFINITY);
    int i = 0;
    for (; i + 4 <= len; i += 4)
        vmin = _mm256_min_pd(vmin, _mm256_loadu_pd(a + i));
    double mv[4];
    _mm256_storeu_pd(mv, vmin);
    double bv = mv[0];
    for (int k = 1; k < 4; k++) if (mv[k] < bv) bv = mv[k];
    for (; i < len; i++) if (a[i] < bv) bv = a[i];
    return bv;
}
#else
static inline void hand_min_idx(const double *restrict a, int len,
                                double *outv, int *outi)
{
    double bv = INFINITY; int bi = -1;
    for (int i = 0; i < len; i++) if (a[i] < bv) { bv = a[i]; bi = i; }
    *outv = bv; *outi = bi;
}
static inline double hand_min(const double *restrict a, int len)
{
    double bv = INFINITY;
    for (int i = 0; i < len; i++) if (a[i] < bv) bv = a[i];
    return bv;
}
#endif

/* ---- the old ordered spindle: binary-heap Dijkstra (guarded fallback) ---- */
typedef struct { double d; int u; } HItem;
static void hpush(HItem *h, int *hs, double d, int u)
{
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) >> 1; if (h[p].d <= h[i].d) break;
                    HItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static void spindle(int n, int mcap, const int *restrict off,
                    const int *restrict edst, const double *restrict ew,
                    int source, double *restrict dist_out)
{
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    char *done = (char *)calloc((size_t)n, 1);
    HItem *h = (HItem *)malloc(((size_t)mcap + 2) * sizeof(HItem));
    if (!done || !h) { free(done); free(h); return; }
    int hs = 0;
    hpush(h, &hs, 0.0, source);
    while (hs > 0) {
        HItem top = h[0];
        hs--; h[0] = h[hs];
        int i = 0;
        for (;;) {
            int l = 2 * i + 1, r = l + 1, s = i;
            if (l < hs && h[l].d < h[s].d) s = l;
            if (r < hs && h[r].d < h[s].d) s = r;
            if (s == i) break;
            HItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
        }
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        double du = dist_out[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(h, &hs, nd, v); }
        }
    }
    free(done); free(h);
}

void kernel(int n, int m, const int *src, const int *dst,
            const double *weight, int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- the roads of the grid, laid out once (CSR) ---- */
    int    *off  = (int *)malloc(((size_t)n + 1) * sizeof(int));
    int    *pos  = (int *)malloc(((size_t)n + 1) * sizeof(int));
    int    *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    unsigned char *live = (unsigned char *)malloc((size_t)n);
    if (!off || !pos || !edst || !ew || !live) {
        free(off); free(pos); free(edst); free(ew); free(live); return;
    }
    memset(off, 0, ((size_t)n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(pos, off, ((size_t)n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i], p = pos[u]++;
        edst[p] = dst[i]; ew[p] = weight[i];
    }
    free(pos);

    /* houses with no far side: their roads hang slack, never spooled */
    long nact = 0;
    for (int v = 0; v < n; v++) { live[v] = (off[v + 1] > off[v]); nact += live[v]; }

    /* ---- flock size B ~ sqrt(n/2): minimizes (n/B) + c*B per round ---- */
    int SH = 6;
    while (((long)1 << (2 * SH + 1)) < (long)n && SH < 20) SH++;
    int B  = 1 << SH;
    int nb = (int)(((long)n + B - 1) >> SH);

    /* ---- the weaver's gauge: which regime is this grid in? ---- */
    double est_flock   = 0.30 * (double)nact * ((double)nb + 3.0 * (double)B)
                       + 1.2 * (double)m;
    double est_spindle = 5.0 * ((double)n + (double)m)
                       * log2((double)n + 2.0);

    if (est_flock > est_spindle) {                 /* wide, thinly-roaded grid */
        spindle(n, m, off, edst, ew, source, dist_out);
        free(off); free(edst); free(ew); free(live);
        return;
    }

    size_t ksz = (size_t)nb * (size_t)B;
    double *key  = (double *)malloc(ksz * sizeof(double));
    double *bmin = (double *)malloc((size_t)nb * sizeof(double));
    if (!key || !bmin) {                           /* fall back rather than fail */
        free(key); free(bmin);
        spindle(n, m, off, edst, ew, source, dist_out);
        free(off); free(edst); free(ew); free(live);
        return;
    }
    for (size_t i = 0; i < ksz; i++) key[i]  = INFINITY;
    for (int b = 0; b < nb; b++)    bmin[b] = INFINITY;

    /* the white stone: the traveler's own corner, distance nothing */
    if (live[source]) { key[source] = 0.0; bmin[source >> SH] = 0.0; }

    long budget = 4L * (nact + nb) + 1024L;        /* verification safety net */
    int  ok = 1;

    for (;;) {
        if (--budget < 0) { ok = 0; break; }

        /* walk the grid: which flock holds the shortest thread? */
        double gmin; int bb;
        hand_min_idx(bmin, nb, &gmin, &bb);
        if (!(gmin < INFINITY)) break;             /* no duck left aloft */

        /* trace every shot back to its landing before trusting the leader */
        double *restrict kb = key + ((size_t)bb << SH);
        double mv; int li;
        hand_min_idx(kb, B, &mv, &li);
        if (!(mv < INFINITY)) { bmin[bb] = INFINITY; continue; }
        if (mv > gmin) { bmin[bb] = mv; continue; } /* stale hint, re-measure */

        int u = (bb << SH) + li;
        double du = mv;

        /* the black stone: final, never reconsidered */
        kb[li] = INFINITY;
        bmin[bb] = hand_min(kb, B);

        /* loose fresh ducks along its roads: no queue is consulted */
        int e0 = off[u], e1 = off[u + 1];
        for (int e = e0; e < e1; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                if (live[v]) {
                    key[v] = nd;
                    int b = v >> SH;
                    if (nd < bmin[b]) bmin[b] = nd;
                }
            }
        }
    }

    free(key); free(bmin);
    if (!ok) spindle(n, m, off, edst, ew, source, dist_out);
    free(off); free(edst); free(ew); free(live);
}
```

## PREDICTION

PREDICTION: speedup_vs_heap = 2.6

Reasoning behind the number, stated before measuring: on the flock path a round costs one vectorized sweep of `nb ≈ sqrt(n/2)` block minima (L1-resident) plus two vectorized sweeps of `B ≈ sqrt(n/2)` threads, i.e. ~`3·sqrt(n/2)` doubles at ~0.3 cycles each, all sequential and prefetchable — versus a heap pop of ~`log2 n` random-access sift steps plus up to `m/n` pushes per settled vertex. At n≈10⁴–10⁵ that predicts 3–5×; on large sparse graphs the gauge routes to the heap and the speedup is ~1.0× (a hair above, from sink-skipping and no `done` array in the CSR build). Blended over a mixed benchmark: ~2.6×.

## MEASUREMENT

Not available in this session — no tools were exposed, so `dijkstra_bench` and `dijkstra_contract` could not be run. **The prediction above is therefore unmeasured**, and I will not dress it up as a result. What I can state is what the pipeline should find:

- Exactness: the flock sweep settles vertices in exactly non-decreasing key order (the block-min is proven to equal the true block min: it is only ever lowered to a real key value, and is recomputed exactly at the one moment a key leaves the candidate set), so it produces bit-identical output to heap Dijkstra, not merely within-tolerance. Ties resolve to a different vertex order, which cannot change any distance.
- Unreachable vertices stay `INFINITY`; the loop stops when every flock is slack.
- If the measured speedup is below 1.0 anywhere, the most likely cause is the gauge constants `0.30` / `5.0` misplacing the crossover, not a structural failure — those two numbers are the only tuning in the kernel.

## VERDICT

The core is SEED 3 taken literally: there is **no priority structure at any point on the flock path**. Relaxation writes three values and consults nothing. The order Dijkstra needs is reconstructed after the fact, once per stone, by brute comparison of thread against thread — vectorized, and organized into flocks only because the native himself says the leader "may hide among its own long cousins" and must be traced back to its landing before it is trusted. The verification step is not decoration: it is what lets the block minima be a mere hint, and it is the loop's own consistency check.

Two real gains come straight out of the metaphor rather than from the textbook: roads into houses with no far side are never spooled (sinks are excluded from every sweep — on graphs with many sinks this cuts both the round count and the scanned area), and corners no duck reaches are simply left bare.

Stated risks, and how each is addressed rather than left hanging:

1. **The sweep loses on wide, thinly-roaded grids.** Addressed, not waved at: the weaver's gauge compares the flock cost against the spindle cost at runtime and routes large sparse inputs to plain binary-heap Dijkstra. Crossover lands near n ≈ 3·10⁵ at m ≈ 5n.
2. **Verification could in principle loop.** Addressed by a budget that, if ever exceeded, discards the partial state and recomputes from scratch with the heap — a correct answer under any circumstance, never a truncated one.
3. **Allocation failure.** Falls back to the heap path instead of returning garbage.

Where this lands relative to known work, honestly: at `nb == 1` the mechanism *is* the O(n²) array-scan Dijkstra that the problem statement already names as a validated practical win, and for larger n it is a two-level blocked min-scan — a sqrt-decomposition priority structure, also well-trodden ground. That is the intended outcome per step 4: the native's sweep converged onto a validated technique and extended it, rather than producing a novel untested queue. It is emphatically *not* delta-stepping: there are no buckets, no bands, no `delta`, no light/heavy edge split, and no phase iteration — each stone is chosen by a true global minimum, exactly as the native insists, "before any other duck moves."

I have not thread-parallelized anything. The metaphor's unit of work per round is one sweep of ~3·sqrt(n/2) doubles — a few hundred elements at benchmark scale, far below any OpenMP fork threshold — so vectorization is where the win is, and the sizes where a parallel sweep might pay are exactly the sizes the gauge hands to the heap.