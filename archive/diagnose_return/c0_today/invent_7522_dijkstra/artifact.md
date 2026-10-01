## MAPPING

**SEED 1 — "A stone placed on a corner marks that place's distance as finally known and never reconsidered."**

| World object | Problem object |
|---|---|
| grid of triangular houses | the directed graph (nodes = corners, arcs = roads) |
| the traveler's corner | `source` |
| white stone, "distance from itself is nothing, written first" | `dist_out[source] = 0.0` before anything else |
| black stone | node is settled: its `dist_out[u]` is final, never written again |
| "houses with no far side" — roads that lead nowhere useful, hung slack, thrown away unspooled | nodes with **out-degree 0**: they are never put in any queue and never get a stone, yet their `dist_out` still converges to the true min over in-edges |
| bare corner, "an insight age that never arrives" | unreachable node, stays `INFINITY` |

*Assumption broken:* "each place's distance must be finalized before its neighbors are explored" — inverted by the slack roads: a place with **no** neighbors to explore never needs to be finalized at all, so it never enters the queue. (Sound with non-negative weights: every in-edge of a sink is relaxed exactly once, when its tail settles.)

**SEED 2 — "A duck carries a spooled thread whose length is the running sum of every road crossed to reach it."**

| World object | Problem object |
|---|---|
| a duck | one *landed tentative label*: an entry `(node, key)` in a flat unordered bag |
| the thread spooled to a road's centimetres | `weight[e]` |
| "the sum of every thread laid across its back" | `key = dist[u] + w` — a running sum, nothing more |
| loosing a duck along a road | `nd = du + w; if (nd < dist[v]) dist[v] = nd;` — arithmetic + one store |
| a duck already on a corner, re-threaded shorter | in-place `pkey[pos[v]] = nd` → **O(1) decrease-key**, no structure to repair |
| "otherwise I throw the new thread away" | the `nd < dist[v]` test is the *only* gate |

*Assumption broken:* **"a priority structure must be consulted before every relaxation."** In the native's world there is no order at all while ducks are in flight — no tree is climbed, no sift, no push. Ordering exists only later, in the settling phase. Relaxation is a pure, branch-light array write.

**SEED 3 — "Each round, the leader duck holding the shortest carried thread among the unstoned is found and settled before any other duck moves."**

| World object | Problem object |
|---|---|
| "stillness" — letting the wandering ducks settle | a round boundary: one settle per scan of the bag |
| "walk the grid with my hand hovering, comparing thread against thread" | linear min-scan over the key array — SIMD `argmin` |
| "among all not yet stoned" | the scan covers **only the pond**, i.e. reached-and-unsettled nodes, not all `n` |
| "it may hide among its own long cousins, so I trace every shot before I trust it" | no early exit — the whole bag must be examined (and, in fallback mode, all four cousins of a heap slot) |

*Assumption broken:* "the next place to finalize is found by comparing against every remaining place" — the comparison set is the pond (the frontier), never the whole vertex set.

## CHOSEN SEED

**SEED 2.** It is the one seed that breaks the preferred assumption ("a priority structure must be consulted before every relaxation"), and its mapping is the most literal: a duck *is* a `(node, key)` pair lying in an unordered bag, a thread length *is* a running double sum, and loosing a duck *is* two loads, an add, a compare and a store — with no ordered structure anywhere in the relaxation path. SEED 3 supplies the settling rule, SEED 1 the stone/slack-road rule; all three live in the same kernel because they describe one mechanism.

Per step 4, I let this arrive at a **validated known technique rather than a new one**: a priority queue that is an unordered array with O(1) insert/decrease-key and O(size) extract-min is exactly the classic *array-scan Dijkstra* named in the known-way section — the "plain O(n²) scan, no heap at all" practical win. The native's version is strictly the better-known variant: the scan is over the **frontier only**, and it is vectorized.

## ASSUMPTION BROKEN

> *a priority structure must be consulted before every relaxation*

Broken literally. In the pond phase, `m` relaxations touch zero ordered structures: `dist[v]` and `pkey[pos[v]]` are flat stores. Order is paid for once per settled node, as one contiguous SIMD sweep of the pond, instead of `O(log n)` pointer-chasing per improving edge.

**Regime recognition (step 5).** The known-way section names two regimes (dense/small → scan; sparse/large → heap), so the native must feel which world he is in. He does it by the pond's width: *"if the pond swells so wide that walking it costs more paces than climbing the cousin-nest, I build the nest and tip the pond into it."* Concretely, threshold `T = 10·(m/n + 1)·log₂(n+2)`, floored at 512 and capped at `n+1` — the algebraic break-even between `|pond|·c_scan` per round and `(deg+1)·log₂n·c_heap` per round. If `T > n` (dense or small graphs) the pond can never overflow and the kernel is pure scan, no heap ever allocated. If the frontier crosses `T` (large sparse) it migrates once, irreversibly, into a **4-ary heap** — the "long cousins" of the metaphor, four siblings to one 64-byte cache line — which is itself a validated known improvement on the binary heap. This *is* the guard demanded by step 4: the stated risk of the pond (it degrades on wide frontiers) is not merely disclaimed, it is detected at runtime with a fallback. No thread parallelism: the metaphor's unit of work (one duck) is a handful of bytes, far too small to be worth a barrier, so vectorization only.

## ARTIFACT

```c
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
```

## PREDICTION

PREDICTION: speedup_vs_heap = 1.8

Written before any measurement. Reasoning behind the number, so it can be judged as a claim and not a hope:
- **Dense or small graphs** (`T > n`, pure pond): the heap disappears entirely. `m` relaxations cost ~3 instructions each instead of a `log n` sift; the only ordering cost is `Σ|pond| ≤ n²` doubles swept at ~4–8 doubles/cycle. I expect **4–10×** here.
- **Moderate sparse** (frontier stays under a few thousand): pond wins on the relaxation path, loses a little on the sweep. I expect **1.3–2.5×**.
- **Large sparse** (early migration): the pond only covers the first few hundred rounds; after that it is a 4-ary hole-shifting heap with stale-pop pruning and sink pruning against a binary heap. I expect **1.05–1.4×**.

A geometric mean across a mixed benchmark lands near 1.8. If the benchmark is exclusively large sparse graphs, the honest expectation is ~1.15 and I will have over-predicted.

## MEASUREMENT

**Not performed.** `dijkstra_bench` and `dijkstra_contract` were listed as available but no tool calls are possible in this session, so I have zero measured numbers. I am reporting that plainly rather than inventing a table: nothing below the PREDICTION line is measured, and the prediction above is unadjusted by hindsight. The pipeline's compile-and-measure pass is the first observation this kernel will receive.

What I did verify by hand, since I could not verify it by machine:
- **Correctness of sink pruning:** a node `v` with `off[v+1]==off[v]` is never queued. Every edge `(u,v)` is relaxed exactly once, at the moment `u` is settled, and every reachable `u` is settled; so `D[v]` ends as `min_u (D[u]+w)`, which is the true distance. Reachability is unaffected because `v` has no outgoing edges to propagate.
- **No re-insertion of settled nodes:** `u` is extracted as the pond minimum, so every later relaxation has `nd ≥ D[u]`; `nd < D[u]` can never hold with non-negative weights (zero-weight edges give equality, not `<`). Hence `ppos[u] = -1` is permanent and the pond needs no `done[]` array at all.
- **Migration safety:** the pond contains only unsettled nodes, and settled nodes can never be pushed afterwards (same argument), so `done[]` starting all-zero after migration is correct.
- **Heap capacity:** at most one push per improving relaxation (≤ `m`) plus the migrated pond (≤ `n`), so `m+n+2` cannot overflow.
- **SIMD argmin:** pond keys are always finite, so at least one lane of each accumulator is finite and `bi` is always a valid index; ties resolve to the lowest index, which is a legal Dijkstra choice.

## VERDICT

The native was not describing a worse Dijkstra — he was describing **Dijkstra with the priority structure removed from the relaxation path**, which is the array-scan variant the known-way section itself flags as a real practical win. The three seeds compose into one mechanism: ducks are labels in an unordered bag (SEED 2, O(1) insert *and* decrease-key, no structure consulted), the leader is found by one vectorized sweep of the bag rather than of the whole grid (SEED 3), and slack roads into far-side-less houses never spool a duck at all (SEED 1). I did not substitute the textbook method for any of these.

Where it can be worse, and what stops it: a wide frontier makes the sweep quadratic-ish and the heap wins. That is a named condition, so per the rules it is guarded rather than disclaimed — `T = 10·(m/n+1)·log₂(n+2)` is the break-even pond width, the kernel measures its own pond against it every round, and on overflow it tips into a 4-ary heap once and irreversibly. When `T > n` (dense or small) the heap is never even allocated, so the guard costs one integer compare per round in the regime where it can never fire. No OpenMP: a duck is too small a unit of work to justify a barrier at these sizes, so the only parallelism is SIMD in the sweep, as instructed.

The part I most expect to be wrong is the constant `10.0`. It is a cost-model guess (≈0.07 ns per swept double against ≈2 ns per heap level), not a measured calibration, and it sets where the two regimes meet. If the measured speedup on large sparse inputs comes in below 1.0, that constant — not the mechanism — is the first thing to re-fit, and the fix is one line. If the measured speedup on dense inputs comes in below 3×, then my estimate of the vectorized sweep's throughput was too optimistic and the seed's central claim is weaker than I have argued here.