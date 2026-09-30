## MAPPING

**SEED 1 — "A stone placed on a corner marks that place's distance as finally known and never reconsidered."**

| world object | problem object |
|---|---|
| grid of triangular houses | the directed graph `(n, m, src, dst, weight)` |
| corner / house | node `0..n-1` |
| traveler's corner | `source` |
| white stone, "distance from itself is nothing" | `dist[source] = 0`, written before anything else |
| black stone | node removed from the candidate set (`key[u] ← INFINITY`) |
| "thread's length set down as final, never touched again" | `dist_out[u]` frozen; no later write can lower it (non-negative weights) |
| bare corner, "an insight age that never arrives" | `INFINITY` for unreachable nodes |

Breaks: *nothing new.* This seed **restates** "each place's distance must be finalized before its neighbors are explored." It is the one seed that agrees with the textbook.

**SEED 2 — "A duck carries a spooled thread whose length is the running sum of every road crossed to reach it."**

| world object | problem object |
|---|---|
| duck | one in-flight tentative distance travelling along one out-edge |
| "loose the ducks" from a stoned corner | the relaxation sweep over `off[u]..off[u+1]` — a *batch*, not one-at-a-time |
| brilliant thread spooled to that road's centimetres | `weight[e]` |
| thread carried on its back = running sum | `nd = dist[u] + w(u,v)` |
| "only if that sum is shorter than the thread the far corner already wears; otherwise throw the new thread away" | `if (nd < dist[v]) dist[v] = nd;` — **a plain array compare-and-store. No queue is touched.** |
| roads toward "houses with no far side", hung slack, thrown away unspooled, never measured again | out-degree-0 nodes are never enqueued/never selected; their distance is just the min of incoming relaxations |
| the duck itself is never stored anywhere | there is **no** container of pending items at all |

Breaks: **"a priority structure must be consulted before every relaxation."** In the native's world the duck *is* the message and it dies on arrival; the only survivor of a relaxation is a number in the far corner's own slot. There is nowhere to push.

**SEED 3 — "Each round, the leader duck holding the shortest carried thread among the unstoned is found and settled before any other duck moves."**

| world object | problem object |
|---|---|
| "wait for stillness / let the wandering ducks settle" | a full round barrier: all relaxations of this round complete before any selection |
| "walk the grid with my hand hovering, comparing thread against thread" | a **contiguous linear sweep** over the key array — a min-reduction, not a heap |
| "it may hide among its own long cousins, so I trace every shot and shuttle back to its landing before I trust it" | no early exit: pass 1 vector-reduces to the true minimum value, pass 2 re-walks to confirm *which* corner carries it |
| leader duck | `argmin` over unstoned keys |
| stoning it | settle |

Breaks: this seed *keeps* "found by comparing against every remaining place" — but that assumption is only a *silent* one for the array variant; for the heap it is false. Combined with Seed 2 it forces the O(n²) scan variant. It is the necessary consequence of Seed 2, not an independent choice.

## CHOSEN SEED

**SEED 2.** It is the only one of the three that breaks the preferred assumption ("a priority structure must be consulted before every relaxation"), and its mapping is the most literal: a duck is a single edge traversal carrying one `double`; relaxation is `nd < dist[v]`; and there is *no object anywhere in the native's world that corresponds to a heap*. Seed 3 is adopted as its forced corollary — if no structure is consulted on push, the leader must be found by sweeping — and Seed 1 supplies the settled/unsettled bookkeeping.

The native also gives, for free, a pruning rule the textbook never states: **"roads toward houses with no far side I throw away unspooled, never measured again."** A sink never relaxes anything, so its selection is pure waste. I take this literally: sinks are permuted to the tail of the node numbering and never enter the sweep (dense path) or the flock (sparse path).

## ASSUMPTION BROKEN

> *a priority structure must be consulted before every relaxation*

In the artifact, the inner relaxation loop of the dense path is:

```c
int v = E[e]; double nd = best + W[e];
if (nd < D[v]) { D[v] = nd; K[v] = nd; }
```

Two contiguous streaming loads, one random compare, two random stores. No push, no sift, no `log n`, no branch into a container. The cost of ordering is paid **once per round** by a SIMD sweep, not **once per edge** by a heap insert. Secondarily, the sink rule breaks *"the whole graph must be explored"* in miniature: `n − k` corners are never visited at all, only written into.

**Regime recognition (required, since `known_way` names two regimes).** The native must also be able to tell a *weaver grid* (few houses, many short roads between them — the hand can hover over the whole grid faster than the ducks can be sorted) from a *sprawl* (vast, thinly-roaded — the hand would walk forever). So before loosing a single duck he counts houses against roads: `k·k` hand-walks versus `~20·(m+n)·log₂n` sorted-flock operations. If the grid is a weave, he walks it with his hand; if it is a sprawl, he falls back to keeping the ducks in a sorted flock — a 4-ary lazy heap, still with the sink rule and still with no `done` array (a stale duck is recognised because its thread is longer than the thread its corner already wears).

**Deliberately not used: threads.** The metaphor's unit of work per round is one sweep of `k` doubles. At the sizes where the sweep is chosen (`k` small enough that `k² ≲ 20·m·log n`), one sweep is ~1–50 µs, and there are `k` rounds, i.e. `k` barriers. An OpenMP barrier (~1–3 µs) would eat the entire sweep. Vectorization only.

## ARTIFACT

```c
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
```

## PREDICTION

PREDICTION: speedup_vs_heap = 2.5

Stated before any measurement, with the falsifiable per-regime breakdown I actually believe (this is where I expect to be shown wrong first):

| regime | path taken | predicted speedup vs reference binary-heap |
|---|---|---|
| n ≤ ~1500, any density | weave | 1.5 – 3× (CSR build starts to dominate both) |
| n ~ 2k–8k, avg degree ≥ 50 | weave | 4 – 8× |
| n ~ 5k–20k, avg degree ~10–20 | borderline; selector should pick heap | 1.2 – 1.5× |
| n ≥ 50k, avg degree ≤ 20 | sprawl | 1.2 – 1.6× (4-ary heap + no `done` array + sink prune) |
| n ≥ 50k, avg degree ≥ ~n/300 | weave | 2 – 4× |

Cost model behind the selector: sweep ≈ 0.04–0.06 ns per element (1.5 contiguous passes, 4 doubles/vminpd, L2-resident at the sizes where it is selected); heap ≈ 1.1 ns per (edge × log₂n). Crossover at `k² ≈ 20·(m+n)·log₂n`.

## MEASUREMENT

**Not measured.** `dijkstra_bench` and `dijkstra_contract` were not callable in this session (no tools were exposed), so I am reporting this as a failure to measure rather than dressing up the model as a result. Nothing below the PREDICTION line was tuned against data — the constant `20.0` comes from the cost model above, not from a fit.

What the pipeline should check, in order, and what would falsify me:

1. **Correctness first** (`dijkstra_contract`): exact match on unreachable-node `INFINITY`, zero-weight edges, self-loops, a source with out-degree 0, `m = 0`, duplicate parallel edges, and a graph with many sinks (the permutation path). The two riskiest lines are `nd = best + W[e]` (uses the scanned minimum rather than re-loading `D[u]` — valid only because `K[u] == D[u]` for every unstoned corner) and `key[i] == best` in pass 2 (exact bit equality with a value copied out of the same array).
2. **Which path fired.** If the bench is a single large sparse graph, the selector takes the sprawl branch and the honest ceiling is ~1.5×, not 2.5 — my headline prediction would then be wrong by ~1.7×, and I would rather say that now than explain it afterwards.
3. **The selector constant.** If the bench sits near `k² ≈ 20(m+n)log n`, sweep `20.0` over `{8, 14, 20, 28, 40}`; a wrong constant here costs more than everything else in the kernel.

## VERDICT

The native's picture is not a metaphor for Dijkstra-with-a-heap; it is a metaphor for Dijkstra-**without** one. The duck dies at the far corner, so the only place a tentative distance can live is the corner's own slot — which forces the O(n²) array-scan variant, and the "hand hovering over the grid" is a min-reduction, which is the one part of shortest-paths that vectorizes perfectly. That is the whole content of the win: the reference pays `log n` pointer-chasing ops *per edge*; this pays a streaming SIMD sweep *per settled node*. It is a real, well-known algorithm, but the seed reached it from the right direction rather than being retrofitted.

The seed also handed over one thing the textbook does not say out loud — the slack-road rule. Sink nodes are never candidates for selection, only targets of writes. In the weave path this shortens every one of the `k` sweeps; in the sprawl path it removes heap traffic. On a graph with no sinks it costs one identity permutation and nothing else.

**Stated risk, and how it is addressed.** The mechanism is *catastrophically* worse than the known way on large sparse graphs: `k²` against `(m+n)log n` is unbounded, a 100× loss at n = 10⁶, m = 10⁷. Per step 4 I have not shipped that risk unguarded — the regime test runs at construction time, is O(n+m) (already paid for the CSR build), and falls back to a complete, improved heap path. The fallback is not a stub: it is strictly better than the reference (4-ary sift halves the depth on the push-dominated workload, the lazy `du > D[u]` test removes the `done` array entirely, and the sink prune is itself guarded by a density test so it costs no random `deg[v]` load when sinks are rare). Thread parallelism was considered and dropped, not forgotten: the metaphor's round barrier would put `k` OpenMP barriers on the critical path, each comparable to the sweep it guards.

**Where I expect to be wrong.** The selector constant. Everything else here is either provably correct or a modest constant-factor bet; `20.0` is a single number standing between a 5× win and a 100× loss, and it was derived from a cost model, not measured. If exactly one thing gets tuned against `dijkstra_bench`, it should be that.