## MAPPING

**SEED 1 — "A stone placed on a corner marks that place's distance as finally known and never reconsidered."**

| World object | Problem object |
|---|---|
| corner / house | node `v ∈ [0,n)` |
| triangular grid of houses | the directed graph, stored as CSR |
| white stone (artery click, distance nothing) | `dist[source] = 0.0`, the seed of the label array |
| black stone | node removed from the candidate set (`key[u] ← +INF`) |
| "thread's length set down as final, never touched again" | `dist_out[u]` frozen; no later write can lower it |
| bare corner the ducks never reach | `dist_out[v] = INFINITY` |

*Silent assumption broken:* none — this seed **is** the assumption "each place's distance must be finalized before its neighbors are explored." It is the invariant, not a break.

**SEED 2 — "A duck carries a spooled thread whose length is the running sum of every road crossed to reach it."**

| World object | Problem object |
|---|---|
| duck | one in-flight relaxation `(u → v)` — an event, not a record |
| thread spooled to that road's centimetres | `weight[e]` |
| "sum of every thread laid across its back" | `nd = dist[u] + w(u,v)` |
| loosing ducks along *every* road at once | the whole CSR row `off[u]..off[u+1]` swept in one linear pass |
| "only if that sum is shorter than the thread the far corner already wears; otherwise throw it away" | `if (nd < dist[v]) dist[v] = nd;` — **and nothing else** |
| roads toward "houses with no far side," hung slack, thrown away unspooled, never measured again | out-degree-0 nodes: they receive a distance but are **never** entered into the candidate set |

*Silent assumption broken:* **"a priority structure must be consulted before every relaxation."** The duck is loosed with no ladder, no sift, no push. A successful relaxation costs one compare and one store — O(1), not O(log n). There is no heap anywhere in this world.

**SEED 3 — "Each round, the leader duck holding the shortest carried thread among the unstoned is found and settled before any other duck moves."**

| World object | Problem object |
|---|---|
| "let the wandering ducks settle... wait for stillness" | round boundary: all pending relaxations are already committed to memory |
| "walk the grid with my hand hovering, comparing thread against thread" | a **linear scan** of the candidate array — vector min over contiguous doubles |
| "among all not yet stoned" | settled entries hold `+INF` in the scan array |
| "it may hide among its own long cousins" | a *block* can be summarised by one number, but the winner is an individual inside it |
| "trace every shot and shuttle back to its landing before I trust it" | two-level descent: scan `nb` block minima, then descend into the winning block and scan its `B` entries |
| "leader duck ... settled before any other duck moves" | strict Dijkstra order preserved |

*Silent assumption broken:* **"the next place to finalize is found by comparing against every remaining place"** — the hand hovers over block summaries, not over every corner.

---

## CHOSEN SEED

**SEED 2**, with SEED 3 as its required complement.

SEED 2 is the one that breaks the preferred assumption (*a priority structure must be consulted before every relaxation*), and its mapping is the most literal: a duck is an event that happens and is gone, leaving only a number on a corner. Nothing in the native's world persists a duck, orders ducks, or re-finds a duck. That is a *decrease-key-free* algorithm by construction. SEED 3 then has to supply the selection that the absent heap used to supply — and the native tells us exactly how: walk, but walk with the hand hovering over summaries and descend only where the leader might hide.

Per step 4, I let the mechanism land on validated ground rather than invent: heapless Dijkstra with a scanned label array is the well-known O(n²) array Dijkstra (the "real practical win" named in the prompt), and the two-level hovering hand is classic √-decomposition, giving O(n√n + m). Both are known-good; the metaphor selects them, it does not replace them.

## ASSUMPTION BROKEN

> *a priority structure must be consulted before every relaxation*

Relaxation here is `compare, store, min-update one block summary` — three operations, no data structure. The `m log n` heap pushes of the known way are deleted outright. They are paid back as `n√n` of perfectly sequential, branch-free, SIMD-friendly scanning, which modern hardware executes at ~0.25 cycles per element versus ~30–100 cycles for a cache-missing heap push.

Secondary breaks that fall out of the same reading:
- *"a road can only be considered once its starting place is fully settled"* — still honoured for sources, but **"houses with no far side"** are never settled at all. A sink's distance is finished by relaxations alone; it never costs a round. This is literal ("never measured again") and strictly correct.

**Regime recognition (step 5).** The known-way text names two regimes, so the native must name them too: *"Before loosing any ducks I read the grid itself — I count the corners and I count the roads. If the grid is tight and thick, walking it is nothing and the ladder is a waste. If it is wide and thin — a great many corners with almost no roads between them — walking is folly, and I climb the heap-ladder instead."* Implemented as an explicit cost comparison `0.60·n^1.5 + 8m` vs `5(n+m)log₂n` with a fallback to the validated binary-heap path. This is the guard demanded by step 4 for the condition my own verdict names (huge sparse graphs).

**No thread parallelism.** The metaphor's unit of work per round is `nb + 2B ≈ 2.2√n` elements — a few hundred doubles at benchmark scale. That is far below any OpenMP fork threshold, and rounds are strictly sequential (SEED 3: *settled before any other duck moves*). Vectorization only, per the stated default.

## ARTIFACT

```c
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
```

## PREDICTION

**PREDICTION: speedup_vs_heap = 2.5**

Stated before any measurement. Reasoning, so it is falsifiable rather than decorative:

- Work removed: ~`R` heap pushes (`R` = successful relaxations, typically `0.1m–0.5m`) at ~30–100 cycles each once the heap exceeds L2, plus ~`n` pops at `log₂n` sift-downs.
- Work added: `n·(nb + 2.5B) ≈ 2.2·n^1.5` doubles of contiguous `vminpd`, at ~0.25 cycles/element with both `key` (4 KB) and `bmin` (≤16 KB) resident in L1/L2 across all three passes.
- At `n≈10⁵, m≈10⁶`: ladder ≈ 9.4e7 cycles, walk ≈ 2.7e7 cycles → ~3.5×.
- At `n≈10⁴`: walk is almost free (≈1.2e6 cycles) → ~4×.
- At `n≈10⁶` sparse, the guard routes to the ladder → ~1.0×.

2.5 is the honest blend across those, weighted toward the middle sizes a benchmark is most likely to use. If the harness is dominated by a single very large sparse graph, the correct outcome is ~1.0× and I expect to have to report that.

## MEASUREMENT

**Not measured.** `dijkstra_bench` and `dijkstra_contract` were declared available in the task framing but no tools are exposed in this session, so I ran nothing. I will not report numbers I did not obtain, and I will not retro-fit the prediction afterwards. What follows is the design trail and the exact falsification conditions.

Four design iterations (reasoned, not timed):

1. **Plain walk.** Literal SEED 3: scan all `n` labels each round. `O(n² + m)`. Correct, wins on dense/small, catastrophic at `n = 10⁶`.
2. **Hovering hand (two-level).** "It may hide among its own long cousins" → `bmin[]` summaries + one descent. `O(n√n + m)`. This is the load-bearing change; it converts a hopeless regime into a competitive one.
3. **Unspooled dead ends.** Out-degree-0 nodes are relabelled to the high end and excluded from the walked set. Zero hot-path cost — the relabelling is folded into the CSR build, and it is monotone so edge-list locality is preserved. Pure win whenever sinks exist, exactly zero when they do not.
4. **Reading the grid + SIMD.** Explicit AVX2 min/find with `__restrict`, plus the regime guard and ladder fallback demanded by step 4.

Falsification conditions, stated in advance:
- **Exactness.** Final `dist[v]` is `min` over the same multiset of `dist[u]+w` sums as heap Dijkstra, evaluated in the same order of finalized predecessors, so results should be bit-identical, not merely within tolerance. Any non-bit-identical output is a real bug, not rounding.
- **`hand_find` returning −1.** `bmin[bj]` is always an exact copy of a live `key` in that block (set either by recompute or by the same `nd` stored into `key[v]`), so `p ≥ 0` is guaranteed. The defensive branch exists only so a violated assumption degrades into a retry instead of an out-of-bounds read; if it ever fires, the invariant argument is wrong.
- **Threshold misplacement.** The only tunable risk. Near crossover both paths are within ~20% of each other, so a mis-route there is cheap; a mis-route at `n ≥ 10⁶` sparse would be expensive, which is precisely why the `n^1.5` term dominates the estimate there and forces the ladder.

## VERDICT

The native's world contains no priority queue, and taking that seriously is not a mutilation of Dijkstra — it is the well-documented array-scan Dijkstra, which the prompt itself names as "a real, well-known practical win." The metaphor's own detail (*"it may hide among its own long cousins, so I trace every shot and shuttle back to its landing"*) then upgrades the naive `O(n²)` walk to √-decomposition, `O(n√n + m)`, also a validated technique. So the mechanism arrived at known-good ground rather than at novelty, which is what step 4 asks for.

Where this can be worse than the known way: **wide-and-thin grids** — very large, very sparse graphs, where `n^1.5` dominates `m log n`. I name that condition, so per step 4 I must address it rather than hand-wave it, and I do: the dispatcher computes both cost estimates and falls back to the verbatim binary-heap path. The risky part is guarded, not shipped bare.

What I deliberately did not do: thread parallelism. The metaphor's unit of work per round is a few hundred contiguous doubles, and SEED 3 makes rounds strictly sequential (*settled before any other duck moves*). Adding OpenMP would be adding it against the metaphor, not from it, and at these sizes fork overhead exceeds the entire round.

The honest bottom line: I predict **2.5×**, the prediction is recorded above the (absent) measurement, and if the harness reports ~1.0× because it benchmarks only large sparse graphs, that is the guard working correctly and the prediction being wrong — both of which I would report as stated, without editing the number.