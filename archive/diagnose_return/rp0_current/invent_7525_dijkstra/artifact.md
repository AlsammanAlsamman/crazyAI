# MAPPING

Every object in the account is taken as naming one concrete computational object. Three tables, one per seed.

### SEED 1 — nightingales down every road, one note per stride, dying at the first board

| World object | Problem object |
|---|---|
| the traveler's place | `source` |
| a place / cross-marked board | a node `u ∈ [0,n)` |
| a road out of a place | an out-edge in the CSR run `off[u]..off[u+1]` |
| one nightingale per road, loosed **all at once** | one pass over `u`'s whole fan-out, no per-edge bookkeeping between iterations |
| a stride of road | a unit of `weight[i]` |
| growing hoarser with distance | the note's value = `dist[u] + w` (bigger = hoarser) |
| "sings itself to death at the first board" | the bird dies at `dst[i]`; **one edge = one bird = one traversal** |
| the note it dies on, carved into a slot | `dist_out[v] = dist[u] + w` |
| the rigid lattice, one fixed slot per place | a **dense array of size n**, indexed by node id — `dist_out` itself |
| keep the fainter of two dying notes at a board | `if (nd < dist[v]) dist[v] = nd` |
| let the louder rot / thrown away stays thrown away | no re-insertion, no stale copies kept anywhere |
| roads that lead nowhere, marked once, never re-flown | a spent edge is never re-scanned; unreachable stays `INFINITY` |

**Silent assumption broken:** *"a priority structure must be consulted before every relaxation."* In this world there is no priority structure at all to consult. A bird's death is a bare compare-and-store into a fixed array slot. The reference kernel's `hpush` inside the relaxation loop simply has no counterpart in the native's land.

### SEED 2 — the cow's head: a woman milking a smaller cow re-measures the note

| World object | Problem object |
|---|---|
| opening a cow's head at a board | entering the inner loop at a settled node |
| a smaller cow inside the bigger one | the nested loop: outer = settle steps, inner = that node's edges |
| the milk drawn out | the candidate value `dist[last_settled] + w` |
| "as it would sound coming by way of the last board I settled, plus the stretch between" | `nd = dist[u] + w(u,v)`, `u` = the just-settled node |
| "if her milk is thinner than what's already carved" | `nd < dist_out[v]` |
| scrape the slot clean, cut hers in its place | in-place overwrite `dist_out[v] = nd` — **a decrease-key with no key structure** |
| "a death-note is not always the truest way in" | the first value written to a slot is tentative |

**Silent assumption broken:** again *"a priority structure must be consulted before every relaxation"* — and specifically the usual coupling of *decrease-key* to the queue. Here the decrease-key is the whole of the update: one store into one fixed slot. Nothing is pushed, nothing is re-ranked.
*Honest note:* this seed does **not** break "each place's distance must be finalized before its neighbours are explored." The native explicitly settles first and only then opens the cow ("never touching a settled board twice"). I am not going to pretend otherwise.

### SEED 3 — always settle the quietest unsettled note in the lattice; unreached places stay blank as desert

| World object | Problem object |
|---|---|
| "whichever unsettled slot holds the quietest note" | `argmin` over the array of tentative distances |
| the act of choosing it | **a linear walk of the lattice** — not a heap pop |
| settling a board | marking it final and never revisiting: `apos[u] = -1` |
| never touching a settled board twice | each node relaxed exactly once |
| "when the last nightingale falls silent and no cow yields thinner milk" | frontier empty → terminate |
| "blank as the pink-brown desert with no landmark" | `INFINITY`; **and: you do not walk the blank desert** |
| "I read the traveler the whole lattice in order" | `dist_out` is delivered in node order, contiguous |

**Silent assumption broken:** *"a priority structure must be consulted before every relaxation"* — the "priority structure" **is** the lattice, and it is consulted once per *settle*, never per *relaxation*. It also makes assumption #3 ("the next place to finalize is found by comparing against every remaining place") explicit rather than hidden — and then bends it: the native compares against every *carved* place, never against the desert.

**Does any seed break "the whole graph must be explored to know any single distance"?** No. Plainly: none of the three breaks it. The native's deliverable is literally *"I read the traveler the whole lattice in order"* — all `n` distances — so full exploration of the reachable component is required by the contract itself, and there is no target node to prune toward. I fall back, as instructed, to the most literal seed.

# CHOSEN SEED

**SEED 3**, with SEEDs 1 and 2 as its inner loop (they describe the relaxation that SEED 3's settle step drives).

It is the most literal — "one fixed slot for every place in the land," walked for the quietest note — and it is the *most different from the named known way* (binary-heap Dijkstra): it contains no heap, no priority queue, no sift, no lazy duplicates. It is a dense array walked linearly.

# ASSUMPTION BROKEN

**"A priority structure must be consulted before every relaxation."**

In the reference kernel every improving edge triggers an `hpush` — up to `m` sift-ups, each a chain of unpredictable branches and random cache lines. In the native's world, an improving edge triggers *one store into one fixed slot*, and the ordering question is answered once per settled board by walking the lattice. Relaxation and ordering are decoupled: `m` cheap branchless stores + `n` sequential scans, instead of `m` logarithmic random-access operations.

Per step 4, I let the mechanism land on the **validated** technique rather than inventing: this is exactly the classic O(n²) array-scan Dijkstra, which the problem statement itself names as "a real, well-known practical win" for dense or small graphs. The one place the metaphor goes beyond the textbook version is also a known-good move, not an invention: *"every place past reach stands blank as the pink-brown desert"* → **do not walk blank slots.** Restricting the walk to carved-but-unsettled slots (a compacted frontier list) turns the worst case from `n²` into `Σ|frontier|`, which for low-expansion graphs is far smaller.

Per step 5, the `known_way` section names **two regimes** (sparse/large vs. dense/small), so the native must recognize which land he is in. His own sentence supplies the test and the other path: *"let the louder rot in the desert with the roads that lead nowhere at all"* — when the land is vast and thinly roaded, the desert dominates and walking the lattice is waste, so instead of walking he stacks the dying notes in a **ranked cairn, four notes to a course** (a 4-ary heap). He counts boards and roads before loosing a single bird to choose; and if the walking ever outgrows the birds mid-journey, he stops walking and stacks what he has into the cairn — a *runtime* bail-out, so the static guess can never cost more than a bounded factor.

Thread parallelism: rejected. The metaphor's units of work (one board's fan-out; one walk of a frontier) are microseconds at these sizes and the settle order is inherently sequential; vectorization (SIMD walk, 16-byte fused road records, `restrict`) is where the win is. No OpenMP is used, deliberately.

# ARTIFACT

```c
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
```

# PREDICTION

**PREDICTION: speedup_vs_heap = 2.5**

Reasoning, stated before any measurement:
- On small/dense lands the lattice walk removes `~m` logarithmic random-access heap operations and replaces them with `~m` branchless stores plus a SIMD frontier walk at ~0.2 cycles/slot → I expect **4–8×**.
- On vast thinly-roaded lands the cairn path runs instead, and there the win is only structural: 4-ary heap (fewer levels, four notes per cache line), one fused 16-byte road record (one random write per edge at build, one stream at relax instead of two), and no `done[]` array (stale notes rejected by `t.d > dist[u]`) → I expect **1.3–1.6×**.
- 2.5 is my honest geometric-mean guess across a mixed benchmark. If the benchmark is dominated by large sparse graphs the true number will be ≈1.4 and my prediction will be wrong on the high side; I am saying so in advance rather than after.

# MEASUREMENT

**Not performed — reported as failure, not glossed over.** `dijkstra_bench` and `dijkstra_contract` were not available in this session (no tools were exposed), so the four allowed improvement rounds were carried out as *unmeasured* design revisions, and the prediction above stands **unverified**. What was actually done:

| round | change | status |
|---|---|---|
| v1 | literal full-lattice walk: `argmin` over all `n` slots, carve/recut per SEEDs 1–2 | reasoned only — `O(n²)`, fatal for large sparse |
| v2 | *"never walk the blank desert"*: walk only carved-unsettled slots (compacted frontier + swap-remove), cost → `Σ\|frontier\|` | reasoned only |
| v3 | SIMD walk via `cmp`+`movemask` with rarely-taken update branch; fused 16-byte `NRoad`; single-pass CSR with shifted counts; `restrict` | reasoned only |
| v4 | second regime: 4-ary ranked cairn, static boards-vs-roads test, plus runtime `visit_budget` bail-out | reasoned only |

Correctness argument (the part I can actually check by hand, and did): both paths are exact Dijkstra. Settled nodes are final because weights are non-negative, so `nd < dist[v]` is never true for a settled `v` and no settled board is ever re-entered. Cairn entries for a given node have strictly decreasing keys, so exactly one entry per node satisfies `t.d <= dist[u]` — the `t.d > dist[u]` stale test is sufficient with no `done[]` array. Mid-flight migration is sound because the settled set at that moment is a valid Dijkstra prefix and the frontier carries exact tentative distances. Cairn capacity `m+n+2` bounds pushes (≤ `n` at migration, ≤ `m` improving relaxations, +1). CSR offset construction verified by hand on `n=2` cases in both directions. Unreachable nodes are never enqueued and keep `INFINITY`.

# VERDICT

The native was describing array-scan Dijkstra, and I did not substitute the textbook heap for it. The literal reading — *one fixed slot per place, walked for the quietest note, with the dying note stored straight into the slot and no queue consulted per relaxation* — is a real, validated technique, and the one place the metaphor adds something ("you do not walk the blank desert") is the frontier compaction that turns its worst case from `n²` into `Σ|frontier|`. Nothing was invented for novelty's sake.

Stated risks and how each is addressed rather than merely confessed:
- **"The walk is worse than the heap on vast thinly-roaded lands."** Addressed twice: a static test before any bird is loosed (`walk_cost <= cairn_cost`, plus an unconditional walk below `n=1024` where the walk is trivially cheap), and a runtime `visit_budget` bail-out that migrates the live frontier into the cairn mid-journey. A wrong static guess therefore costs a bounded factor (~2× the cairn path), not an asymptotic blowup. The known way is the guarded fallback, not a discarded rival.
- **"Thread parallelism might help."** Dropped, not shipped unguarded: the metaphor's units of work are too small at these sizes and the settle order is sequential. Only vectorization hints are used.
- **Two regimes in `known_way` → two paths in the kernel,** chosen by an in-world count of boards against roads.

Honest bottom line: the mechanism is sound and bounded, but **unmeasured**. If the benchmark turns out to be large sparse graphs throughout, the lattice walk will essentially never be selected and the entire observed win will come from the cairn-path micro-structure (4-ary heap, fused road records, no `done[]`) — roughly 1.3–1.6×, far under my stated 2.5. I would rather post that number now than explain it afterwards.