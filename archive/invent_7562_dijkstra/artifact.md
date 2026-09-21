# MAPPING

**SEED 1 — "bead of blood, no coat / black wrist-coats"**

| World object | Problem object |
|---|---|
| traveler's wrist / the knot they stand at | source node |
| one bead of blood, wearing no coat | `dist_out[source] = 0` (known immediately, not the sentinel value) |
| every other knot draped in a heavy, unmarked black wrist-coat | `dist_out[i] = INFINITY` for all `i != source` |

Breaks: none of the five listed assumptions — pure initialization, identical in heap Dijkstra and array-scan Dijkstra alike.

**SEED 2 — "shortest undone gasp / the lever / re-coating"**

| World object | Problem object |
|---|---|
| knot | graph node |
| road / vein | directed weighted edge |
| chant team, each knot holding a breath | every node's tentative distance held simultaneously in `dist_out[]` |
| "go first to whoever holds the shortest gasp still undone" — found by a **massed comparison of the whole team**, not a hierarchy | pick arg-min of `dist_out[]` among unsettled nodes by a **direct linear scan** (no heap) |
| walk its veins outward | iterate `off[u]..off[u+1]` of the newly settled node |
| the lever: road's own length pressed against beats already counted | `nd = dist_out[u] + weight(e)` |
| coat stripped, fresher tongue-etched count hung | `if (nd < dist_out[v]) dist_out[v] = nd` |

Breaks: **"a priority structure must be consulted before every relaxation."** In this mapping, relaxation (the lever/re-coat step) never touches any heap — it's a flat array write. The only priority-related work is one linear scan per settled node, decoupled from the individual edge relaxations that follow it.

**SEED 3 — "veins past the last house / mirror-lake"**

| World object | Problem object |
|---|---|
| veins running past the last house, no wall, no king | edges/paths leading to nodes never reached from source |
| cut at the slack end, dropped in the mirror-lake as a gasp caught by cloud | such nodes are left uncoated: `dist_out[i] = INFINITY`, exactly the contract's required output for unreachable nodes |

Breaks: none of the five — this is boundary-condition bookkeeping (unreachability), not a structural change to the algorithm.

# CHOSEN SEED

Seed 2. It's the only one of the three that breaks the preferred assumption ("a priority structure must be consulted before every relaxation"), and it is the most literal, most different-from-the-known-way description of the *core loop* itself (the other two seeds are boundary conditions, not the loop).

# ASSUMPTION BROKEN

"A priority structure must be consulted before every relaxation." Seed 2's "chant team" holds every node's tentative distance directly in an array; relaxation is a flat compare-and-write, and the priority comparison ("shortest gasp still undone") happens once per settled node via a full scan, not once per edge.

Per instruction 4: this is exactly the **validated known technique** already named in the problem statement — plain O(n²) array-scan Dijkstra, "a real, well-known practical win" for dense/small graphs. So I let the metaphor arrive there rather than inventing something novel.

Per instruction 4's risk clause: the known_way itself says this only wins for dense/small graphs — for large sparse graphs O(n²) is a real liability. So the kernel below guards this with a runtime cost check and falls back to the (also-known, validated) binary-heap path — satisfying instruction 5's requirement to encode both regimes the known_way section describes (dense/small vs. sparse/large).

# ARTIFACT

```c
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct { double d; int u; } HeapItem;

static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p;
    }
}
static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}

typedef struct { double val; int idx; } MinPair;

#pragma omp declare reduction(minpair : MinPair : \
    omp_out = (omp_in.val < omp_out.val ? omp_in : omp_out)) \
    initializer(omp_priv = (MinPair){INFINITY, -1})

/* Regime A ("chant team", the taught seed): dist_out[] is every knot's
   currently held breath. The knot holding the shortest undone gasp is
   found by a direct massed comparison (vectorized linear scan) — no
   heap is ever consulted during relaxation itself. Best for small n or
   densely veined cities. */
static void dense_dijkstra(int n, const int * restrict off, const int * restrict edst,
                            const double * restrict ew, int source, double * restrict dist_out,
                            char * restrict done) {
    for (int i = 0; i < n; i++) { dist_out[i] = INFINITY; done[i] = 0; }
    dist_out[source] = 0.0;

    for (int iter = 0; iter < n; iter++) {
        MinPair mp; mp.val = INFINITY; mp.idx = -1;
        #pragma omp simd reduction(minpair:mp)
        for (int i = 0; i < n; i++) {
            double di = done[i] ? INFINITY : dist_out[i];
            if (di < mp.val) { mp.val = di; mp.idx = i; }
        }
        int u = mp.idx;
        if (u < 0) break;
        double best = mp.val;
        done[u] = 1;
        int lo = off[u], hi = off[u + 1];
        #pragma omp simd
        for (int e = lo; e < hi; e++) {
            int v = edst[e];
            double nd = best + ew[e];
            if (nd < dist_out[v]) dist_out[v] = nd;
        }
    }
}

/* Regime B: classic binary-heap Dijkstra — the validated known
   technique, kept as the fallback for the regime the massed-chant scan
   does not cover: sparse, large graphs ("veins that run out past the
   last house into valleys no king ever built a wall for"), where an
   O(n^2) scan would burn far more breath than a hierarchy does. */
static void heap_dijkstra(int n, const int * restrict off, const int * restrict edst,
                           const double * restrict ew, int source, double * restrict dist_out,
                           char * restrict done, HeapItem * restrict heap) {
    for (int i = 0; i < n; i++) { dist_out[i] = INFINITY; done[i] = 0; }
    dist_out[source] = 0.0;
    int hs = 0;
    hpush(heap, &hs, 0.0, source);
    while (hs > 0) {
        HeapItem top = hpop(heap, &hs);
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        int lo = off[u], hi = off[u + 1];
        for (int e = lo; e < hi; e++) {
            int v = edst[e];
            double nd = dist_out[u] + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;

    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }

    /* Regime detection, taken literally from the metaphor: the native
       only ever taught the massed "chant team" scan; it is cheap (and
       vectorizes to many comparisons/cycle) when the city is small or
       densely veined, but wasteful once the city sprawls out sparsely.
       Compare the two costs and pick the cheaper regime at runtime,
       with the validated heap as fallback -- this addresses the known
       risk of O(n^2) blowing up on large, sparse graphs instead of
       shipping the risky part unguarded. */
    double log2n = log2((double)(n > 1 ? n : 2));
    double dense_cost = (double)n * (double)n / 16.0;
    double heap_cost   = (double)(n + m) * log2n * 2.0;
    int use_dense = dense_cost <= heap_cost;

    char *done = malloc((size_t)n);
    if (use_dense) {
        dense_dijkstra(n, off, edst, ew, source, dist_out, done);
    } else {
        HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
        heap_dijkstra(n, off, edst, ew, source, dist_out, done, heap);
        free(heap);
    }

    free(deg); free(off); free(edst); free(ew); free(fill); free(done);
}
```

PREDICTION: speedup_vs_heap = 2.5

# MEASUREMENT

No `dijkstra_bench`/`dijkstra_contract` tool is available in this session (per the environment's explicit note), so no run was performed here. This is the single artifact submitted for the external pipeline to compile and measure; the prediction above was written before any measurement, as required. I'm not fabricating numbers I didn't obtain.

# VERDICT

The literal reading of "shortest undone gasp, found by the massed chant team, then lever-relax with no heap in between" maps exactly onto plain O(n²) array-scan Dijkstra — a real, already-validated technique for dense/small graphs, not a novel invention. That satisfies instruction 4 directly.

The known_way section names two regimes (heap for general/sparse, O(n²) scan for dense/small), so a single-strategy kernel would be wrong per instruction 5; the kernel above encodes runtime regime detection (`dense_cost` vs `heap_cost`, derived from `n`, `m`) and falls back to the classic validated heap path for the regime — large, sparse graphs — where the seed's own metaphor ("veins running into valleys no king ever walled") warns the massed-scan approach breaks down. That guard directly addresses the risk instruction 4 requires me to not ship unguarded.

Vectorization (SIMD hints via `#pragma omp simd` with a custom `minpair` reduction for the arg-min scan, `restrict` pointers, contiguous CSR layout for relaxation) was applied before any thread-level parallelism, per instruction 4's stated default. No OpenMP thread parallelism was added: the per-node relaxation work (`off[u+1]-off[u]` edges) is typically far too small at realistic benchmark sizes to amortize thread launch/sync overhead, so adding it would risk the exact "overhead if small" failure instruction 4 warns against — leaving it out is the guard here.

Only one kernel iteration is submitted since no in-session measurement was possible to drive further refinement; genuine improvement passes (e.g., tuning the `dense_cost`/`heap_cost` crossover constants, or reconsidering thread parallelism for very large dense graphs) should be done against real `dijkstra_bench` numbers once the pipeline runs this artifact.