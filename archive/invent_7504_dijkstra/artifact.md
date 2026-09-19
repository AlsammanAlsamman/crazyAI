# MAPPING (per SEED)

| World object | Problem object | Assumption it targets |
|---|---|---|
| **SEED 1** — clay slab / bowl / grey stone / white stone | node; the `dist_out[i]` slot; "tentative, unsettled" state; "finalized, proven" state (`settled[i]`) | Encodes assumption "*a place's distance must be finalized before its neighbors are explored*" — but it doesn't break it, it *implements* it literally (settled/unsettled split is kept as-is). |
| **SEED 2** — knotted cords, one bead per span of ground | directed edge `src[i]->dst[i]` with weight `weight[i]`; "cordless roads" = edges never usable because their far end is unreachable | No assumption broken — this is just the literal edge/weight data itself (adjacency), a faithful re-statement of the input, not an algorithmic choice. |
| **SEED 3** — shadow-stick crossing scratches; scanning *every* grey slab each tick; burning a cord that names a worse count | the outer loop of `n` rounds; the **selection of the next node to settle by a plain linear scan of every unsettled node's current `dist_out` value**, with no auxiliary structure; the relax check `if (nd < dist_out[v])` | Directly breaks "*a priority structure must be consulted before every relaxation*" (and its corollary, "*the next place to finalize is found by comparing against every remaining place*" — here that comparison-against-everyone is not an assumption to avoid, it is literally *how* the native does it, replacing the heap). |

# CHOSEN SEED
**SEED 3** — it is the most literal (the shadow-stick tick = outer loop; "look over every clay slab still holding a grey stone" = an explicit, unambiguous instruction to linearly scan *all* remaining nodes) and the most different from the reference `kernel` given in the contract, which pops a binary heap instead.

# ASSUMPTION BROKEN
"*A priority structure must be consulted before every relaxation.*" The native keeps no heap, no bucket queue, nothing sorted — settling the next slab is a brute force scan over every still-grey bowl, each shadow-tick. This is exactly the textbook-known "plain O(n²) array-scan Dijkstra" alternative to the heap.

# ARTIFACT

```c
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    /* --- lay the cords: build CSR adjacency, one bucket of cords per slab --- */
    int *deg = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fill = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(fill);
    free(deg);

    /* --- every slab gets a bowl; grey stone (unsettled) is the default --- */
    char *settled = (char*)calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;

    /* the traveler's own bowl gets its white stone at zero cost, immediately */
    dist_out[source] = 0.0;

    /* --- the shadow-stick: one tick per slab to settle --- */
    for (int round = 0; round < n; round++) {

        /* look over every slab still holding a grey stone; no priority
           structure consulted -- a plain scan for the smallest chalk-candidate */
        int best_u = -1;
        double best_d = INFINITY;

#ifdef _OPENMP
        #pragma omp parallel if(n > 20000)
        {
            int local_u = -1;
            double local_d = INFINITY;
            #pragma omp for nowait schedule(static)
            for (int v = 0; v < n; v++) {
                if (!settled[v] && dist_out[v] < local_d) {
                    local_d = dist_out[v];
                    local_u = v;
                }
            }
            #pragma omp critical
            {
                if (local_d < best_d) { best_d = local_d; best_u = local_u; }
            }
        }
#else
        for (int v = 0; v < n; v++) {
            if (!settled[v] && dist_out[v] < best_d) {
                best_d = dist_out[v];
                best_u = v;
            }
        }
#endif

        /* no cord ever reached the remaining slabs: their bowls stay empty */
        if (best_u == -1) break;

        /* set the white stone: this place's distance is now proven true */
        settled[best_u] = 1;

        /* follow every cord out of the newly white-stoned slab; a cord that
           would name a longer count than the chalk already there is burned */
        int rs = off[best_u], re = off[best_u + 1];
        for (int e = rs; e < re; e++) {
            int v = edst[e];
            if (!settled[v]) {
                double nd = best_d + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
    }

    free(off);
    free(edst);
    free(ew);
    free(settled);
}
```

Improvement steps folded into this single final version (conceptually V1→V4, since no live measurement loop was available this session):
1. **V1 (literal):** plain serial O(n²) scan + relax, exactly as narrated.
2. **V2:** CSR adjacency build instead of repeatedly re-reading the raw edge list, so each round's relax touches contiguous memory only.
3. **V3:** OpenMP-parallel reduction over the scan step (the true O(n²) bottleneck) for large `n`, gated behind a size threshold so small graphs don't pay thread-spawn overhead.
4. **V4:** kept the relax loop serial (total work there is only O(m) across the whole run, and it's small/branchy per node — parallelizing it risked write races on duplicate-edge targets for no real gain).

# PREDICTION
Native's O(n²)-no-heap design should win only when `n` is small/moderate or the graph is dense (better cache behaviour, zero heap bookkeeping, perfectly branch-predictable inner loops) — and should lose badly, potentially catastrophically, on large sparse graphs where O(n²) >> O((n+m) log n).

PREDICTION: speedup_vs_heap = 1.4

(reasoning: expecting a moderate/dense-ish default benchmark graph, ~1.3–1.5x from removing heap-push/pop overhead and improving locality; flagging explicitly that this could invert to <<1 if `dijkstra_bench` uses a large sparse graph — that is a real, expected failure mode of this literal translation, not something to paper over.)

# MEASUREMENT
No tool access was available in this session (`dijkstra_bench` was not invoked here per the environment note). The prediction above was recorded before any measurement, as required; the actual number must come from the pipeline running `dijkstra_bench` against this artifact. I am not fabricating a measured value.

# VERDICT
Unverified in this session — honestly reported as such. Expected outcome, stated plainly: a real win (~1.2–1.6x) on small/dense/moderate graphs consistent with the well-known "O(n²) beats the heap when the graph is dense or small" fact the problem itself calls out; a real loss (large slowdown, speedup well below 1) if the benchmark's graph is large and sparse, since this literal translation of the native's story makes no attempt to avoid the O(n²) scan — that would require abandoning the seed's literal instruction ("look over every clay slab") in favor of a heap, which is precisely the assumption we chose to break.