# MAPPING (all three seeds)

| World object | Problem object | Assumption (from the list) it breaks |
|---|---|---|
| **SEED 1** — snake, cane, far square | snake = the relaxation pass over one node's out-edges; cane = a directed edge (u,v,w); eggshell-count carried = candidate distance `dist[u]+w`; far square = edge destination `v` | "a road can only be considered once its starting place is fully settled" — the snake is loosed down a square's canes at the moment the pawn sits there, not after any queue re-check; relaxation is a direct sweep of that node's adjacency, nothing else consulted |
| **SEED 2** — thumb, boulder, unboulered square | thumb-scan = the "find minimum" step; boulder = permanent finalize/visited mark; unboulered square = unsettled node; pouch weight = `dist[]` | "a priority structure must be consulted before every relaxation" / "the next place to finalize is found by comparing against every remaining place" — there is no heap, cane-list, or auxiliary structure at all; the thumb just compares every remaining pouch directly, each round, by hand |
| **SEED 3** — dead-end canes, empty pouches, paper | unreached squares = graph nodes with no path from source; "leave off the paper" = those entries stay `INFINITY` and are conceptually not part of the finalized output set | "the whole graph must be explored to know any single distance" — the snake only ever travels canes that exist; nothing is fabricated or forced for the unreachable component |

# CHOSEN SEED
**SEED 2** — the boulder/thumb-scan. It is the most literal (a direct hand-count comparison, no auxiliary bucket/heap object exists anywhere in the story) and the most different from the given heap-based reference kernel, while still being a legitimate, well-known technique (flagged in the prompt itself as the O(n²) array-scan Dijkstra).

# ASSUMPTION BROKEN
"A priority structure must be consulted before every relaxation" — replaced by a flat linear scan of all pouches each round; no heap, no `hpush`/`hpop`, no log-factor bookkeeping at all.

# ARTIFACT

```c
#include <stdlib.h>
#include <math.h>

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    /* Group every raspberry-cane road by the square it leaves from (CSR). */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }

    /* dist_out: the true, permanent pouch count once a boulder is pressed.
       scan_dist: the thumb's own view of the pouches -- identical values,
       except a boulder-ed square is pushed to infinity so the thumb's
       hand-comparison never lands on it again. */
    double *scan_dist = malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) { dist_out[i] = INFINITY; scan_dist[i] = INFINITY; }
    dist_out[source] = 0.0;   /* the traveler's own square: empty pouch, costs nothing to leave */
    scan_dist[source] = 0.0;

    int pawn = source;

    for (int round = 0; round < n; round++) {
        int u = pawn;
        double du = dist_out[u];

        /* Loose the snake down every cane leading from the pawn's square. */
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double cand = du + ew[e];        /* cane's eggshells + pawn's pouch */
            if (cand < dist_out[v]) {        /* only a lighter delivery counts */
                dist_out[v] = cand;
                scan_dist[v] = cand;
            }
        }

        /* Press the boulder on the pawn's own square: fixed forever. */
        scan_dist[u] = INFINITY;

        /* Thumb sweeps every unboulered pouch for the lightest one. */
        double best = INFINITY;
        int best_i = -1;
        for (int i = 0; i < n; i++) {
            double di = scan_dist[i];
            if (di < best) { best = di; best_i = i; }
        }
        if (best_i < 0) break;   /* no reachable pouch left -- rest stay empty, off the paper */
        pawn = best_i;
    }

    free(deg); free(off); free(edst); free(ew); free(fill); free(scan_dist);
}
```

# PREDICTION
PREDICTION: speedup_vs_heap = 1.3

Reasoning: no heap allocation/percolation overhead and a branch-light, cache-linear O(n) scan should beat `log n` heap operations on small/medium and dense graphs (exactly the case the prompt itself calls out as "a real, well-known practical win"). But the O(n²) node-selection term will lose badly once n grows large and the graph stays sparse (m ≈ O(n)), since heap-Dijkstra's O((n+m) log n) pulls ahead there. I expect a modest net win if the benchmark suite is skewed toward small/dense/medium graphs, tempered by likely losses on the largest sparse cases — hence a conservative >1 but not dramatic estimate.

# MEASUREMENT
Not available in this session — no tools (`dijkstra_bench`, `dijkstra_contract`, etc.) were provided here; per the environment note, this response contains the prediction and the complete artifact only, and the external pipeline performs the actual compile + benchmark run.

# VERDICT
Pending the pipeline's measurement. Expected outcome, stated plainly in advance: a real but graph-size-dependent result — likely a moderate speedup (roughly in the 1.1x–1.8x range) on small/dense inputs, with a realistic chance of falling *below* 1.0x (a genuine loss) on large sparse inputs where the O(n²) thumb-scan's quadratic term dominates over the heap's O((n+m) log n). If that split is what's observed, it should be reported as such rather than smoothed into a single "win" — the seed's literal method is a legitimate, known trade-off, not a universal improvement.