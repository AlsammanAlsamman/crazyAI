# MAPPING

**SEED 1** — "A runner walks each road carrying its notch and lays a new stick on the far heap equal to the home ember's length plus that notch."

| World object | Problem object |
|---|---|
| road | directed edge (u→v, weight w) |
| notch on the stick | edge weight w |
| home heap's lit ember | current tentative distance dist[u] |
| far heap | destination node's candidate-distance collection |
| new stick laid on far heap | candidate value dist[u] + w offered to v |

Breaks: **"a road can only be considered once its starting place is fully settled"** — relaxation fires off *any* node that currently holds an ember (a tentative distance), settled or not. This alone is just generic label-correction (Bellman-Ford's basic move); it does not touch the global-comparison assumption.

**SEED 2** — "Every heap throws its sticks together in one land-wide throw, and the row read straight across picks each heap's shortest sliver as its new ember while the rest are burned for fuel."

| World object | Problem object |
|---|---|
| all heaps thrown together, one throw for the whole land | one synchronous round applied to *every* node simultaneously (BSP superstep) |
| a heap's row, read straight across | one destination node v's full list of incoming candidate edges (reverse-adjacency row) |
| shortest sliver in the row | min over that row → new dist[v] for this round |
| burning the longer slivers | discarding all non-minimal candidates, keeping only the winner |

Breaks: **"the next place to finalize is found by comparing against every remaining place"** — there is no scan over remaining nodes to pick one winner; every node picks its own local minimum in the same synchronized pass.

**SEED 3** — "A garrison seals a heap once its ember stops changing between throws, fixing its distance for good and halting its runners."

| World object | Problem object |
|---|---|
| garrison riding in | marking a node "no longer a relaxation source" |
| ember unchanged between throws | dist[v] identical this round vs. last round |
| sealed heap refuses new sticks | edges into an already-sealed node are dropped |

Breaks the same assumption as Seed 2 (finalization by local stability, not global comparison) — but taken **literally** (permanently forbidding any future update the instant a value stalls for one round) it is **unsafe**: a node's value can plateau for a round and then improve in a later round once a longer, cheaper multi-hop path catches up (verified by hand with a 4-node counterexample: direct edge 10, alternate 3-hop path of total weight 3 that only resolves two rounds later — naive sealing would freeze the node at 10). So Seed 3's literal mechanism is dropped; its *safe* residue — "stop sending runners from a node once it stops changing" (not "stop receiving") — survives as the well-known, validated **SPFA / active-node Bellman-Ford** pruning rule.

# CHOSEN SEED
**Seed 2** — the synchronous "one throw for the whole land, row read straight across" is the literal, load-bearing mechanism (a BSP round over reverse-adjacency rows), and it is the one most unlike heap-Dijkstra: no heap object exists anywhere.

# ASSUMPTION BROKEN
"The next place to finalize is found by comparing against every remaining place." Here no such global scan ever happens — finalization is a *local*, per-row, per-round convergence fact, computed for all n rows in parallel.

# ARTIFACT

Places = nodes, roads = directed edges, a heap = one destination node's row in the **reverse** adjacency list, a stick = a candidate distance offered along one incoming edge, an ember = dist[v], a throw = one synchronous round over all rows, sealing = "this node stopped changing, stop treating it as a source" (safe pruning only, never blocking future *incoming* updates — that was the part of Seed 3 that broke correctness). Because worst-case hop-count can approach n on long sparse chains (the risk Bellman-Ford-style methods are known for), a density/size guard falls back to the reference heap-Dijkstra, which is the validated technique for that regime.

```c
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- heap-based Dijkstra: validated fallback for sparse/large graphs ---- */
typedef struct { double d; int u; } HeapItem;
static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) / 2; if (h[p].d <= h[i].d) break; HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
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
static void heap_dijkstra(int n, int m, const int *off, const int *edst, const double *ew,
                           int source, double *dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    char *done = calloc((size_t)n, 1);
    HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0;
    hpush(heap, &hs, 0.0, source);
    while (hs > 0) {
        HeapItem top = hpop(heap, &hs);
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = dist_out[u] + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }
    free(done); free(heap);
}

/* ---- round-synchronous "throw" (active-node Bellman-Ford, Seed 2) ---- */
void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) { int u = src[i]; int pos = off[u] + fill[u]++; edst[pos] = dst[i]; ew[pos] = weight[i]; }

    /* guard: round-based cost is O(rounds*m); rounds stays small only
       when the graph is dense or small (short hop-diameter). For
       sparse/large graphs, fall back to the validated heap method. */
    double avg_deg = (n > 0) ? (double)m / (double)n : 0.0;
    int use_rounds = (n <= 4000) || (avg_deg >= 8.0);

    if (!use_rounds) {
        heap_dijkstra(n, m, off, edst, ew, source, dist_out);
        free(deg); free(off); free(edst); free(ew); free(fill);
        return;
    }

    /* reverse CSR: row v = every incoming road (u, w) into heap v */
    int *rdeg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) rdeg[dst[i]]++;
    int *roff = malloc((size_t)(n + 1) * sizeof(int));
    roff[0] = 0;
    for (int i = 0; i < n; i++) roff[i + 1] = roff[i] + rdeg[i];
    int *rsrc = malloc((size_t)m * sizeof(int));
    double *rw = malloc((size_t)m * sizeof(double));
    int *rfill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) { int v = dst[i]; int pos = roff[v] + rfill[v]++; rsrc[pos] = src[i]; rw[pos] = weight[i]; }

    double *dist = malloc((size_t)n * sizeof(double));
    double *ndist = malloc((size_t)n * sizeof(double));
    char *changed = malloc((size_t)n);
    char *nchanged = malloc((size_t)n);
    for (int i = 0; i < n; i++) { dist[i] = INFINITY; changed[i] = 0; }
    dist[source] = 0.0;
    changed[source] = 1;

    int any_active = 1;
    int rounds_cap = n > 0 ? n - 1 : 0;
    for (int round = 0; round < rounds_cap && any_active; round++) {
        any_active = 0;
        #pragma omp parallel for schedule(static) reduction(||:any_active) if(n > 20000)
        for (int v = 0; v < n; v++) {
            double best = dist[v];
            int rs = roff[v], re = roff[v + 1];
            for (int e = rs; e < re; e++) {
                int u = rsrc[e];
                if (!changed[u]) continue;            /* runners stopped from a sealed source */
                double du = dist[u];
                if (du == INFINITY) continue;
                double cand = du + rw[e];
                if (cand < best) best = cand;
            }
            nchanged[v] = (char)(best < dist[v]);
            ndist[v] = best;
            if (nchanged[v]) any_active = 1;
        }
        double *td = dist; dist = ndist; ndist = td;
        char *tc = changed; changed = nchanged; nchanged = tc;
    }

    memcpy(dist_out, dist, (size_t)n * sizeof(double));

    free(deg); free(off); free(edst); free(ew); free(fill);
    free(rdeg); free(roff); free(rsrc); free(rw); free(rfill);
    free(dist); free(ndist); free(changed); free(nchanged);
}
```

Iteration notes (reasoned, not measured — see MEASUREMENT): v1 was plain full-graph synchronous Bellman-Ford (every row, every round, no pruning) → v2 added the `changed[u]` guard (SPFA-style active-node pruning, safe because an unchanged source can't produce a better candidate) → v3 switched from forward-edge scatter (which needs atomics/races across threads) to reverse-CSR gather-per-row (each v writes only its own cell — races impossible, and OpenMP/`if(n>20000)` becomes free) → v4 added the density/size guard with fallback to heap Dijkstra, directly addressing the method's own known weakness (long sparse chains → up to n rounds → O(n·m)).

PREDICTION: speedup_vs_heap = 1.4

# MEASUREMENT
Not run: no tool access was available in this session (`dijkstra_bench` could not be invoked here). Per the harness note, the pipeline compiles and measures this artifact externally — I am not fabricating a number. Reasoned expectation by regime: on the sparse/large branch the code takes the identical heap-Dijkstra path as the reference (parity, ~1.0×, plus a small extra reverse-CSR-skip so effectively no penalty); on the dense/small branch (hop-diameter typically small) the row-gather rounds should win by avoiding heap pointer-chasing entirely, favoring contiguous SIMD-friendly scans, so noticeably faster there. The blended 1.4 reflects "no worse than baseline on most graphs, clearly better on the dense/small subset."

# VERDICT
The literal native mechanism (Seed 2's synchronous throw) is real and does break the targeted assumption, but its companion piece (Seed 3, sealing on one round of no-change) is provably unsafe in general — I found and hand-verified a counterexample where premature sealing gives a wrong answer. I did not ship that broken part; instead the mechanism was steered to the well-known, validated technique that already satisfies the same "no global comparison" property safely: active-node (SPFA-style) synchronous Bellman-Ford, with reverse-CSR rows for race-free parallel gather. Its own stated risk — O(n·m) blowup on long sparse chains — is addressed, not hand-waved: a density/size guard routes that regime straight to the reference heap-Dijkstra, so the mechanism can only help (dense/small graphs) and never meaningfully hurt (sparse/large graphs fall back exactly to the known-good path). Thread parallelism is gated by `if(n>20000)` per the vectorization-before-threading preference; the everyday win is meant to come from the contiguous, branch-light per-row array scan, not from OpenMP.