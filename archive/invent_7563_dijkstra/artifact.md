# MAPPING (all three seeds)

| World object | Problem object | Assumption it touches |
|---|---|---|
| **Seed 1 — eggshell cracks once, stays fixed** | An eggshell = a node's `dist_out` slot + a "settled" flag. Cracking = marking `done[u]=1` and freezing its value. "Fresh swarms released only from a newly cracked organ" = edges out of `u` are only relaxed after `u` is finalized. | This **embodies** assumption 4 ("a road can only be considered once its starting place is fully settled") and the immutability half of assumption 1. It does not *break* anything — it's a restatement of standard finalization. |
| **Seed 2 — melt compensated per rib length, cumulative** | melt spent = accumulated path weight; "compensating melt in proportion to rib length" = `nd = dist[u] + weight(u,v)`; carrying forward = passing `dist[u]` into the relaxation of `v`. | Pure restatement of the relaxation formula. Breaks none of the five assumptions — it's the arithmetic every Dijkstra variant shares. |
| **Seed 3 — wait for least melt among all crawling swarms, discard swarms hitting a cracked shell** | A "swarm" = a pending (distance, node) relaxation event; "all those still moving" = the active event set; "wait for least melt" = extract-min; "throw away swarms reaching a cracked organ" = lazy deletion of stale heap entries. | The comparison pool is "all still-moving swarms" (the dispatched event multiset), **not** "every remaining place" (every unfinalized node). So it does **break** assumption 3 in the same way a binary heap already does — but it converges exactly onto the already-known, already-validated heap-Dijkstra technique (which is in fact the reference kernel given to us). |

# CHOSEN SEED

**Seed 3.** It's the only one of the three that touches assumption 3 at all, and per step 2 that gives it priority even though the mechanism it describes is, taken completely literally, ordinary lazy-deletion heap Dijkstra.

# ASSUMPTION BROKEN

"The next place to finalize is found by comparing against every remaining place." Seed 3's swarms only compete against the swarms *currently in flight*, never against nodes that haven't been reached yet — exactly what a priority queue buys you over an O(n) full-array scan.

Per step 4: since the mechanism Seed 3 literally describes is already the well-known, validated binary-heap Dijkstra (the reference kernel itself), I do not invent a new selection mechanism to re-break the same assumption a second, novel way. Instead I keep the heap (upgraded to a d-ary heap — fewer levels, better cache reuse, a real, well-documented practical tweak, not exotic) as the "swarm race" mechanism.

Per step 4's other clause ("if your own VERDICT names a specific condition where your mechanism could be worse... guard it"): a pure heap is known to lose to plain O(n²) array-scan Dijkstra on dense or small graphs (stated explicitly in `known_way`). Per step 5, the metaphor itself must therefore also encode which regime it is in. I read this out of Seed 1's own text: "every place not yet cracked stays sealed... that blankness is itself the answer" together with "the ordinary foreign business of the roads settling into what is known" — i.e. when the roads out of every court are so numerous that the courts are basically all interconnected (dense), there's no need to keep an archive of swarms at all — you can just look at every place directly. That licenses a runtime regime check: dense/small → plain array scan (no heap object exists); sparse → d-ary swarm-race heap.

# ARTIFACT

```c
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define DARY 4  /* d-ary heap: fewer levels than binary, better cache reuse per pop */

typedef struct { double d; int u; } HeapItem;

static inline void dheap_push(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) / DARY;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem dheap_pop(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    int last = --(*hs);
    h[0] = h[last];
    int i = 0;
    for (;;) {
        int base = DARY * i + 1;
        int s = i;
        for (int k = 0; k < DARY; k++) {
            int c = base + k;
            if (c < *hs && h[c].d < h[s].d) s = c;
        }
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    /* --- build CSR: every place's own bundle of outgoing ribs --- */
    int *off  = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    {
        int *deg = (int*)calloc((size_t)n, sizeof(int));
        for (int i = 0; i < m; i++) deg[src[i]]++;
        off[0] = 0;
        for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
        int *cursor = (int*)malloc((size_t)n * sizeof(int));
        memcpy(cursor, off, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = cursor[u]++;
            edst[pos] = dst[i];
            ew[pos]   = weight[i];
        }
        free(deg); free(cursor);
    }

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n > 0) dist_out[source] = 0.0;

    /* Regime check (Seed 1's "ordinary foreign business of the roads
       settling into what is known"): if courts are so richly connected
       (dense) or so few (small) that maintaining a swarm archive costs
       more than just looking at every place directly, don't build one. */
    double avg_out = (n > 0) ? (double)m / (double)n : 0.0;
    int use_dense = (n <= 256) || (avg_out > (double)n / 32.0);

    char *done = (char*)calloc((size_t)n, 1);

    if (use_dense) {
        /* ---- dense / small regime: no priority structure at all ---- */
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] < best) { best = dist_out[v]; u = v; }
            }
            if (u < 0) break; /* remaining places sealed forever: unreachable */
            done[u] = 1;
            double du = dist_out[u];
            int e0 = off[u], e1 = off[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
    } else {
        /* ---- sparse regime: swarm race via a d-ary priority heap ----
           Each relaxation dispatches a fresh swarm (heap entry) carrying
           cumulative melt. We crack whichever shell the least-melt swarm
           among all still-crawling swarms reaches; a swarm that later
           reaches an already-cracked shell is thrown away (lazy deletion),
           and a cracked shell's value is never overwritten. */
        HeapItem *heap = (HeapItem*)malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        dheap_push(heap, &hs, 0.0, source);
        while (hs > 0) {
            HeapItem top = dheap_pop(heap, &hs);
            int u = top.u;
            if (done[u]) continue;   /* spent for no purpose */
            done[u] = 1;             /* fixed shell stays fixed */
            double du = dist_out[u];
            int e0 = off[u], e1 = off[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    dheap_push(heap, &hs, nd, v);
                }
            }
        }
        free(heap);
    }

    free(done);
    free(off); free(edst); free(ew);
}
```

PREDICTION: speedup_vs_heap = 1.3

# MEASUREMENT

Not run in this session — no `dijkstra_bench`/`dijkstra_contract` tool is reachable here (session note: "No tools are available in this session"). I am not fabricating a benchmark result. The pipeline that compiles and measures this kernel against the reference binary-heap baseline should fill this section with the actual numbers; the prediction above stands as the honest pre-registered guess: modest (~1.1–1.3x) gains on typical sparse-graph benchmark cases from the d-ary heap and leaner CSR build, with potentially much larger gains (multiple x) on any dense/small test cases that trip the array-scan fallback, and possible near-parity or slight loss on already near-complete/tiny cases if the threshold constants (`n<=256`, `avg_out > n/32`) are mistuned for the actual benchmark's graph sizes.

# VERDICT

Honest self-assessment of risk, per step 4: the one condition under which this kernel could be *worse* than the plain reference is if the dense/small threshold is wrong for the benchmark's actual graphs — e.g. a "sparse-looking" graph that's actually small enough that heap overhead dominates but doesn't trip `n<=256`, or a graph just past the density threshold where the O(n²) scan's constant-factor win hasn't kicked in yet. That risk is already addressed structurally, not left dangling: the kernel *always* has both paths compiled in and picks between them with a cheap, explicit runtime check rather than betting on one strategy — exactly the "recognize the regime, offer a fallback" requirement, and exactly what step 4 asks for (arrive at the two known, validated techniques — heap and O(n²) scan — rather than inventing a third, riskier one). No thread-level parallelism was added: at the node/edge-relaxation granularity here, per-iteration work is too fine-grained for OpenMP to pay off over heap/array bookkeeping overhead, so I stayed with vectorization-friendly straight-line loops (`restrict`-free but branch-light) as instructed, and did not add threads without a size check that would justify them.