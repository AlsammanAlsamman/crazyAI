# MAPPING

| SEED | World object | Problem object | Assumption it breaks |
|---|---|---|---|
| 1 | Grid of triangular houses; white/black stone on a corner; "distance from itself is nothing" | Graph nodes; `dist_out[]` entry finalized (source dist=0); a `done`/finalized flag per node, set once and never touched again | None of the five — it *confirms* "finalize before exploring neighbors" (A1), it doesn't break it. |
| 2 | Duck loosed along a road, carrying "a thread spooled to exactly that road's centimetres... the sum of every thread laid across its back"; "otherwise I throw the new thread away" | Edge relaxation: `nd = dist[u] + w(u,v)`; if `nd < dist[v]` overwrite, else discard. The duck *is* the tentative-distance array cell, not a heap node — nothing is pushed into or popped from any priority structure at relax time. | Breaks **"a priority structure must be consulted before every relaxation"** — relaxation here is a bare array write/compare, with no heap push/decrease-key at all. |
| 3 | "Walk the grid with my hand hovering, comparing thread against thread... find the leader duck... among all not yet stoned" | Finding `argmin` of the tentative-distance array over all unfinalized nodes by linear scan, once per round | Breaks **"the next place to finalize is found by [something other than] comparing against every remaining place"** — i.e. it *is* the brute-force comparison, confirming rather than avoiding A3. |

# CHOSEN SEED

**SEED 2** — the duck-carries-a-running-sum-thread seed. It is the one that most literally and most differently (from the heap) describes relaxation: a plain array write with no consultation of any priority structure. Combined with SEED 3 (linear-scan leader-finding) and SEED 1 (permanent stone = finalize flag), the *complete* metaphor turns out to be the classical **O(n²) array-scan Dijkstra, no heap at all** — which the problem statement itself names as "a real, well-known practical win" for dense/small graphs. Per instruction, I let the mechanism land on that validated known technique rather than inventing something novel.

# ASSUMPTION BROKEN

"A priority structure must be consulted before every relaxation." In the world, relaxation = spooling a thread onto a duck's back (an O(1) array compare-and-write); the priority structure (comparing threads) is consulted only **once per round**, not once per relaxed edge.

Self-named risk (per the problem's own text): this is only a win "for dense or small graphs" — for large sparse graphs O(n²) loses badly to O((n+m) log n). I guard this explicitly with a size/density check and fall back to the reference binary-heap algorithm (untouched, well-tested) when the graph is large and sparse.

# ARTIFACT

```c
#include <stdlib.h>
#include <math.h>

/* ---- binary-heap fallback: identical mechanism to the known/reference
   O((n+m) log n) solution, used when the graph is large AND sparse,
   the exact regime where the stone/scan mechanism is known to lose. ---- */
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

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;

    /* ---- shared CSR build ("roads out of each house") ---- */
    int *deg  = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off  = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int    *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew   = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i]; int pos = off[u] + fill[u]++;
        edst[pos] = dst[i]; ew[pos] = weight[i];
    }

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    /* Guard against the mechanism's own named weakness: O(n^2) only when
       small or dense; otherwise fall back to the validated heap path. */
    long long nn = (long long)n * (long long)n;
    int use_array = (n <= 3000) || ((long long)m * 16 >= nn);

    if (use_array) {
        /* ---- stones-and-ducks world, literally ---- */
        double *scan = malloc((size_t)n * sizeof(double)); /* "thread lengths of ducks not yet stoned" */
        for (int i = 0; i < n; i++) scan[i] = INFINITY;
        scan[source] = 0.0;

        const int    *restrict offp  = off;
        const int    *restrict edstp = edst;
        const double *restrict ewp   = ew;
        double *restrict distp = dist_out;
        double *restrict scanp = scan;

        for (int round = 0; round < n; round++) {
            /* "let the wandering ducks settle" -> value-only reduction, vectorizes cleanly */
            double best = INFINITY;
            for (int i = 0; i < n; i++) { double s = scanp[i]; if (s < best) best = s; }
            if (best == INFINITY) break; /* rest of the grid the ducks never reach */

            /* "trace every shot back to its landing" -> find which corner holds it */
            int u = -1;
            for (int i = 0; i < n; i++) { if (scanp[i] == best) { u = i; break; } }

            scanp[u] = INFINITY; /* black stone: never reconsidered */
            double du = distp[u];

            for (int e = offp[u]; e < offp[u + 1]; e++) {
                int v = edstp[e];
                double nd = du + ewp[e];
                if (nd < distp[v]) { distp[v] = nd; scanp[v] = nd; }
            }
        }
        free(scan);
    } else {
        /* ---- large & sparse: known heap-based path, unmodified ---- */
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

    free(deg); free(off); free(edst); free(ew); free(fill);
}
```

# PREDICTION: speedup_vs_heap = 1.15

Reasoning behind the number: the guard makes this identical in algorithm and cost to the reference heap solution whenever the benchmark uses large sparse graphs (the common case for shortest-path benchmarks), so on those instances I expect speedup ≈ 1.0 (small overhead from the extra `use_array` branch and duplicate CSR-build code, negligible). On small or dense instances the array/no-heap path removes all heap push/pop overhead (no `O(log n)` per relaxation, no heap array traffic, tight vectorizable scan loops), where I'd expect 2–4x on those specific cases. Blended across a typical mixed benchmark suite, I predict a modest net win, not a dramatic one — hence 1.15, not something aspirational like 3x.

# MEASUREMENT

No `dijkstra_bench` / `dijkstra_contract` tool was actually reachable in this session (the tool list available to me here is Gmail/Slack/Docs/Drive only — the benchmarking tools named in the task were not present as callable functions). I am not going to fabricate a benchmark run. Per the task's own framing, the prediction above is submitted for the external pipeline to compile and measure against the reference; I have not personally executed or timed it, and I'm reporting that plainly rather than presenting a guess as a measurement.

# VERDICT

The metaphor's most literal, most heap-different reading (SEED 2's "duck = bare running-sum thread, no priority structure touched at relax time") resolves to the well-known O(n²) array-scan Dijkstra — exactly the validated technique the problem statement itself names, not a novel invention. Its own stated risk ("only a win for dense or small graphs") is addressed directly with a `n<=3000 || density>=1/16` guard that falls back to the untouched, correct heap implementation otherwise, so the kernel can't be asymptotically worse than the known way on any input class — only, at worst, marginally slower on large sparse graphs due to the branch/duplicate-build overhead, and meaningfully faster on small/dense ones. No thread-level parallelism was added: at the sizes where this mechanism actually differs from the heap (small/dense graphs, O(n²) work), per-round work units are tiny (single array scans/relaxations), well below the threshold where OpenMP thread launch overhead would pay for itself, so vectorization-friendly scalar loops with `restrict` were used instead, per the effort-ordering rule.