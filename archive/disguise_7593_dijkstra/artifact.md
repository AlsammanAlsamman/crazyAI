# APPROACH

The three disguised solutions map directly onto three classic SSSP strategies:

- **Solution 1** = plain **O(n²) array-scan Dijkstra** (no priority structure at all — scan the whole board every round).
- **Solution 2** = **bucket queue / Dial's algorithm** (only works cleanly for small-range integer weights — not safe here since weights are arbitrary non-negative doubles with no known bound, so I reject this one for correctness/generality reasons).
- **Solution 3** = **binary/d-ary heap Dijkstra with lazy deletion** (exactly what the reference already does).

Since weights are arbitrary doubles, Solution 2 is unsafe to use directly (no bounded bucket range without extra assumptions), so I discard it. That leaves Solutions 1 and 3 as the two legitimate, general-purpose strategies — and the problem statement explicitly flags that for dense/small graphs the O(n²) array scan is a real practical win over a heap, while for large sparse graphs the heap's O((n+m) log n) wins asymptotically.

So I build a **hybrid**: same CSR construction as the reference (unchanged, it's already O(n+m) and near-optimal), then pick at runtime between:
1. A tight O(n²) array-scan Dijkstra (Solution 1) — no allocations inside the loop, simple branchy scan that vectorizes/pipelines well, zero heap bookkeeping overhead.
2. A 4-ary heap Dijkstra with lazy deletion (Solution 3, upgraded from binary to 4-ary) — fewer heap levels than a binary heap for the same node count, generally fewer cache misses per pop/push, same lazy-deletion correctness argument as the reference.

The choice is made by comparing a rough cost model `n²` vs `(n+m)·log2(n)·6` (the constant accounts for heap ops being several times more expensive per unit of "work" than a simple array compare). This keeps both regimes fast without gambling on graph shape.

# ARTIFACT

```c
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { double d; int u; } HeapItem;

static inline void heap_push(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 2; /* 4-ary heap parent */
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem heap_pop(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    int last = --(*hs);
    h[0] = h[last];
    int i = 0;
    while (1) {
        int c0 = 4 * i + 1;
        if (c0 >= *hs) break;
        int c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
        int s = c0;
        double sv = h[c0].d;
        if (c1 < *hs && h[c1].d < sv) { s = c1; sv = h[c1].d; }
        if (c2 < *hs && h[c2].d < sv) { s = c2; sv = h[c2].d; }
        if (c3 < *hs && h[c3].d < sv) { s = c3; sv = h[c3].d; }
        if (sv >= h[i].d) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;

    /* --- Build CSR (same counting-sort approach as reference, O(n+m)) --- */
    int *off  = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *deg  = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];

    int    *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew   = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));

    memset(deg, 0, (size_t)n * sizeof(int)); /* reuse as fill counters */
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + deg[u]++;
        edst[pos] = dst[i];
        ew[pos]   = weight[i];
    }
    free(deg);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    /* --- Cost model: pick array-scan vs heap --- */
    double logn = log2((double)n + 2.0);
    double cost_array = (double)n * (double)n;
    double cost_heap  = (double)(n + m) * logn * 6.0;

    if (cost_array <= cost_heap) {
        /* O(n^2) array-scan Dijkstra, no priority structure at all */
        char *done = (char*)calloc((size_t)n, 1);
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] < best) { best = dist_out[v]; u = v; }
            }
            if (u < 0) break; /* everything remaining is unreachable */
            done[u] = 1;
            double du = dist_out[u];
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
        free(done);
    } else {
        /* 4-ary heap Dijkstra with lazy deletion */
        char *done = (char*)calloc((size_t)n, 1);
        HeapItem *heap = (HeapItem*)malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        heap_push(heap, &hs, 0.0, source);
        while (hs > 0) {
            HeapItem top = heap_pop(heap, &hs);
            int u = top.u;
            if (done[u]) continue;
            done[u] = 1;
            double du = dist_out[u];
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    heap_push(heap, &hs, nd, v);
                }
            }
        }
        free(done);
        free(heap);
    }

    free(off);
    free(edst);
    free(ew);
}
```

PREDICTION: 1.4

# MEASUREMENT

No tools are available in this session (per the environment notice), so I could not invoke `dijkstra_bench` / `dijkstra_contract` myself this turn. The reasoning above and the prediction were produced before any run, as required; actual timing/correctness verification will need to happen in the external pipeline (or a follow-up session where tools are available) before the prediction can be checked against real numbers.

# VERDICT

Unverified this turn — correctness is architecturally sound (CSR build is byte-for-byte the same counting-sort as the reference; the heap branch is the reference's lazy-deletion algorithm with a 4-ary instead of binary heap, which preserves the exact same relaxation order/tie behavior modulo heap layout, so results should match within floating-point tolerance; the array-scan branch is textbook O(n²) Dijkstra, valid for non-negative weights). Performance claim (~1.4x) is a reasoned prediction based on: (a) avoiding heap overhead entirely on small/dense inputs where O(n²) is cache-friendly and branch-predictable, and (b) reducing heap depth (log₄ n vs log₂ n) for large sparse inputs. This needs an actual `dijkstra_bench` run to confirm or refute — flagging plainly that no measurement was performed here.