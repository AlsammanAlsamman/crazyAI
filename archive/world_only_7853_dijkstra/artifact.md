# APPROACH

The reference implementation is textbook binary-heap Dijkstra: O((n+m) log n). The problem statement itself hints that for dense or small graphs a plain O(n²) array-scan Dijkstra (no heap at all) is a real, well-known practical win — because for those graphs the O(n²) scan term is smaller than the m·log(n) heap-push term, *and* the array scan has far better cache behavior and no heap-management branching/pointer chasing.

Since we don't know a priori whether a given test graph is sparse-large or dense/small, the fastest robust strategy is **adaptive**: build the CSR once (same cost either way), then pick the algorithm cheaply from `n` and the edge density `m/n²`:

- If `n` is small (heap constant-factor overhead dominates regardless of asymptotics) **or** the graph is dense (`m` close to `n²`), run **O(n²) array-scan Dijkstra** — tight double loops, no allocation beyond one `done` array, extremely cache- and branch-predictor-friendly.
- Otherwise (large, sparse graph) run **heap Dijkstra**, but with a **4-ary heap** instead of binary — fewer levels (log₄ n vs log₂ n comparisons... actually same number of moves but fewer *levels* of pointer chasing) and better cache locality per heap node, which is a well-known practical speedup over binary heaps for Dijkstra.

CSR construction is kept single-threaded and identical in spirit to the reference (counting sort), since it's O(n+m) and not the bottleneck for either regime; adding OpenMP there would only add thread-spawn overhead risk for small/medium graphs without a clear win.

This keeps worst case matching the reference's asymptotic complexity while giving a real win on the dense/small end that the problem explicitly flags.

# ARTIFACT

```c
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { double d; int u; } HeapItem;

/* 4-ary heap: fewer levels / better cache locality than binary heap */
static inline void hpush4(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem hpop4(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    int last = --(*hs);
    h[0] = h[last];
    int i = 0, sz = *hs;
    while (1) {
        int c0 = 4 * i + 1;
        if (c0 >= sz) break;
        int best = c0;
        double bd = h[c0].d;
        int c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
        if (c1 < sz && h[c1].d < bd) { best = c1; bd = h[c1].d; }
        if (c2 < sz && h[c2].d < bd) { best = c2; bd = h[c2].d; }
        if (c3 < sz && h[c3].d < bd) { best = c3; bd = h[c3].d; }
        if (bd >= h[i].d) break;
        HeapItem t = h[i]; h[i] = h[best]; h[best] = t;
        i = best;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* Build CSR (counting sort by src) */
    int *deg = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew = (double *)malloc((size_t)m * sizeof(double));
    int *cursor = (int *)malloc((size_t)n * sizeof(int));
    memcpy(cursor, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cursor[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(deg);
    free(cursor);

    /* Choose algorithm: O(n^2) array scan for small/dense graphs
       (low constant factor, no heap overhead, cache-friendly),
       else O((n+m) log n) 4-ary heap for large sparse graphs. */
    double density = (double)m / ((double)n * (double)n + 1.0);
    int use_array = (n <= 2048) || (density > 0.05);

    if (use_array) {
        char *done = (char *)calloc((size_t)n, 1);
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int i = 0; i < n; i++) {
                if (!done[i] && dist_out[i] < best) { best = dist_out[i]; u = i; }
            }
            if (u < 0) break;
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
        char *done = (char *)calloc((size_t)n, 1);
        HeapItem *heap = (HeapItem *)malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        hpush4(heap, &hs, 0.0, source);
        while (hs > 0) {
            HeapItem top = hpop4(heap, &hs);
            int u = top.u;
            if (done[u]) continue;
            done[u] = 1;
            double du = dist_out[u];
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    hpush4(heap, &hs, nd, v);
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

PREDICTION: 2.0

# MEASUREMENT

No benchmarking tools (`dijkstra_bench`, `dijkstra_contract`, etc.) were actually available in this session — the environment note explicitly stated no tools would be invoked, so I could not run the compile/measure/iterate loop the task normally calls for. I'm reporting this plainly rather than fabricating numbers: the PREDICTION above (≈2x aggregate speedup) is an analytical estimate, not a measured result.

Reasoning behind the estimate:
- On dense/small graphs (the case the problem explicitly flags), array-scan Dijkstra avoids all heap push/pop overhead and duplicate-entry bloat (the reference heap re-pushes a node on every relaxation, so heap size can reach O(m)); it should be meaningfully faster there, plausibly 3–6x.
- On large sparse graphs, the 4-ary heap should give a modest win (typically cited as 10–30% over binary heap in practice) from fewer cache-line jumps per level, with identical asymptotics and identical duplicate-push behavior to the reference — so no regression is expected, only a smallish gain.
- The threshold (`n <= 2048` or `density > 0.05`) is a heuristic, not tuned against real data, so it could misclassify some medium-density/medium-size graphs and leave a smaller win than ideal on those; this is the biggest source of prediction uncertainty.

# VERDICT

Unverified. The code is a straightforward, contract-compliant adaptive Dijkstra (O(n²) array scan for small/dense inputs, 4-ary-heap Dijkstra otherwise) that should be at least as fast as the reference in every regime and meaningfully faster on the dense/small regime the problem calls out — but since no measurement tool actually ran in this session, I cannot confirm the predicted ~2x figure and am reporting the absence of real measurement rather than claiming a result I don't have.