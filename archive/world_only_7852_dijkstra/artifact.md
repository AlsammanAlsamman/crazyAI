## APPROACH

Dijkstra is inherently sequential in its priority-selection step, so the real lever is **which data structure finds "next closest unvisited node."** The standard binary-heap solution pays `O(log n)` per relaxation via a heap, which is asymptotically best for sparse graphs but has real constant-factor overhead (pointer chasing, heap array growth, cache misses) that a plain `O(n)` linear scan doesn't have. For small or dense graphs (`m` close to `n²`), the `O(n² + m)` array-scan variant does strictly less work and touches memory more predictably, so it wins in practice — this is the well-known "dense graphs prefer array Dijkstra" fact.

Since the contract gives no hint about graph shape ahead of time, I build the CSR adjacency once (needed by both algorithms) and then **pick a strategy at runtime** by comparing the two asymptotic cost estimates: `n²+m` vs `(n+m)·log2(n)`. This keeps heap performance for genuinely sparse/large graphs and switches to the branch-light array scan for small/dense graphs, without ever being worse than either baseline by more than a small constant.

I kept the heap implementation essentially identical to the reference (same push/pop scheme, lazy deletion on pop, `m+2` upper bound on heap size) since it's already a solid array-based binary heap — no unnecessary abstraction, no decrease-key bookkeeping.

## ARTIFACT

```c
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { double d; int u; } HeapItem;

static inline void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    int ns = --(*hs);
    h[0] = h[ns];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < ns && h[l].d < h[s].d) s = l;
        if (r < ns && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;

    /* --- Build CSR adjacency once (used by both strategies) --- */
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];

    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    memset(deg, 0, (size_t)n * sizeof(int)); /* reuse deg as fill cursor */
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + deg[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(deg);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) { free(off); free(edst); free(ew); return; }
    dist_out[source] = 0.0;

    /* --- Choose strategy by asymptotic cost estimate --- */
    double logn = (n > 1) ? log2((double)n) : 1.0;
    double heap_cost  = (double)(n + m) * logn;
    double array_cost = (double)n * (double)n + (double)m;

    if (array_cost <= heap_cost) {
        /* Plain O(n^2 + m) array-scan Dijkstra: wins for small/dense graphs */
        char *done = calloc((size_t)n, 1);
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] < best) { best = dist_out[v]; u = v; }
            }
            if (u < 0) break; /* rest are unreachable */
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
        /* Binary-heap Dijkstra: wins for sparse/large graphs */
        char *done = calloc((size_t)n, 1);
        HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
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
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    hpush(heap, &hs, nd, v);
                }
            }
        }
        free(done);
        free(heap);
    }

    free(off); free(edst); free(ew);
}
```

PREDICTION: 1.4

## MEASUREMENT

No benchmarking tools (`dijkstra_bench`, `dijkstra_contract`, etc.) were exposed in this session — the tool list available to me contained only Docs/Gmail/Drive/Slack integrations, none of which can compile or run this kernel. I was unable to execute the measurement step myself. Per the task setup, the pipeline downstream of this response is expected to compile and benchmark the artifact against the reference; I cannot report real numbers here.

## VERDICT

Unverified in this session due to lack of a benchmarking tool. Reasoning-based expectation: on graphs where `m` is small relative to `n²` (typical sparse benchmark cases) this degrades to essentially the same binary-heap algorithm as the baseline (≈parity, maybe slightly slower due to the extra cost-estimate branch and CSR build being unchanged), while on small (`n` in the low thousands or less) or dense graphs it should noticeably beat the heap version by avoiding `log n` heap overhead entirely — a real win exactly in the cases the prompt flags. If the actual test suite skews toward large sparse graphs, the practical speedup will be closer to 1.0–1.1x rather than the predicted 1.4x; if it includes small/dense cases, the hybrid should clear 1.4x comfortably. This should be re-measured with the actual `dijkstra_bench` tool before trusting the prediction.