## APPROACH

The reference always uses a binary-heap Dijkstra. That's asymptotically optimal for sparse graphs, but for small or dense graphs a plain O(n²+m) array-scan Dijkstra (no heap, no allocation churn beyond one boolean array) is known to win in practice: no pointer-chasing heap operations, no O(m log n) heap-push overhead, and a much more cache-friendly inner loop (linear scan over a `done[]`/`dist_out[]` array plus a CSR sweep).

Since the contract gives no hint about graph shape ahead of time, I build the CSR (needed either way) and then pick between the two algorithms per-call using a simple cost model instead of a fixed threshold, so it stays correct/fast across the whole size×density spectrum rather than just tuned for one region:

- `costArray ≈ n² + m` (array-scan Dijkstra work)
- `costHeap ≈ K·(n + m)·log2(n)` (heap Dijkstra work, K>1 to account for heap's worse cache behavior per operation)

Pick array-scan when its estimated cost is lower (small n, or dense large n); otherwise use the heap. This keeps the heap path essentially identical to the known-good reference for genuinely large sparse graphs (no regression there), while avoiding the O(n²) trap for large sparse graphs and avoiding heap overhead for small/dense graphs.

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
        int p = (i - 1) / 2;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    (*hs)--;
    h[0] = h[*hs];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n <= 0) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* build CSR (src -> dst,weight) */
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    {
        int *cursor = malloc((size_t)n * sizeof(int));
        memcpy(cursor, off, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = cursor[u]++;
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
        free(cursor);
    }

    char *done = calloc((size_t)n, 1);

    /* cost model: pick array-scan O(n^2+m) vs heap O((n+m)log n) */
    double log2n = log2((double)n + 2.0);
    double costArray = (double)n * (double)n + (double)m;
    double costHeap = 3.0 * ((double)n + (double)m) * log2n;
    int use_array = (n <= 64) || (costArray <= costHeap);

    if (use_array) {
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] < best) { best = dist_out[v]; u = v; }
            }
            if (u < 0) break;
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
        HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        hpush(heap, &hs, 0.0, source);
        while (hs > 0) {
            HeapItem top = hpop(heap, &hs);
            int u = top.u;
            if (done[u]) continue;
            done[u] = 1;
            double du = dist_out[u];
            int e0 = off[u], e1 = off[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    hpush(heap, &hs, nd, v);
                }
            }
        }
        free(heap);
    }

    free(done);
    free(off);
    free(edst);
    free(ew);
}
```

PREDICTION: 1.3

## MEASUREMENT

No benchmarking tools were available in this session (the tool list provided to me contains only Docs/Gmail/Slack/Drive tools — `dijkstra_bench`/`dijkstra_contract` were listed in the task description but not actually exposed to me here). I could not execute the artifact or compare timings myself; the reasoning above (cost model favoring array-scan on small/dense graphs, heap on large/sparse ones) is analytical, not measured. The pipeline is expected to compile and measure this artifact against the reference.

## VERDICT

Unverified pending pipeline measurement. Algorithmically: on large sparse graphs this degrades to essentially the same heap Dijkstra as the reference (expect ~parity, maybe marginal wins from inlining), and on small or dense graphs it should be meaningfully faster (no heap allocation/percolation overhead, tight cache-friendly loops) — so a net win or tie across a mixed benchmark suite is expected, with no case where it should be asymptotically worse than the reference. Cannot confirm the actual number without the measurement tool.