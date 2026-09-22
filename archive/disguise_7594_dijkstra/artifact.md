# APPROACH

Of the three retellings, **Solution 2 is exactly the reference algorithm** (binary-heap Dijkstra with lazy deletion), and **Solution 1 is the classic O(n²) array-scan Dijkstra** (no heap, linear scan for the minimum each round). Solution 3 (SPFA-style unconditional gossip/Bellman-Ford relaxation) is *not* the fast path here — without the "settle before propagating" discipline it can revisit nodes many times and has no good worst-case bound; it's a correctness curiosity, not a speed play, for non-negative weights.

The real lever is: **the settling rule (must-settle-before-relaxing) and the "must consult a priority structure" rule are both true requirements for correctness with a heap, but the *method* of finding "next smallest" is a free choice.** For sparse graphs the heap (Solution 2) is asymptotically better; for dense graphs or small n, the O(n²) scan (Solution 1) beats it in practice because:
- it avoids O(log n) heap-push overhead per edge relaxation,
- the linear scan over a contiguous `dist`/`done` array is trivially SIMD-friendly and cache-friendly,
- it avoids the extra heap array allocation and pointer-chasing.

So the fastest general-purpose kernel is a **hybrid**: build CSR adjacency once, then pick array-scan Dijkstra when n is small or the graph is dense (crossover roughly where `avg_degree > n / log2(n)`), else fall back to the binary-heap Dijkstra (same as reference, essentially). I also removed the reference's extra `fill[]` array in CSR construction (reuse a `cursor` array copied from `off`, one fewer allocation/pass) and split the array-scan's "find minimum" into a value pass and an index pass so the hot reduction loop vectorizes cleanly (no argmin dependency chain in the vectorizable loop).

# ARTIFACT

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

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;

    /* --- Build CSR adjacency (counting sort, no separate fill[] array) --- */
    int *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];

    int *cursor = (int*)malloc((size_t)n * sizeof(int));
    memcpy(cursor, off, (size_t)n * sizeof(int));

    int *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cursor[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(cursor);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    char *done = (char*)calloc((size_t)n, 1);

    /* Choose strategy: array-scan O(n^2 + m) vs binary-heap O((n+m) log n).
       Crossover roughly where n^2 ~ m*log2(n)  =>  avg_deg ~ n/log2(n).
       Small n is cheap and cache-friendly as array-scan regardless. */
    double avg_deg = (double)m / (double)n;
    double log2n = n > 1 ? log2((double)n) : 1.0;
    int use_dense = (n <= 4096) || (avg_deg > (double)n / log2n);

    if (use_dense) {
        for (int iter = 0; iter < n; iter++) {
            double best = INFINITY;
            for (int v = 0; v < n; v++) {
                double dv = done[v] ? INFINITY : dist_out[v];
                if (dv < best) best = dv;
            }
            if (best == INFINITY) break;
            int u = -1;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] == best) { u = v; break; }
            }
            done[u] = 1;
            double du = dist_out[u];
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
    } else {
        HeapItem *heap = (HeapItem*)malloc((size_t)(m + 2) * sizeof(HeapItem));
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
        free(heap);
    }

    free(off); free(edst); free(ew); free(done);
}
```

PREDICTION: 1.4

# MEASUREMENT

No benchmarking tool (`dijkstra_bench`/`dijkstra_contract`) is present in this session's actual tool set — only Docs/Gmail/Drive/Slack tools are available, so I cannot execute the harness myself. Reasoning about expected behavior instead: for small/dense test graphs the array-scan branch should noticeably beat the reference heap (removing `log n` per-edge overhead and enabling vectorized min-finding); for large sparse graphs the heap branch is nearly identical to the reference (same algorithm, one fewer allocation), so gains there should be small (near 1.0x–1.1x), possibly with slight overhead from the `avg_deg`/`log2` branch check (negligible, O(1)). Net expected improvement across a mixed size/density benchmark suite: modest, single-digit-to-40% speedup, dominated by whatever fraction of test cases are dense/small.

# VERDICT

Unverified in this session — no measurement tool was reachable to confirm the prediction. The implementation is a straightforward, believed-correct hybrid of Solution 1 (array-scan) and Solution 2/reference (binary heap), selected by a principled density threshold; it should be at worst roughly on par with the reference (sparse/large case ≈ same algorithm) and meaningfully faster on dense/small graphs, but the actual verdict must come from running it through the real `dijkstra_bench` pipeline.