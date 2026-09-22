## APPROACH

All three "mouse" solutions are Dijkstra variants for the same non-negative-weight SSSP problem, differing only in how the next node is picked:

- **Solution 1** = plain O(n²) array-scan Dijkstra (linear scan to find min each round).
- **Solution 2** = binary-heap Dijkstra (same locking order as #1, just found in O(log n) via a sorted structure) — this is exactly the reference/"known way".
- **Solution 3** = Bellman-Ford/SPFA-style repeated relaxation to fixpoint — strictly more work (O(n·m) worst case) for a problem where weights are guaranteed non-negative, so it brings no benefit here and is discarded.

Since #1 and #2 compute the *identical* result (same finalize order, same relaxations) and the contract explicitly flags that O(n²) scanning is a real practical win for dense/small graphs, the fastest robust strategy is to **build the CSR once and pick between array-scan Dijkstra and heap Dijkstra per call**, based on an estimated cost comparison of `n² + m` vs `(n+m)·log₂n`. This guarantees we're never worse than the reference (heap) approach on sparse graphs, and beat it substantially on dense/small ones by avoiding heap bookkeeping entirely.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
        int l = 2*i+1, r = 2*i+2, s = i;
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
    int *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *deg = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];

    int *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *cursor = (int*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (n > 0) memcpy(cursor, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cursor[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(deg); free(cursor);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n > 0) dist_out[source] = 0.0;

    double logn = log2((double)n + 2.0);
    double heap_cost = (double)(n + m) * logn;
    double array_cost = (double)n * (double)n + (double)m;

    if (array_cost <= heap_cost) {
        /* O(n^2) array-scan Dijkstra: wins for dense/small graphs */
        char *done = (char*)calloc((size_t)n, 1);
        for (int iter = 0; iter < n; iter++) {
            double best = INFINITY;
            int u = -1;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] < best) {
                    best = dist_out[v];
                    u = v;
                }
            }
            if (u < 0) break; /* remaining nodes unreachable */
            done[u] = 1;
            double du = dist_out[u];
            int e0 = off[u], e1 = off[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
        free(done);
    } else {
        /* Binary-heap Dijkstra: wins for sparse graphs */
        char *done = (char*)calloc((size_t)n, 1);
        HeapItem *heap = (HeapItem*)malloc((size_t)(m + 2) * sizeof(HeapItem));
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
        free(done); free(heap);
    }

    free(off); free(edst); free(ew);
}
```

PREDICTION: 1.4

## MEASUREMENT

No tools were available in this session to run `dijkstra_bench`/`dijkstra_contract`, so I could not execute the actual measurement pipeline. The prediction above (≈1.4x faster than the pure heap-based reference) is based on reasoning alone: on sparse test graphs the adaptive kernel degenerates to essentially the same heap algorithm as the reference (with minor constant-factor savings from restrict-friendly loop shapes and no extra `fill` array reuse pattern), while on dense/small graphs it switches to O(n²) array-scan, which avoids all heap push/pop overhead and is far more cache-friendly — this is where most of the predicted speedup would come from, assuming the benchmark suite contains at least some moderately dense or small-n cases.

## VERDICT

Unverified — the implementation is complete and I'm confident it is correct (it is provably equivalent to the reference: same CSR relaxation, same finalize-in-increasing-distance-order semantics, just two interchangeable strategies for picking the next node to finalize, chosen by a cost estimate that never picks the asymptotically worse option by more than a constant factor). However, without being able to actually invoke `dijkstra_bench`/`dijkstra_contract` in this session, I cannot report a measured speedup or confirm the prediction — this should be treated as an untested artifact pending an actual benchmark run.