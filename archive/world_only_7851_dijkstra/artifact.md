## APPROACH

The reference solution always pays `O((n+m) log n)` via a binary heap, even though the problem statement itself flags that array-scan Dijkstra (`O(n²+m)`, no heap, no `log n` per edge) is a well-known practical win on dense or small graphs, where heap bookkeeping (allocation, sift-up/down, cache-unfriendly pointer chasing) costs more than a tight linear scan does.

So I keep the CSR build (unavoidable, `O(n+m)`, needed either way) but make the actual shortest-path phase **adaptive**:

- Estimate heap cost ≈ `m·log2(n)` (edge relaxations dominate pushes) vs array-scan cost ≈ `n²+m`.
- If `n` is small (≤1500, where any algorithm is fast but heap constant-factor/alloc overhead dominates) **or** the graph is dense enough that `n² ≲ 3·m·log2(n)`, run the branch-free array-scan variant.
- Otherwise (large sparse graph — the heap's actual sweet spot) fall back to the same binary-heap approach as the reference, just with `cur[]` cursor instead of `fill[u]++` addition (saves one add per edge in CSR build).

The array-scan variant uses a second buffer `sdist[]` that mirrors `dist_out[]` but gets set to `+INF` when a node is settled — this turns "find unsettled min" into a single branch-free linear min-reduction over a plain double array, which the compiler can autovectorize under `-O3 -march=native`, instead of an `if(!done[u])` branchy scan.

This should match the reference exactly in output (same relaxation semantics, non-negative weights ⇒ both algorithms are exact Dijkstra) while being faster on dense/small inputs and roughly tied on large sparse ones.

## ARTIFACT

```c
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct { double d; int u; } HeapItem;

static inline void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 1;
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
    int hs_local = *hs;
    while (1) {
        int l = 2*i+1, r = 2*i+2, s = i;
        if (l < hs_local && h[l].d < h[s].d) s = l;
        if (r < hs_local && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;

    /* Build CSR */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew  = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *cur = malloc((size_t)n * sizeof(int));
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cur[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(deg); free(cur);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    double logn = log2((double)n + 2.0);
    int use_array = (n <= 1500) || ((double)m * logn * 3.0 >= (double)n * (double)n);

    if (use_array) {
        double *sdist = malloc((size_t)n * sizeof(double));
        memcpy(sdist, dist_out, (size_t)n * sizeof(double));
        for (int iter = 0; iter < n; iter++) {
            double best = INFINITY;
            int bu = -1;
            for (int i = 0; i < n; i++) {
                double di = sdist[i];
                if (di < best) { best = di; bu = i; }
            }
            if (bu < 0) break;
            sdist[bu] = INFINITY;
            double du = dist_out[bu];
            for (int e = off[bu]; e < off[bu + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    sdist[v] = nd;
                }
            }
        }
        free(sdist);
    } else {
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
        free(done); free(heap);
    }

    free(off); free(edst); free(ew);
}
```

PREDICTION: 1.4

## MEASUREMENT

No benchmarking tools were available in this session (explicitly stated: "No tools are available in this session... the pipeline will compile and measure it"). I was unable to run `dijkstra_bench` or `dijkstra_contract` myself, so no empirical numbers are produced here — the prediction above stands unverified by me.

## VERDICT

Unmeasured. Reasoning-only expectation: correctness should match the reference exactly (identical Dijkstra relaxation logic, non-negative weights, same CSR construction), and the adaptive array-scan/heap switch should give a real (though input-dependent) speedup on dense or small graphs while being at worst roughly on par with the reference on large sparse graphs where it falls back to the same heap algorithm. Actual speedup depends entirely on the benchmark's graph mix, which I could not observe.