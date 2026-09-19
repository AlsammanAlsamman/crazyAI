## APPROACH

The reference is a plain binary-heap Dijkstra with lazy deletion (push-on-relax, skip stale pops). Its known weaknesses:

1. **Binary heap overhead** for sparse graphs — each push/pop touches `log2(n)` levels with a full swap chain.
2. **No adaptation to graph density** — for dense graphs (`m` close to `n²`) or small `n`, a heap is pure overhead: the classic `O(n²)` array-scan Dijkstra (no heap, no allocation churn) does strictly less work and is far more cache-friendly.
3. **Redundant heap pushes** for edges into already-finalized nodes.

My implementation:
- Builds the same CSR layout (needed either way).
- Picks, per call, between two kernels using a simple cost model (`n²` vs `(n+m)·log2(n)`), matching the problem's explicit hint that dense/small graphs favor array-scan.
- For the sparse/heap path, replaces the binary heap with a **4-ary heap** using struct-of-arrays layout and *insertion-shift* (move-based) push/pop instead of swap-based — fewer memory writes, better cache use, shallower tree than binary heap for a given size.
- Skips relaxation into already-`done` nodes (valid since weights are non-negative — a finalized distance can never decrease), avoiding wasted heap pushes.
- Avoids the extra `fill` array from the reference by reusing a `cur[]` copy of `off[]`.

This should not regress on sparse graphs (4-ary heap is a well-known win of roughly 10–25% over binary heap for pure extract-min/insert workloads) and should win decisively on dense/small graphs by avoiding heap machinery entirely.

## ARTIFACT

```c
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define HD 4

static inline void heap_push(double * __restrict hd, int * __restrict hu, int *hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) / HD;
        if (hd[p] <= d) break;
        hd[i] = hd[p]; hu[i] = hu[p];
        i = p;
    }
    hd[i] = d; hu[i] = u;
}

static inline void heap_pop(double * __restrict hd, int * __restrict hu, int *hs, double *outd, int *outu) {
    *outd = hd[0]; *outu = hu[0];
    int n = --(*hs);
    double d = hd[n]; int u = hu[n];
    int i = 0;
    for (;;) {
        int first = HD * i + 1;
        if (first >= n) break;
        int s = first;
        double sval = hd[first];
        int last = first + HD; if (last > n) last = n;
        for (int c = first + 1; c < last; c++) {
            if (hd[c] < sval) { sval = hd[c]; s = c; }
        }
        if (sval >= d) break;
        hd[i] = hd[s]; hu[i] = hu[s];
        i = s;
    }
    hd[i] = d; hu[i] = u;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    int *off  = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *edst = (int*)malloc((size_t)m * sizeof(int));
    double *ew = (double*)malloc((size_t)m * sizeof(double));
    {
        int *cnt = (int*)calloc((size_t)n, sizeof(int));
        for (int i = 0; i < m; i++) cnt[src[i]]++;
        off[0] = 0;
        for (int i = 0; i < n; i++) off[i + 1] = off[i] + cnt[i];
        int *cur = (int*)malloc((size_t)n * sizeof(int));
        memcpy(cur, off, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = cur[u]++;
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
        free(cnt); free(cur);
    }

    double log2n = log2((double)n + 2.0);
    double cost_dense = (double)n * (double)n;
    double cost_heap  = ((double)n + (double)m) * log2n * 1.5;
    int use_dense = cost_dense < cost_heap;

    const int * __restrict Poff = off;
    const int * __restrict Pedst = edst;
    const double * __restrict Pew = ew;

    if (use_dense) {
        char *done = (char*)calloc((size_t)n, 1);
        double * __restrict D = dist_out;
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int i = 0; i < n; i++) {
                if (!done[i] && D[i] < best) { best = D[i]; u = i; }
            }
            if (u < 0) break;
            done[u] = 1;
            double du = D[u];
            int e0 = Poff[u], e1 = Poff[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = Pedst[e];
                double nd = du + Pew[e];
                if (nd < D[v]) D[v] = nd;
            }
        }
        free(done);
    } else {
        char *done = (char*)calloc((size_t)n, 1);
        double *hd = (double*)malloc((size_t)(m + 2) * sizeof(double));
        int *hu = (int*)malloc((size_t)(m + 2) * sizeof(int));
        int hs = 0;
        heap_push(hd, hu, &hs, 0.0, source);
        while (hs > 0) {
            double d; int u;
            heap_pop(hd, hu, &hs, &d, &u);
            if (done[u]) continue;
            done[u] = 1;
            double du = dist_out[u];
            int e0 = Poff[u], e1 = Poff[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = Pedst[e];
                if (done[v]) continue;
                double nd = du + Pew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    heap_push(hd, hu, &hs, nd, v);
                }
            }
        }
        free(done); free(hd); free(hu);
    }

    free(off); free(edst); free(ew);
}
```

PREDICTION: 1.3

## MEASUREMENT

No benchmarking tool (`dijkstra_bench`) was available in this session — the system explicitly indicated no tools were provided and that the pipeline would compile and measure the artifact after submission. I have not executed or timed this code myself; the prediction above (≈1.3× the reference's throughput, averaged over a mix of sparse/dense/small graph shapes) is derived purely from algorithmic reasoning (4-ary heap vs binary heap literature gains of ~10–25% on sparse graphs, plus an expected large win on dense/small graphs where the reference pays `O(m log n)` for what a `O(n²)` scan does with far less overhead).

## VERDICT

Unverified — correctness follows from the same invariants as the reference (non-negative weights ⇒ once-finalized distances are final; CSR construction is a straightforward stable bucket build), and the two code paths are individually well-established (binary/d-ary heap Dijkstra, array-scan Dijkstra). The hybrid selection is a heuristic and its crossover constant (1.5) is a guess rather than tuned; if the actual measurement shows the heuristic mis-classifies the benchmark's graph shapes, the fix would be to adjust that constant rather than the algorithms themselves. Pending real numbers from `dijkstra_bench`.