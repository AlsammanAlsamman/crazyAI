APPROACH:
The reference is textbook binary-heap Dijkstra: O((n+m) log n), lazy deletion, CSR built once. Its known weakness is that for dense graphs (m close to n²) or graphs where average out-degree is large, the log n heap factor is pure overhead compared to a plain O(n²) array-scan Dijkstra, which has no heap bookkeeping and a linear-scan inner loop that auto-vectorizes trivially under `-O3 -march=native`. Conversely, for sparse graphs the heap approach is asymptotically better and should stay.

So the fastest *general* kernel is adaptive: build the CSR once (unavoidable, O(n+m)), then pick the algorithm based on the actual density of the given graph:
- **Dense path** (m roughly ≥ n²/16): plain O(n²) scan-based Dijkstra using a `key[]` array (distance if unvisited, +∞ once finalized) so the "find min unvisited" step is a single unconditional linear scan (no branch on a separate `done[]` array), which the compiler vectorizes well.
- **Sparse path**: the same lazy-deletion binary heap as the reference, but with push/pop inlined directly into the main loop (no function-call indirection, no struct-copy through helper calls) and `restrict`-qualified local pointers to help the compiler avoid alias checks in the hot relaxation loop.

This keeps the sparse case at parity with (or slightly ahead of) the reference, and gives a real asymptotic win on dense/small-diameter graphs, matching exactly the "known way" hint in the problem statement.

I did not attempt OpenMP parallelization of the min-scan or relaxation: a per-iteration OpenMP `parallel for` with manual arg-min reduction adds thread-spawn/join overhead on every one of the n outer iterations, which for the n sizes typically used in these benchmarks (tens of thousands or fewer) is very likely to be a net loss, and a race on duplicate-edge relaxation writes is a correctness risk I'd rather not take for an unclear payoff. I judged this not worth the risk without being able to measure it.

ARTIFACT:
```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { double d; int u; } HeapItem;

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    const int * restrict rsrc = src;
    const int * restrict rdst = dst;
    const double * restrict rw = weight;

    /* Build CSR (unavoidable, O(n+m)) */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[rsrc[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    {
        int *fill = calloc((size_t)n, sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = rsrc[i];
            int pos = off[u] + fill[u]++;
            edst[pos] = rdst[i];
            ew[pos] = rw[i];
        }
        free(fill);
    }
    free(deg);

    const int * restrict roff = off;
    const int * restrict redst = edst;
    const double * restrict rew = ew;

    /* Adaptive choice: O(n^2) array scan for dense graphs, heap for sparse */
    double density = (double)m / ((double)n * (double)n + 1.0);
    int use_dense = (n > 1) && (density > 0.0625);

    if (use_dense) {
        double * restrict key = malloc((size_t)n * sizeof(double));
        memcpy(key, dist_out, (size_t)n * sizeof(double));

        for (int iter = 0; iter < n; iter++) {
            double best = INFINITY;
            int u = -1;
            for (int v = 0; v < n; v++) {
                double kv = key[v];
                if (kv < best) { best = kv; u = v; }
            }
            if (u < 0 || best == INFINITY) break;
            key[u] = INFINITY;
            double du = dist_out[u];
            int end = roff[u + 1];
            for (int e = roff[u]; e < end; e++) {
                int v = redst[e];
                double nd = du + rew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    key[v] = nd;
                }
            }
        }
        free(key);
    } else {
        HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        char *done = calloc((size_t)n, 1);

        heap[0].d = 0.0; heap[0].u = source; hs = 1;

        while (hs > 0) {
            HeapItem top = heap[0];
            hs--;
            heap[0] = heap[hs];
            {
                int i = 0;
                while (1) {
                    int l = 2 * i + 1, r = 2 * i + 2, s = i;
                    if (l < hs && heap[l].d < heap[s].d) s = l;
                    if (r < hs && heap[r].d < heap[s].d) s = r;
                    if (s == i) break;
                    HeapItem t = heap[s]; heap[s] = heap[i]; heap[i] = t;
                    i = s;
                }
            }
            int u = top.u;
            if (done[u]) continue;
            done[u] = 1;
            double du = dist_out[u];
            int end = roff[u + 1];
            for (int e = roff[u]; e < end; e++) {
                int v = redst[e];
                double nd = du + rew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    int i = hs++;
                    heap[i].d = nd; heap[i].u = v;
                    while (i > 0) {
                        int p = (i - 1) / 2;
                        if (heap[p].d <= heap[i].d) break;
                        HeapItem t = heap[p]; heap[p] = heap[i]; heap[i] = t;
                        i = p;
                    }
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

MEASUREMENT:
No benchmarking tools (`dijkstra_bench`, `dijkstra_contract`, etc.) were actually available in this session — the environment explicitly stated that no tools could be invoked, so I did not run, and could not run, any measurement. I am not going to fabricate numbers. The 1.4 prediction above is a reasoning-based estimate (near-parity on sparse graphs from the inlined heap, larger wins — plausibly 2–5x — specifically on dense/high-average-degree graphs from dropping the log n factor and getting a vectorizable scan, averaged against an unknown mix of test-graph shapes) rather than an observed result.

VERDICT:
Unverified. The implementation is a straightforward, low-risk generalization of the reference (same CSR build, same lazy-deletion heap logic for the sparse case, standard textbook O(n²) Dijkstra for the dense case) so I have reasonable confidence it is *correct*, but I have no actual timing evidence that it is faster, and I'm reporting that gap honestly rather than claiming a measured win I don't have.