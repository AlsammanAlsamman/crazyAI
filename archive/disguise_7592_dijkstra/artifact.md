# APPROACH

The reference solution is a fine general-purpose lazy-deletion binary-heap Dijkstra. The problem statement itself flags the one real algorithmic lever available here: for **dense** graphs (m close to n²) or **small** n, a heap adds O(log n) overhead per relaxation for no benefit, and a plain O(n²+m) array-scan Dijkstra (no heap, no allocations per push, pure sequential/branch-predictable memory access) beats it in practice. For sparse, large graphs the binary heap remains asymptotically and practically the right choice.

So: build the CSR representation once (same counting-sort approach as the reference — correct and already near-optimal for that step), then pick one of two well-known correct Dijkstra variants based on a density/size heuristic:

- `n <= 1500` **or** `m*6 >= n²`  → O(n²+m) array-scan Dijkstra (no heap allocation churn at all).
- otherwise → binary-heap lazy-deletion Dijkstra (essentially the reference's proven-correct core, with `restrict` pointers and a tightened relax loop).

Both branches preserve the same invariants the standard solution relies on (finalize-before-explore, non-decreasing finalization order under non-negative weights), so correctness carries over from the well-known proofs for each variant individually — I'm not inventing a new algorithm, just routing to the right classical one.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Binary heap (lazy-deletion) for sparse graphs ---- */
typedef struct { double d; int u; } HeapItem;

static inline void hpush(HeapItem *restrict h, int *restrict hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline HeapItem hpop(HeapItem *restrict h, int *restrict hs) {
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

    /* Build CSR via counting sort (O(n+m)) */
    int *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *deg = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];

    int *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    {
        int *cursor = (int*)malloc((size_t)n * sizeof(int));
        memcpy(cursor, off, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = cursor[u]++;
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
        free(cursor);
    }
    free(deg);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) { free(off); free(edst); free(ew); return; }
    dist_out[source] = 0.0;

    /* Density/size heuristic: dense or small -> array scan; else heap */
    int use_array = (n <= 1500) || ((long long)m * 6LL >= (long long)n * (long long)n);

    if (use_array) {
        char *visited = (char*)calloc((size_t)n, 1);
        double *key = (double*)malloc((size_t)n * sizeof(double));
        memcpy(key, dist_out, (size_t)n * sizeof(double));

        for (int iter = 0; iter < n; iter++) {
            double best = INFINITY;
            int bu = -1;
            for (int i = 0; i < n; i++) {
                double kv = key[i];
                if (kv < best) { best = kv; bu = i; }
            }
            if (bu < 0) break;
            visited[bu] = 1;
            key[bu] = INFINITY;
            double du = dist_out[bu];
            for (int e = off[bu]; e < off[bu + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    key[v] = nd; /* harmless if v already visited: nd can't be < dist_out[v] then */
                }
            }
        }
        free(visited);
        free(key);
    } else {
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

PREDICTION: 1.5

# MEASUREMENT

No benchmarking tools (`dijkstra_bench`/`dijkstra_contract`) were reachable in this session, so the artifact above was **not executed or timed by me** — I did not fabricate numbers. Reasoning about expected results instead of measured ones:

- Sparse, large-n instances (m = O(n)) will route to the heap branch, which is structurally the same lazy-deletion binary heap as the reference (same push/pop logic), so I expect roughly parity, maybe a few percent faster from `restrict` and tighter codegen — not a meaningful win.
- Dense instances (m ≳ n²/6) or small-n instances will route to the O(n²+m) array-scan branch, which avoids all heap allocation/percolation overhead (O(m log n) → O(n²+m)); this is where the real win should show, potentially several-x on the densest cases.
- Overall average across a mixed benchmark suite is what PREDICTION=1.5 estimates — driven up by the dense/small cases, pulled down toward 1.0 by the sparse cases.

# VERDICT

Correctness: both branches are textbook-correct non-negative-weight Dijkstra variants (finalize-in-nondecreasing-order invariant holds for each), CSR construction is unchanged from the reference's proven approach, and the contract signature/semantics (INFINITY for unreachable, `dist_out` length n) are preserved exactly. I did not verify this by actually running `dijkstra_contract` in this session — that check still needs to happen against the harness before this can be called confirmed-correct, not just argued-correct. Performance claim is an untested prediction based on complexity analysis, not a measured result; flagging that plainly rather than presenting it as verified.