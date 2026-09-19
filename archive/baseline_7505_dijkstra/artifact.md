## APPROACH

The reference is a textbook binary-heap Dijkstra with lazy deletion. The problem statement itself flags the well-known practical fact that dense/small graphs are better served by a heap-free O(n²) array scan than by O((n+m) log n) heap operations, because the array scan is branch-light, cache-sequential, and vectorizable, while heap operations involve pointer-chasing and unpredictable branches.

So I built a **dual-algorithm kernel**:

1. **CSR build** — same idea as the reference but with one fewer full-array pass/alloc (compute offsets directly via a counting pass into `off[1..n]`, prefix-sum in place, then copy `off` into a cursor array instead of allocating separate `deg`+`fill` arrays).
2. **Algorithm selection** at runtime: if the graph is tiny (`n ≤ 64`) or dense (`m·8 ≥ n²`, i.e. ≥12.5% density), run a **heap-free O(n²+m) array-scan Dijkstra**, whose "find minimum unvisited key" step is vectorized with AVX2 (`immintrin.h`, 4 doubles/lane, tracking both min value and index), with a scalar fallback if AVX2 isn't available. This turns a red-flagged "must compare against every remaining place" scan into 4-wide SIMD compares.
3. Otherwise (the common sparse case), run a **4-ary heap Dijkstra** instead of a binary heap. A d-ary heap with d=4 has fewer levels (log₄ n vs log₂ n) and each sift-down step touches 4 contiguous children (≤2 cache lines), which in practice beats binary heaps for Dijkstra workloads. I also added a `done[v]` check before pushing on relaxation (skip pushing decrease-key updates for already-finalized neighbors), reducing wasted heap churn — a legal optimization the reference doesn't do.

This directly attacks the stated hidden assumptions: it removes the "must consult a priority structure before every relaxation" and "compare against every remaining place" costs where they're actually cheaper done differently (dense/small graphs), and it reduces the heap's per-operation constant for the sparse case.

No cross-node parallelism (OpenMP) is used for the core algorithm since Dijkstra's finalization order is inherently sequential and correctness risk from a delta-stepping-style parallel SSSP wasn't worth it at this effort level; only the CSR build is trivially fast without needing threads either.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

typedef struct { double d; int u; } HeapItem;

static inline void heap_push(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 2;
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
        int s = c0; double sv = h[c0].d;
        if (c1 < *hs && h[c1].d < sv) { s = c1; sv = h[c1].d; }
        if (c2 < *hs && h[c2].d < sv) { s = c2; sv = h[c2].d; }
        if (c3 < *hs && h[c3].d < sv) { s = c3; sv = h[c3].d; }
        if (sv >= h[i].d) break;
        HeapItem t = h[i]; h[i] = h[s]; h[s] = t;
        i = s;
    }
    return top;
}

static void dijkstra_heap(int n, const int *off, const int *edst, const double *ew,
                           int source, double *dist_out, int m) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    char *done = calloc((size_t)n, 1);
    HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0;
    heap_push(heap, &hs, 0.0, source);
    while (hs > 0) {
        HeapItem top = heap_pop(heap, &hs);
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        double du = dist_out[u];
        int end = off[u + 1];
        for (int e = off[u]; e < end; e++) {
            int v = edst[e];
            if (done[v]) continue;
            double nd = du + ew[e];
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                heap_push(heap, &hs, nd, v);
            }
        }
    }
    free(done); free(heap);
}

#if defined(__AVX2__)
static inline void argmin_vec(const double *key, int n, double *out_val, int *out_idx) {
    int i = 0;
    __m256d vmin = _mm256_set1_pd(INFINITY);
    __m256i vidx = _mm256_set1_epi64x(-1);
    __m256i vcur = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i vinc = _mm256_set1_epi64x(4);
    for (; i + 4 <= n; i += 4) {
        __m256d vals = _mm256_loadu_pd(key + i);
        __m256d cmp = _mm256_cmp_pd(vals, vmin, _CMP_LT_OQ);
        vmin = _mm256_min_pd(vals, vmin);
        vidx = _mm256_blendv_epi8(vidx, vcur, _mm256_castpd_si256(cmp));
        vcur = _mm256_add_epi64(vcur, vinc);
    }
    double tmp_val[4]; int64_t tmp_idx[4];
    _mm256_storeu_pd(tmp_val, vmin);
    _mm256_storeu_si256((__m256i *)tmp_idx, vidx);
    double best = INFINITY; long long besti = -1;
    for (int k = 0; k < 4; k++) if (tmp_val[k] < best) { best = tmp_val[k]; besti = tmp_idx[k]; }
    for (; i < n; i++) if (key[i] < best) { best = key[i]; besti = i; }
    *out_val = best; *out_idx = (int)besti;
}
#endif

static void dijkstra_array(int n, const int *off, const int *edst, const double *ew,
                            int source, double *dist_out) {
    double *key = malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) { dist_out[i] = INFINITY; key[i] = INFINITY; }
    dist_out[source] = 0.0; key[source] = 0.0;
    for (int iter = 0; iter < n; iter++) {
        double best; int u;
#if defined(__AVX2__)
        argmin_vec(key, n, &best, &u);
#else
        best = INFINITY; u = -1;
        for (int v = 0; v < n; v++) if (key[v] < best) { best = key[v]; u = v; }
#endif
        if (u < 0 || !isfinite(best)) break;
        key[u] = INFINITY;
        double du = dist_out[u];
        int end = off[u + 1];
        for (int e = off[u]; e < end; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; key[v] = nd; }
        }
    }
    free(key);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;

    int *off = calloc((size_t)(n + 1), sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];

    int *cur = malloc((size_t)n * sizeof(int));
    memcpy(cur, off, (size_t)n * sizeof(int));

    size_t msz = (size_t)(m > 0 ? m : 1);
    int *edst = malloc(msz * sizeof(int));
    double *ew = malloc(msz * sizeof(double));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cur[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(cur);

    long long nn = (long long)n * (long long)n;
    int use_array = (n <= 64) || ((long long)m * 8 >= nn);

    if (use_array) dijkstra_array(n, off, edst, ew, source, dist_out);
    else           dijkstra_heap(n, off, edst, ew, source, dist_out, m);

    free(off); free(edst); free(ew);
}
```

PREDICTION: 1.3

## MEASUREMENT

No `dijkstra_bench`/`dijkstra_contract` tool was actually available in this session (the tool list provided to me contained no such functions, despite the task text mentioning them), so I could not execute or benchmark this artifact myself. I am reporting this plainly rather than fabricating numbers: **the prediction above is unverified in this session** and must be measured by the external pipeline that compiles and runs it.

Reasoning-based expectation, stated honestly as estimate not measurement:
- Sparse/medium graphs (the likely majority case): 4-ary heap + skip-push-on-done should give a modest ~1.1–1.3x over binary-heap reference.
- Dense graphs (m·8 ≥ n²) or tiny graphs (n≤64): switching to the vectorized O(n²) array scan should give a large win (potentially several times faster), since heap-based cost there is asymptotically worse.
- Blended over an unknown mix of test cases, ~1.3x is my best single-number estimate, with high variance depending on how many dense/small instances are actually in the suite.

## VERDICT

Implementation is complete and, by construction/inspection, should be correct (same CSR construction discipline as the reference, standard 4-ary binary-heap Dijkstra with lazy deletion for sparse graphs, standard O(n²) selection-based Dijkstra with a decoupled selection-key array for dense/small graphs, AVX2 argmin with a safe scalar fallback). However, I could not run `dijkstra_bench` in this session to confirm either correctness-under-tolerance or the predicted speedup — that verification is outstanding and should be treated as the actual pass/fail gate, not my reasoning above.