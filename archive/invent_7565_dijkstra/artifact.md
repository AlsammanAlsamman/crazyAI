# MAPPING

| World object | Problem object | Silent assumption it touches |
|---|---|---|
| **SEED 1** — waxed thread, thin/long or thick/short "exactly as long as the road is hard", fire crawling *down every thread at once* at the *same unhurried pace* | thread = directed edge (u→v); thread length ∝ edge weight w; "fire crawls" = tentative arrival value advancing; *all live threads compared simultaneously* instead of one-at-a-time | Breaks **"a priority structure must be consulted before every relaxation"** — the min-arrival knot is found by comparing all live candidates together (a batch/SIMD reduction), not by pushing/popping a heap |
| **SEED 2** — duck-fat bead fixed at first arrival, later longer flame pinched dead and fed to the fire in the clearing | bead = finalized distance at a node; pinching a later flame = rejecting a relaxation into an already-settled node | Matches, does **not break**, "each place's distance must be finalized before its neighbors are explored" and "a road can only be considered once its starting place is fully settled" — this is exactly standard Dijkstra's settle-once rule, restated |
| **SEED 3** — dark, beadless knots left dark on purpose, traveler told not to trust them | unreachable node, dist = INFINITY | Matches, does **not break**, anything — it's the ordinary sentinel-value handling every Dijkstra implementation already does |

Check against "the whole graph must be explored to know any single distance": none of the three seeds touch this. The fire still must physically propagate edge-by-edge outward from the source knot to reach any target knot — there is no shortcut in the story that reveals one distance without simulating the crawl through the graph's structure. **So none of the three seeds breaks that assumption; stated plainly, and falling back to the most literal seed as instructed.**

# CHOSEN SEED
SEED 1 — the simultaneous, equal-paced, whole-cloth fire crawl. It is the most literal (thread length = weight, uniform crawl speed, "at once") and the most different from the reference's serial heap-pop loop.

# ASSUMPTION BROKEN
"A priority structure must be consulted before every relaxation." Instead of a binary heap, the next knot to bead is found by comparing every live thread-end simultaneously — literally realized as a SIMD (4-wide AVX) linear scan over tentative distances. Per instruction #4, this is exactly the well-known, validated **array-scan Dijkstra** already named in the problem's own `known_way` text ("a plain O(n^2) array scan, with no heap at all, is a real, well-known practical win") — the metaphor is let to arrive at that known technique rather than inventing something untested. Per instruction #5, the `known_way` names two regimes (sparse/heap vs. dense-or-small/array-scan), so the kernel must recognize which regime it's in and fall back — implemented as a runtime density/size check with the reference's own heap Dijkstra as the sparse/large fallback, so the worst case is parity with the baseline, never asymptotically worse.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

typedef struct { double d; int u; } HeapItem;
static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) / 2; if (h[p].d <= h[i].d) break; HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}

/* "The fire does not leap from knot to knot; it crawls, down every thread
   at once, at the same unhurried pace along each one." Find the unvisited
   knot whose flame arrives first by comparing all live threads together in
   SIMD lanes instead of consulting a heap one pop at a time. */
static int fire_argmin(const double * restrict key, int n) {
    int i = 0;
    __m256d minVal = _mm256_set1_pd(INFINITY);
    __m256d minIdx = _mm256_set1_pd(-1.0);
    __m256d curIdx = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
    const __m256d four = _mm256_set1_pd(4.0);
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(&key[i]);
        __m256d cmp = _mm256_cmp_pd(v, minVal, _CMP_LT_OQ);
        minVal = _mm256_blendv_pd(minVal, v, cmp);
        minIdx = _mm256_blendv_pd(minIdx, curIdx, cmp);
        curIdx = _mm256_add_pd(curIdx, four);
    }
    double vb[4], ib[4];
    _mm256_storeu_pd(vb, minVal);
    _mm256_storeu_pd(ib, minIdx);
    double bestVal = INFINITY; int bestIdx = -1;
    for (int k = 0; k < 4; k++) if (vb[k] < bestVal) { bestVal = vb[k]; bestIdx = (int)ib[k]; }
    for (; i < n; i++) if (key[i] < bestVal) { bestVal = key[i]; bestIdx = i; }
    return bestIdx;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) { int u = src[i]; int pos = off[u] + fill[u]++; edst[pos] = dst[i]; ew[pos] = weight[i]; }
    free(fill);

    double * restrict d = dist_out;
    for (int i = 0; i < n; i++) d[i] = INFINITY;
    if (n > 0) d[source] = 0.0;

    /* Regime check ("is this a thick, small weave, or a thin, sprawling
       one?"): mirrors the two regimes the known solution itself names —
       dense/small -> flat simultaneous fire-crawl wins; sparse/large ->
       a priority wick (heap) wins. */
    double avg_deg = (n > 0) ? (double)m / (double)n : 0.0;
    int dense_or_small = (n <= 2000) || (avg_deg >= 0.1 * (double)n);

    if (!dense_or_small) {
        /* sparse/large fallback: unchanged binary-heap Dijkstra (known way) */
        char *done = calloc((size_t)n, 1);
        HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        hpush(heap, &hs, 0.0, source);
        while (hs > 0) {
            HeapItem top = hpop(heap, &hs);
            int u = top.u;
            if (done[u]) continue;
            done[u] = 1;
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = d[u] + ew[e];
                if (nd < d[v]) { d[v] = nd; hpush(heap, &hs, nd, v); }
            }
        }
        free(done); free(heap);
    } else {
        /* dense/small: vectorized simultaneous fire-crawl */
        char *visited = calloc((size_t)n, 1);
        double *key = malloc((size_t)n * sizeof(double));
        memcpy(key, d, (size_t)n * sizeof(double));

        for (int iter = 0; iter < n; iter++) {
            int u = fire_argmin(key, n);
            if (u < 0) break; /* every remaining knot dark: unreachable, left beadless */
            visited[u] = 1;
            key[u] = INFINITY; /* bead pressed: nothing more owed to this knot */
            double du = d[u];
            int lo = off[u], hi = off[u + 1];
            for (int e = lo; e < hi; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < d[v]) {
                    d[v] = nd;
                    if (!visited[v]) key[v] = nd;
                }
            }
        }
        free(visited); free(key);
    }

    free(deg); free(off); free(edst); free(ew);
}
```

PREDICTION: speedup_vs_heap = 1.5

# MEASUREMENT
No `dijkstra_bench` / `dijkstra_contract` tool was reachable in this session (only Slack/Gmail/Drive/Docs tools were available, and the outer instructions for this task explicitly said no tools are available here — "the pipeline will compile and measure it"). I did not fabricate a number: the prediction above (1.5×) is a stated estimate, not a measured result, and I ran **0 of the allowed 4 improvement iterations** because there was no measurement feedback loop to react to. This should be reported as a plain limitation, not papered over.

# VERDICT
The mechanism is a literal reading of SEED 1: uniform-speed simultaneous fire-crawl → replace serial heap-pop with a SIMD batch comparison over all live tentative distances, which is exactly the known, validated array-scan Dijkstra the problem statement itself names for the dense/small regime. Per instruction #5, since `known_way` names two regimes, the kernel detects density/size at runtime (`avg_deg`, `n`) and falls back verbatim to the reference's binary-heap Dijkstra for the sparse/large regime — so in that regime the kernel is byte-for-byte the known way (no regression possible, speedup ≈ 1.0×), and only in the dense/small regime does it take the vectorized path where it should win.

Named risk and how it was addressed: an OpenMP-parallel relaxation loop ("fire crawling down every thread at once" taken as literal thread-level parallelism) was considered, since the metaphor's own units of work (out-edges of a node) map directly onto parallel work items. It was **dropped**, not shipped, because parallel writes to shared `dist_out[v]`/`key[v]` for duplicate destination edges within one node's adjacency create a genuine, unaddressed data race; instruction #4 requires guarding or dropping a named risk, and no cheap safe guard (atomic CAS on doubles) was available within scope, so vectorization (SIMD) alone was kept as the mechanism, consistent with "default to vectorization before thread-level parallelism."

Bottom line: correctness should hold on both regimes (dense path is provably the same array-scan algorithm as the reference's known-good technique; sparse path is the reference's own heap code, unmodified). The predicted 1.5× is unverified in this session pending an actual `dijkstra_bench` run.