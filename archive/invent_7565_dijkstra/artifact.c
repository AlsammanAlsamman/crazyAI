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
