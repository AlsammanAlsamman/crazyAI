#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------------- sparse/general path: binary-heap Dijkstra ----------------
   Validated, well-known technique for the sparse/large regime. We do not
   reinvent it -- we fall back to it whenever the dense array-scan metaphor's
   own regime (small palace, or richly-corridored palace) does not apply. */
typedef struct { double d; int u; } HeapItem;

static void hpush(HeapItem *restrict h, int *restrict hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) / 2; if (h[p].d <= h[i].d) break; HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HeapItem hpop(HeapItem *restrict h, int *restrict hs) {
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

static void dijkstra_heap(int n, int m, const int *restrict src, const int *restrict dst,
                           const double *restrict weight, int source, double *restrict dist_out)
{
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) { int u = src[i]; int pos = off[u] + fill[u]++; edst[pos] = dst[i]; ew[pos] = weight[i]; }
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0;
    char *done = calloc((size_t)n, 1);
    HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0;
    hpush(heap, &hs, 0, source);
    while (hs > 0) {
        HeapItem top = hpop(heap, &hs);
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = dist_out[u] + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }
    free(deg); free(off); free(edst); free(ew); free(fill); free(done); free(heap);
}

/* ---------------- dense/small path: the one-legged milker ----------------
   Source shrine's cow tipped full first (dist[source]=0), every other bowl
   left dry (INFINITY) -- SEED 1. Corridors become a dense bowl-to-bowl
   matrix, one tally-cord length per pair, keeping only the shortest cord
   when several run the same way -- SEED 3. The milker repeatedly scans
   every not-yet-sealed bowl (no priority structure at all -- SEED 2),
   hops to the driest bowl holding the least milk, burns its cow (seals it
   forever), then pours along every corridor out of that room, overwriting
   (never keeping two claims on) a farther bowl only when the new amount
   beats what is already there -- and only after the source room is sealed,
   per SEED 3. */
static void dijkstra_dense(int n, int m, const int *restrict src, const int *restrict dst,
                            const double *restrict weight, int source, double *restrict dist_out)
{
    double *mat = malloc((size_t)n * (size_t)n * sizeof(double));

#ifdef _OPENMP
    if (n > 2000) {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)n * n; i++) mat[i] = INFINITY;
    } else
#endif
    {
        for (long i = 0; i < (long)n * n; i++) mat[i] = INFINITY;
    }

    for (int i = 0; i < m; i++) {
        double w = weight[i];
        double *cell = &mat[(size_t)src[i] * n + dst[i]];
        if (w < *cell) *cell = w; /* one cord per pair: keep the shortest */
    }

    char *sealed = calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0; /* the source shrine's cow, tipped full first */

    for (int iter = 0; iter < n; iter++) {
        /* the milker: hop to the driest-not-yet-sealed bowl with least milk */
        int best = -1;
        double bestd = INFINITY;
        for (int v = 0; v < n; v++) {
            if (!sealed[v] && dist_out[v] < bestd) { bestd = dist_out[v]; best = v; }
        }
        if (best < 0) break; /* every remaining bowl is dry forever: unreachable */
        sealed[best] = 1; /* burn the cow: seal the room's distance forever */

        /* walk out along every corridor from the newly sealed room */
        const double *restrict row = &mat[(size_t)best * n];
        double db = dist_out[best];
        #pragma omp simd
        for (int v = 0; v < n; v++) {
            double nd = db + row[v];
            if (!sealed[v] && nd < dist_out[v]) dist_out[v] = nd;
        }
    }
    free(mat); free(sealed);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    /* Regime detection, encoded literally in the world as: is the palace
       small enough, or richly-corridored enough, that the milker's flat
       walk beats carrying a priority shrine (heap) to every pour? */
    const int N_SMALL = 1500;       /* small palace: flat scan wins even if sparse */
    const int N_MAX_MATRIX = 8000;  /* memory cap: n*n doubles must stay <= ~512MB */
    double density = (double)m / ((double)n * (double)n + 1.0);
    const double DENSE_FRAC = 0.08; /* >=8% of all possible corridors: call it dense */

    int use_dense = (n <= N_MAX_MATRIX) && (n <= N_SMALL || density >= DENSE_FRAC);

    if (use_dense) {
        dijkstra_dense(n, m, src, dst, weight, source, dist_out);
    } else {
        dijkstra_heap(n, m, src, dst, weight, source, dist_out);
    }
}
