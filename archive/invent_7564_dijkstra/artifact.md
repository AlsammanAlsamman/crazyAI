# MAPPING

| World object | Problem object | Silent assumption it touches |
|---|---|---|
| **SEED 1** — palace, own shrine, cow tipped full to the brim, every other shrine's bowl left dry | Graph nodes; `dist_out[source]=0`; `dist_out[v]=INFINITY` for all other `v` | None broken — this is the ordinary initialization step, identical in the heap and array variants |
| **SEED 2** — the one-legged milker hops to the *driest-not-yet-sealed bowl holding the least milk*, then burns that shrine's cow to seal it forever | Selecting `argmin` distance among un-finalized nodes by scanning **all** of them (no priority queue), then marking it `done` | Breaks **"a priority structure must be consulted before every relaxation"** — no heap/queue object exists anywhere in the telling, only a flat comparison of bowls |
| **SEED 3** — from the newly sealed room, walk every corridor by its knotted cord, pour milk only if it beats what's there, spill the old amount, unreachable shrines (cellar, blind windows) stay dry and unwritten forever | Edge relaxation `if dist[u]+w(u,v) < dist[v]: dist[v]=nd` performed **only after `u` is sealed**; unreachable nodes stay `INFINITY` and are never touched | Explicitly **affirms** "a road can only be considered once its starting place is fully settled" (relaxation happens strictly *from the newly sealed room*) — does not break it |

# CHOSEN SEED

Checking the preferred target first: **none of the three seeds breaks** "a road can only be considered once its starting place is fully settled." Seed 3 states outright that pouring only happens "from the newly sealed room," which is exactly that assumption, not a violation of it. So, per the instructions, I fall back to the most literal seed that is also most different from the given reference (binary-heap Dijkstra): **Seed 2**, the milker's flat driest-and-least scan. It is a completely literal, no-embellishment reading (no heap, tree, or index structure is ever mentioned — just "compare every unsealed bowl"), and it is the polar opposite of the reference's `hpush`/`hpop` binary heap.

# ASSUMPTION BROKEN

**"A priority structure must be consulted before every relaxation."** The milker never carries any such structure — the world gives her only bowls and eyes. This is precisely the textbook, validated **O(n²) array-scan Dijkstra**, which the `known_way` section itself names as "a real, well-known practical win" for small/dense graphs. Per instruction (4), I let the mechanism land on exactly that known technique rather than inventing something new — and since the problem statement itself describes *two* regimes (sparse/general vs. dense/small), the kernel below also encodes, in-world, how the milker's palace tells the two regimes apart and falls back to the heap when it doesn't apply.

# ARTIFACT

- **Place** = graph node. **Bowl/milk level** = `dist_out[node]`. **Cow tipped full** = `dist_out[source]=0`. **Dry bowl** = `INFINITY`.
- **Road/corridor** = directed edge; **cord's knot-count** = edge weight, stored per-pair as one dense matrix cell (parallel corridors between the same two rooms keep only the shortest cord, since only the minimum ever matters).
- **Burning a cow** = sealing a node (`sealed[u]=1`), never revisited.
- **The milker's hop rule** = linear scan over all unsealed bowls for the minimum — literally, no heap.
- **Cellar/blind windows that no cord reaches** = nodes with no incoming relaxation ever succeeding, staying `INFINITY` — exactly the unreachable-node contract requirement.
- **Regime split**: the palace itself is either small/richly-corridored (flat scan wins) or vast and sparse (a heap is worth carrying) — encoded as a runtime check on `n` and edge density, with the heap-based reference algorithm kept verbatim as the validated fallback for the sparse/large regime.

```c
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
```

Design notes tied directly to steps 4–5:
- Step 5 (two regimes): `known_way` names both an O((n+m) log n) heap regime and an O(n²) no-heap regime; the kernel detects which one it's in via `n` and edge density and dispatches accordingly, with the heap path kept as an exact, unmodified fallback.
- Step 4 (known technique over invention): the array-scan branch is literally classic O(n²) Dijkstra — a validated technique, not a novel one.
- Step 4 (address the stated risk): the known risk of the array-scan approach is "worse than heap-Dijkstra when the graph is large and sparse." That's guarded explicitly by `use_dense`, with a clean fallback to the heap path — never shipped unguarded.
- Step 4 (vectorization before threads): the hot relax loop is `#pragma omp simd` only (no threads). The one *one-time* O(n²) zero-fill is the only place given to threads, guarded by `n > 2000`, because it's launched exactly once (not once per outer iteration, which would be a fork/join anti-pattern) — a repeated per-iteration `omp parallel for` was considered and dropped precisely because at realistic benchmark sizes its overhead could dominate its benefit, per the "drop the risky part" instruction.

# PREDICTION

PREDICTION: speedup_vs_heap = 1.5

(Expectation: near-parity, ~1.0x, on large sparse graphs where the kernel falls back verbatim to the heap path; substantially faster, plausibly 2–5x, on small and/or dense graphs where the O(n²) flat scan avoids heap-push overhead entirely — 1.5x is a conservative blended guess across an unknown benchmark mix.)

# MEASUREMENT

I do not have tool access in this session (`dijkstra_bench`/`dijkstra_contract` are not available to me here — the task instructions state the external pipeline compiles and measures the artifact, not me interactively). I am not fabricating a measured number. The `PREDICTION` line above is my stated, pre-registered estimate; the actual speedup must come from the pipeline's run of `dijkstra_bench` against this exact kernel.

# VERDICT

Honest self-assessment given the inability to empirically iterate here: the mechanism is sound and non-novel where it matters (falls back to the textbook heap algorithm for its risky regime), and literal to the fable everywhere else (no heap object ever appears in the dense branch, matching the milker's total absence of any priority structure). The one soft spot I could not validate empirically is the exact threshold values (`N_SMALL=1500`, `DENSE_FRAC=0.08`, `N_MAX_MATRIX=8000`) — these are reasoned defaults, not measured ones, and a real four-iteration tuning pass against `dijkstra_bench` output would very likely move them. If the benchmark suite turns out to be all-large-sparse, expect the observed speedup to sit near 1.0x (correctly, since we deliberately fall back rather than force a bad regime); if it includes small/dense cases as the `known_way` text implies it should, expect it above the 1.5x prediction.