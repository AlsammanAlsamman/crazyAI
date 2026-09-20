# MAPPING (per SEED)

**SEED 1 — "The nightingales always land first on the stone carrying the smallest owed letter."**

| World object | Problem object |
|---|---|
| Garden of stones | the `n` nodes |
| Casting tower, "already here" house | source node |
| "nothing owed" stone under the tower | `dist_out[source] = 0` |
| unreadable far number (chalk) | `dist_out[i] = INFINITY` (unvisited) |
| nightingales circling *all* unlocked stones, landing on the smallest owed sum | a full linear scan over every still-unlocked distance to find the current minimum — extraction done by comparison, not by any maintained structure |
| stone's letter locked when a nightingale lands | node finalized, never touched again |

Breaks **A2** — "a priority structure must be consulted before every relaxation." Nothing is maintained between relaxations; the whole unlocked set is freshly compared only when a new minimum is *needed* (once per lock), not pushed/sifted on every edge relaxation. This is literally array-scan Dijkstra.

**SEED 2 — "Threads cast from the tower price each road by what its thief is owed, one direction at a time."**

| World object | Problem object |
|---|---|
| thread cast from the newly locked house to a neighbor | directed edge `u->v` |
| thief guarding the road, asked what he's owed, directionally | `weight[edge]`, asymmetric (`src[i]->dst[i]`, not the reverse) |
| "rebuild that letter... stolen from its old amount, replaced" | `if (dist[u]+w < dist[v]) dist[v] = dist[u]+w` |

Does **not** break anything — it *confirms* **A4** ("a road can only be considered once its starting place is fully settled"): threads are explicitly cast *from* the just-locked house.

**SEED 3 — "A locked garden letter is never rebuilt again, and threads that reach nowhere are simply dropped and thrown away."**

| World object | Problem object |
|---|---|
| locked letter, never rebuilt | once-settled distance is immutable (relies on non-negative weights: "no thief anywhere charges a negative toll") |
| threads into bramble/tide/nowhere, dropped | relaxations that don't improve a distance are simply skipped |
| stones never reached, chalked forever | unreachable nodes stay `INFINITY` |

Does **not** break anything — it's the correctness argument *for* **A1/A4** (why it's safe to never revisit a locked node), not a violation of it.

# CHOSEN SEED

None of the three seeds breaks **A1** ("each place's distance must be finalized before its neighbors are explored") — the narrative explicitly preserves settle-then-cast order throughout ("From that newly locked house I cast threads again"). Stating that plainly, per the instructions I fall back to the most literal seed: **SEED 1**, the nightingale scan — it is also the one most mechanically different from the known heap-based way, and it happens to break **A2**.

# ASSUMPTION BROKEN

**A2 — "a priority structure must be consulted before every relaxation."** Replaced by: no persistent structure at all; a fresh O(n) linear scan over the unlocked stones finds the minimum, once per lock event. This is exactly the well-known, validated O(n²) array-scan Dijkstra — per instruction step 4, I let the mechanism land on this known technique rather than inventing something novel.

The story's own stated risk (n² blowing up) is real for large sparse graphs, so per step 4 I guard it: compute both cost estimates and dispatch to the classic heap method when the graph is large/sparse.

# ARTIFACT

```c
#include <stdlib.h>
#include <math.h>

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

/* Garden-nightingale array-scan Dijkstra: O(n^2 + m).
   Literal mapping:
     stone       = node
     letter      = tentative distance, mirrored in key[] which is set to
                   +INF the instant a stone is locked (so the nightingale
                   only ever "circles the unlocked stones")
     thread      = directed edge out of the most-recently-locked stone
     thief price = weight[] of that edge, asked one direction at a time
     lock        = key[u] = INFINITY, dist_out[u] final, never touched again
                   (no visited[] array needed: non-negative weights mean a
                   locked letter can never be beaten later - "no thief
                   charges a negative toll")
     dropped threads = relaxations that fail the improvement test; skipped,
                   exactly like unreachable stones staying +INF forever
*/
static void dijkstra_array_scan(int n, const int *restrict off,
                                 const int *restrict edst,
                                 const double *restrict ew,
                                 int source, double *restrict dist_out,
                                 double *restrict key, int use_omp) {
    for (int i = 0; i < n; i++) { dist_out[i] = INFINITY; key[i] = INFINITY; }
    dist_out[source] = 0.0;
    key[source] = 0.0;

    for (int iter = 0; iter < n; iter++) {
        double best = INFINITY;
        if (use_omp) {
            #pragma omp parallel for reduction(min:best)
            for (int i = 0; i < n; i++) if (key[i] < best) best = key[i];
        } else {
            for (int i = 0; i < n; i++) if (key[i] < best) best = key[i];
        }
        if (best == INFINITY) break; /* rest is bramble/tide: unreachable */

        int u = -1;
        for (int i = 0; i < n; i++) { if (key[i] == best) { u = i; break; } }
        if (u < 0) break;

        key[u] = INFINITY; /* locked: rebuilt one last time, never again */

        double du = dist_out[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; key[v] = nd; }
            /* else: thread dropped and thrown away */
        }
    }
}

static void dijkstra_heap(int n, const int *restrict off,
                           const int *restrict edst, const double *restrict ew,
                           int m, int source, double *restrict dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
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
            double nd = dist_out[u] + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }
    free(done); free(heap);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;

    /* CSR build, shared by both paths (a road is only ever cast from the
       stone that owns it, one direction at a time). */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }

    /* Guard against the array-scan's own stated weakness (O(n^2) blows up
       on large sparse graphs): estimate both costs, pick the cheaper,
       falling back to the well-known binary-heap method when large+sparse. */
    double logn = log2((double)n + 2.0);
    double cost_heap  = ((double)n + (double)m) * logn;
    double cost_array = (double)n * (double)n + (double)m;

    if (cost_array <= cost_heap) {
        double *key = malloc((size_t)n * sizeof(double));
        /* Vectorization first: the scan is split so the min-value pass is a
           clean SIMD reduction. Thread parallelism only added, and only for
           that reduction, when n is large enough per scan (n>=20000) for it
           to pay for itself; serial fallback otherwise (guarded, per spec). */
        int use_omp = (n >= 20000);
        dijkstra_array_scan(n, off, edst, ew, source, dist_out, key, use_omp);
        free(key);
    } else {
        dijkstra_heap(n, off, edst, ew, m, source, dist_out);
    }

    free(deg); free(off); free(edst); free(ew); free(fill);
}
```

Improvement history (reasoned, since no measurement loop was runnable here):
1. Naive always-array-scan version — correct but risks O(n²) blowup on large sparse graphs (the story's own risk).
2. Added the `cost_array` vs `cost_heap` dispatcher, falling back to the standard heap method — addresses that risk directly rather than ignoring it.
3. Split "find minimum" into a clean value-only reduction pass (vectorizable) instead of one branchy value+index loop.
4. Added an OpenMP `reduction(min:best)` on that pass, but only when `n>=20000` (each of the n scans is itself big), with an explicit serial fallback — vectorization before threads, threads guarded by a size check, per instructions.

PREDICTION: speedup_vs_heap = 3.0

# MEASUREMENT

Tools (`dijkstra_bench`, etc.) are declared unavailable in this session — I cannot actually execute the benchmark here, so I am not going to fabricate numbers. The prediction above is a complexity-based estimate: parity (~1.0×) expected on large/sparse cases (heap path taken, near-identical code to the reference), with large wins (plausibly 5–20×) on dense or small graphs where the array-scan path is chosen, since it avoids O(m log n) heap push/pop entirely in favor of a cache-friendly O(n²+m) scan. A geomean around 3× is my honest estimate for a mixed benchmark suite, pending the pipeline actually compiling and running it against `dijkstra_contract`.

# VERDICT

The mechanism breaks A2, not A1 (nothing in the native's story does) — stated plainly rather than forced. It arrives at a known, validated technique (plain array-scan Dijkstra) rather than inventing something new, as instructed. Its one named risk — O(n²) is worse than the heap on large sparse graphs — is addressed with a concrete guard (dynamic cost comparison, automatic fallback to the reference heap algorithm), not left as a caveat. Thread parallelism is added only for the one embarrassingly-parallel reduction, only above a size threshold, with a serial fallback — vectorization-first as instructed. Real speedup is unverified in this session pending an actual `dijkstra_bench` run.