# MAPPING

## SEED 1 — "A glint released from the traveler's jewel crawls down every cord at a pace set by the cord's own length."

| world object | problem object |
|---|---|
| jewel | node `0..n-1` |
| infinite net / threads | the graph's incidence structure (CSR) |
| blood-dark cord, jewel→jewel | directed edge `src[i] -> dst[i]` |
| length of the cord (short/taut, long/slack) | `weight[i]` |
| touching the traveler's jewel | `dist[source] = 0` |
| glint | a *dated arrival event*: the pair (node, arrival time) |
| "moving no faster than the cord is long" | arrival time = departure time + `weight`; the glint is **deposited in the future**, not chased |
| glint enters *every* cord at the instant of the flare | when a node flares, **all** its out-edges are consumed in one pass, unconditionally |
| "a short cord flashes at once, a long one keeps the glint waiting **on the way**" | the delay lives **in the cord**, i.e. work is parked in a time-indexed structure, not re-derived by comparison |
| "I do not chase the glint; I crouch and I wait" | a monotone clock is advanced; nothing is ever searched for |

**Breaks:** *"a priority structure must be consulted before every relaxation"* and *"the next place to finalize is found by comparing against every remaining place."* There is no comparison at all: arrival order is produced by the physics (the cord length) rather than recovered by a sort.

## SEED 2 — "The first flare to reach each jewel is chalked once and every later flare along a longer cord is thrown away unmarked."

| world object | problem object |
|---|---|
| chalk grid, one square per jewel | `dist_out[0..n-1]` |
| white dust number in a square | the tentative/settled distance |
| "a jewel already reflects every other jewel touching it and needs only its first true flare" | each node re-emits exactly once per *improvement*; no `done[]`-gated settling ceremony |
| "whatever flare reaches a jewel afterward … I throw away without marking twice" | lazy discard: `if (nd < dist[v])` on arrival, plus `if (du < proc[u])` on pop — a late glint is dropped, never re-queued, never re-sorted |
| "wasted color no dyer's cloth would keep" | stale queue entries are *skipped*, not deleted (no decrease-key, no sift) |

**Breaks:** *"each place's distance must be finalized before its neighbors are explored."* The chalked number is authoritative-on-arrival; correctness comes from the *time order of arrivals*, not from a finalization step.

## SEED 3 — "A jewel's square that never catches any flare is left blank and crossed out like a wind-erased desert track."

| world object | problem object |
|---|---|
| blank square, crossed with a scratch | `dist_out[v] = INFINITY` |
| "however long I crouch" | the clock stops when no glint is in flight (`pending == 0`), **not** after `n` extractions |
| "I will not send the traveler chasing wind" | unreachable nodes cost zero work — they are never touched, never scanned, never popped |

**Breaks:** *"the whole graph must be explored to know any single distance"* — the crouch terminates on an empty sky, touching only the reachable component.

# CHOSEN SEED

**SEED 1.**

Honest statement first, as required: **none of the three seeds cleanly breaks "a road can only be considered once its starting place is fully settled."** In the native's world, physics makes first-flare *identical* to settled — a jewel cannot glow before its true arrival time — so the assumption is never violated, only made vacuous. SEED 1 comes closest to touching it (the glint is committed into the cord at the instant of the flare, and the cord holds it in flight while nothing at either end is being verified; the decision of whether the arrival *matters* is taken at the far end, at arrival time), so I fall back to it as both the most literal and the most different from the heap.

SEED 1 is the mechanism; SEEDs 2 and 3 are parts of the same machine (the chalk grid's write-once-per-improvement rule, and the blank-and-crossed unreachable squares) and are implemented inside it.

# ASSUMPTION BROKEN

Primary: **"a priority structure must be consulted before every relaxation"** — and with it, **"the next place to finalize is found by comparing against every remaining place."**

The native never compares two glints. He lays a *row of chalk squares in time* — one per hand-span of waiting — drops each released glint into the square matching the instant it will arrive, and then advances his attention one square at a time. Ordering is an **address computation** (`floor(arrival / Δ)`), not a search. Secondary (SEED 2): finalization is replaced by *first-arrival-wins* with late glints discarded; secondary (SEED 3): the sky emptying, not `n` extractions, is the stopping rule.

Computationally this is a **circular time-wheel with drain-to-fixpoint buckets** (the Dial/Δ-stepping family), reached from the metaphor rather than from the textbook.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================================================================== *
 *  "the old road, walked jewel by jewel"                               *
 *  Kept ONLY as a guarded fallback for the regime the chalk row        *
 *  cannot hold (see wheel_ok / the patience budget below).             *
 * ==================================================================== */
typedef struct { double d; int u; } HItem;

static void hp_push(HItem *restrict h, int *restrict hs, double d, int u)
{
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d;
    h[i].u = u;
}

static void heap_dijkstra(int n, int m,
                          const int *restrict off, const int *restrict edst,
                          const double *restrict ew, int source,
                          double *restrict dist)
{
    HItem *h    = (HItem *)malloc((size_t)(m + 2) * sizeof(HItem));
    char  *done = (char  *)calloc((size_t)n, 1);
    int hs = 0;
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    if (!h || !done) { free(h); free(done); return; }
    hp_push(h, &hs, 0.0, source);
    while (hs > 0) {
        HItem top = h[0];
        int last = --hs;
        if (last > 0) {
            HItem mv = h[last];
            int i = 0;
            for (;;) {
                int l = 2 * i + 1;
                if (l >= last) break;
                int r = l + 1, s = l;
                if (r < last && h[r].d < h[l].d) s = r;
                if (h[s].d >= mv.d) break;
                h[i] = h[s];
                i = s;
            }
            h[i] = mv;
        }
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        double du = dist[u];
        int s0 = off[u], t0 = off[u + 1];
        for (int e = s0; e < t0; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; hp_push(h, &hs, nd, v); }
        }
    }
    free(h);
    free(done);
}

/* ==================================================================== *
 *  "if the grid is small enough to take in with one glance"             *
 *  Dense / tiny regime: no chalk row at all, sweep the whole grid.      *
 *  Contiguous double array + sentinel => auto-vectorizable min scan.    *
 * ==================================================================== */
static void glance_scan(int n,
                        const int *restrict off, const int *restrict edst,
                        const double *restrict ew, int source,
                        double *restrict dist)
{
    double *key = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    if (!key) { heap_dijkstra(n, off[n], off, edst, ew, source, dist); return; }
    for (int i = 0; i < n; i++) key[i] = INFINITY;
    key[source] = 0.0;
    for (int it = 0; it < n; it++) {
        double best = INFINITY;
        int u = -1;
        for (int i = 0; i < n; i++) {           /* one glance across the grid */
            double kv = key[i];
            if (kv < best) { best = kv; u = i; }
        }
        if (u < 0) break;                        /* sky empty: rest stays blank */
        key[u] = INFINITY;
        double du = dist[u];
        int s0 = off[u], t0 = off[u + 1];
        for (int e = s0; e < t0; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; key[v] = nd; }
        }
    }
    free(key);
}

/* ==================================================================== *
 *  THE NATIVE'S MECHANISM: glints in flight, a circular chalk row.      *
 *  Returns 1 if the crouch completed, 0 if patience ran out.           *
 * ==================================================================== */
static int wheel_run(int n, int m,
                     const int *restrict off, const int *restrict edst,
                     const double *restrict ew,
                     double wmax, double delta, int B,
                     int source, double *restrict dist)
{
    int    *bhead = (int    *)malloc((size_t)B * sizeof(int));     /* chalk row  */
    double *proc  = (double *)malloc((size_t)n * sizeof(double));  /* flared-at  */
    int ecap = m + 64;
    int *enode = (int *)malloc((size_t)ecap * sizeof(int));        /* glints     */
    int *enext = (int *)malloc((size_t)ecap * sizeof(int));
    if (!bhead || !proc || !enode || !enext) {
        free(bhead); free(proc); free(enode); free(enext);
        return 0;
    }
    for (int i = 0; i < B; i++) bhead[i] = -1;
    for (int i = 0; i < n; i++) proc[i] = INFINITY;

    const int  Bm   = B - 1;                 /* B is a power of two */
    const double dinv = 1.0 / delta;
    long pending = 1, k = 0, scans = 0;
    long budget  = 8L * (long)m + 32L * (long)n + 4096L;   /* patience */
    int  ecnt = 1, aborted = 0;

    enode[0] = source; enext[0] = -1; bhead[0] = 0;   /* touch the one jewel */

    while (pending > 0) {                 /* "I crouch, and I wait"          */
        int slot = (int)(k & (long)Bm);    /* the square the clock stands on  */
        int e = bhead[slot];
        while (e >= 0) {                  /* drain this instant to fixpoint  */
            bhead[slot] = enext[e];
            pending--;
            int u = enode[e];
            double du = dist[u];
            if (du < proc[u]) {           /* SEED 2: a late flare is dropped */
                proc[u] = du;
                int s0 = off[u], t0 = off[u + 1];
                scans += (long)(t0 - s0);
                for (int j = s0; j < t0; j++) {
                    if (j + 8 < t0) __builtin_prefetch(&dist[edst[j + 8]], 1, 1);
                    int v = edst[j];
                    double nd = du + ew[j];          /* arrival = flare + cord */
                    if (nd < dist[v]) {              /* first light only       */
                        dist[v] = nd;
                        long kb = (long)(nd * dinv); /* which future square    */
                        if (kb < k) kb = k;
                        int sl = (int)(kb & (long)Bm);
                        if (ecnt >= ecap) {
                            int nc = ecap + (ecap >> 1) + 64;
                            int *a = (int *)realloc(enode, (size_t)nc * sizeof(int));
                            if (a) enode = a;
                            int *b = (int *)realloc(enext, (size_t)nc * sizeof(int));
                            if (b) enext = b;
                            if (!a || !b) { aborted = 1; goto finish; }
                            ecap = nc;
                        }
                        enode[ecnt] = v;
                        enext[ecnt] = bhead[sl];
                        bhead[sl] = ecnt;            /* park the glint in time */
                        ecnt++;
                        pending++;
                    }
                }
                if (scans > budget) { aborted = 1; goto finish; }
            }
            e = bhead[slot];              /* glints landing on *this* instant */
        }
        k++;                              /* the clock moves one square on    */
    }
finish:
    free(bhead); free(proc); free(enode); free(enext);
    return aborted ? 0 : 1;
}

/* ==================================================================== */
void kernel(int n, int m, const int *restrict src, const int *restrict dst,
            const double *restrict weight, int source, double *restrict dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;   /* SEED 3: all blank */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- thread the net: CSR ---- */
    int    *off  = (int    *)calloc((size_t)n + 1, sizeof(int));
    int    *edst = (int    *)malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    int    *cur  = (int    *)malloc((size_t)n * sizeof(int));
    if (!off || !edst || !ew || !cur) {
        free(off); free(edst); free(ew); free(cur);
        return;
    }
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)n * sizeof(int));

    double wmax = 0.0, wsum = 0.0;
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int p = cur[u]++;
        double w = weight[i];
        edst[p] = dst[i];
        ew[p]   = w;
        wsum += w;
        if (w > wmax) wmax = w;
    }
    free(cur);

    /* ---- regime 1: a grid small enough for one glance ---- */
    if (n <= 512) {
        glance_scan(n, off, edst, ew, source, dist_out);
        free(off); free(edst); free(ew);
        return;
    }

    /* ---- choose the spacing of the chalk row ---- */
    double davg = (double)m / (double)n;
    if (davg < 1.0) davg = 1.0;
    double wmean = wsum / (double)m;
    double delta = (wmax > 0.0) ? (wmax / (2.0 * davg)) : 1.0;
    if (!(delta > 0.0) || !isfinite(delta)) delta = 1.0;

    long nbneed = (long)(wmax / delta) + 3;
    int  B = 8;
    while ((long)B < nbneed && B < (1 << 18)) B <<= 1;

    /* ---- regime 2 test: can the row hold these cords? ----
       If the longest cord dwarfs its kin, one chalk square would span many
       typical cords, the drains degenerate, and the old road is better.    */
    int wheel_ok = ((long)B >= nbneed) && (wmax <= 4.0 * davg * wmean);

    if (!wheel_ok || !wheel_run(n, m, off, edst, ew, wmax, delta, B,
                                source, dist_out))
        heap_dijkstra(n, m, off, edst, ew, source, dist_out);

    free(off); free(edst); free(ew);
}
```

**Which code is which part of the native's mechanism**

| native's words | code |
|---|---|
| jewel hung in the net | node index; `off/edst/ew` CSR |
| cord, its length | `edst[j]`, `ew[j]` |
| touching the traveler's jewel | `enode[0]=source; bhead[0]=0; dist[source]=0` |
| glint released down **every** cord at the flare | the unconditional `for (j=s0;j<t0;j++)` loop — all out-cords consumed in one pass |
| "moving no faster than the cord is long" | `nd = du + ew[j]`; `kb = (long)(nd*dinv)` — the glint is *parked in the future square* |
| the cord keeping the glint waiting on the way | `enode/enext` arena + `bhead[sl]`: the glint physically sits in the row until the clock gets there |
| "I do not chase the glint; I crouch and wait" | `k++` — a monotone clock, **zero comparisons**, no heap, no scan |
| the chalk grid, one square per jewel | `dist_out[]` |
| marking the instant a jewel first catches fire | `dist[v] = nd` |
| late flare on a longer cord, thrown away unmarked | `if (nd < dist[v])` and `if (du < proc[u])` — SEED 2, lazy discard instead of decrease-key |
| a jewel needs only its first true flare | `proc[u]` records the instant it flared; never re-scanned at that instant |
| squares left blank and crossed out | `INFINITY` init, never written — SEED 3 |
| "however long I crouch" / stopping | `while (pending > 0)` — the sky emptying, not `n` extractions |
| "a grid small enough to take in with one glance" | `glance_scan` (dense/small regime, the O(n²) sweep) |
| "if the cords are ill-matched to my row" | `wheel_ok` test |
| "past all patience → walk the old road" | `budget` abort → `heap_dijkstra` |

Correctness note (why the metaphor is exact, not approximate): any glint created while the clock stands on square `k` has arrival `≥ k·Δ`, so it can only land on square `k` or later — never on a square already passed. Squares are drained **to fixpoint** (glints landing on the current square are picked up by the same `while (e >= 0)` loop), so when the clock leaves square `k`, every node whose chalked number lies in `[kΔ,(k+1)Δ)` is final. `B ≥ ⌊w_max/Δ⌋+3` (power of two) guarantees the live window `[k, k+B-2]` never aliases in the circular row. Zero-weight cords are handled by the same fixpoint drain. No quantization ever touches a distance value — `Δ` only chooses a *slot*, so the arithmetic is bit-identical to the reference along the chosen path.

Step-4 compliance: every risk my own verdict names is guarded — small/dense → `n <= 512` glance path; pathological weight spread → `wheel_ok` up-front test **plus** a `scans > budget` backstop, both falling back to the plain heap. Vectorization/cache hints only (`restrict`, CSR contiguity, sentinel-array min scan, prefetch); **no thread parallelism**, because the metaphor's unit of work is a single glint arrival on a monotone clock — too small and too order-dependent to be worth a barrier at these sizes.

# PREDICTION

PREDICTION: speedup_vs_heap = 2.2

Reasoning behind the number, stated before any measurement: the heap path pays `O(log n)` random-access sift work per push/pop with cache-missing `HItem` traffic (~16 B/entry, heap depth ~20 at n=10⁶); the chalk row pays one integer multiply-and-truncate plus two stores per glint, with `bhead` at `B≈32` living permanently in L1. Against that gain I charge an expected 1.2–1.5× redundant edge scans from within-square re-flaring at `Δ = w_max/(2·m/n)`, plus one extra `double[n]` (`proc`) touched randomly. Net ≈ 2–2.5×. I expect the *largest* speedups on sparse graphs with homogeneous weights and the *smallest* (possibly < 1.3×) on graphs whose shortest-path tree is deep relative to `w_max` (long chains/grids), where empty-square sweeping adds ~`n` cheap iterations.

# MEASUREMENT

**Not obtained.** `dijkstra_bench` and `dijkstra_contract` were listed as available but no tools are actually callable in this session, so I ran zero measurements and performed zero measured improvement rounds (0 of the 4 permitted). I will not dress design reasoning up as data: the speedup above is a prediction only, and the four analytical design choices I made (Δ = w_max/(2·davg); power-of-two circular row with `B ≥ ⌊w_max/Δ⌋+3`; `proc[]` re-flare suppression instead of decrease-key; the two regime guards) are **unvalidated**.

Concretely, what the pipeline should check:
1. Exactness vs. the reference on all cases, including unreachable nodes (`INFINITY`), zero-weight edges, `source` with no out-edges, and multi-edges.
2. `speedup_vs_heap` on each size/density. If it lands below ~1.3× on sparse-deep graphs, the correct next move **within this mechanism** is to coarsen `Δ` (raise the `2.0*davg` divisor toward `1.0*davg`) to cut empty-square sweeping — not to swap in a heap.
3. That the `wheel_ok` test and the `budget` abort actually fire on a heavy-tailed weight distribution (one cord orders of magnitude longer than its kin) and that the fallback result is still exact.

# VERDICT

The native's description is not a picture of Dijkstra with the search step hidden — it is a picture of a **physical wavefront simulated on a time-indexed row of chalk**, and taken literally it yields a genuinely different machine: ordering by address computation instead of by comparison, first-arrival chalking instead of settle-then-expand, and termination on an empty sky instead of on `n` extractions. It lands, independently, in the Dial/Δ-stepping family — which is a real and well-documented practical win over binary-heap Dijkstra, so I expect this to be faster, not merely exotic.

Two honest weaknesses, both named and both guarded rather than hand-waved:

- **Heterogeneous cord lengths.** When `w_max ≫ davg · w_mean`, one chalk square spans many typical cords, the fixpoint drains degenerate toward Bellman-Ford, and the mechanism loses. Guarded twice: the `wheel_ok` pre-test (no wasted work) and the `scans > budget` backstop (catches what the pre-test misses), both falling back to the heap.
- **Deep shortest-path trees.** Empty-square sweeping costs `D_max/Δ` clock ticks, ~`n` cheap iterations for a long chain. Bounded by tying `Δ` to `w_max/davg` rather than to `w_min`; not eliminated, and it is where I expect the prediction to be weakest.

The third weakness I cannot guard away and will state flatly: I have not measured anything. If the measured speedup comes back below 1.0, the prediction was simply wrong, and the fix belongs inside this mechanism — the spacing of the chalk row — not in replacing it with the textbook method.