# MAPPING

Every world-object is taken as naming one concrete computational object. The net is the graph; nothing in the description is decoration.

**SEED 1 — "A glint released from the traveler's jewel crawls down every cord at a pace set by the cord's own length."**

| world-object | problem-object |
|---|---|
| polished jewel, one per place | node `0..n-1` |
| the infinite net threading them | adjacency structure (CSR: `off[]`, `edst[]`, `ew[]`) |
| blood-dark cord, jewel→jewel | one directed edge `src[i] -> dst[i]` |
| length of the cord (taut/slack) | `weight[i]` = *delay* the glint suffers on that cord |
| the traveler's jewel, touched | `source` |
| the glint | one relaxation in flight, carrying an arrival time |
| "moving no faster than the cord is long" | arrival time at `v` = `dist[u] + w(u,v)` |
| "a short cord flashes at once" | *light* cord, `w < Δ`: lands in the **same** time-slice → fires inside the current tick |
| "a long one keeps the glint waiting on the way" | *heavy* cord, `w ≥ Δ`: the glint is **parked** in a later time-slice (bucket) and nobody touches it until the clock gets there |
| "a jewel already reflects every other jewel touching it and needs only its first true flare" | a node re-emits on **all** its out-cords the instant it is first lit — **no settling step at all** |
| "I do not chase the glint; I crouch and wait" | the algorithm never searches for the next node; the cord's own length *files* the glint into the tick where it belongs |

*Silent assumption broken:* **"a road can only be considered once its starting place is fully settled"** — and, as a bonus, **"a priority structure must be consulted before every relaxation"**: the arrival time is its own index, so there is no queue to consult.

**SEED 2 — "The first flare to reach each jewel is chalked once and every later flare along a longer cord is thrown away unmarked."**

| world-object | problem-object |
|---|---|
| chalk grid, one square per jewel | `dist_out[]` |
| the white-dust number in a square | current best arrival time |
| "the instant each square's jewel first catches fire" | earliest arrival = shortest distance |
| a flare arriving late along a slacker cord | a relaxation with `nd >= dist_out[v]` |
| "thrown away without marking twice" | `if (nd < dist_out[v])` fails → discard; and a stale parked glint for a jewel already lit is skipped when its tick comes up (lazy deletion) |
| "the first light is the only light that counts" | correct **only because the clock advances in order** — which is what makes the time-slices load-bearing rather than cosmetic |

*Silent assumption broken:* **"each place's distance must be finalized before its neighbors are explored"** and **"the next place to finalize is found by comparing against every remaining place."**

**SEED 3 — "A jewel's square that never catches any flare is left blank and crossed out like a wind-erased desert track."**

| world-object | problem-object |
|---|---|
| a square that stays dark "however long I crouch" | unreachable node |
| the crossing scratch | `INFINITY` |
| "however long I crouch" ending | termination when **no glint is in flight** (`pending == 0`), *not* when all `n` nodes have been settled |
| "I will not send the traveler chasing wind" | unreachable nodes cost zero work; their cords are never read |

*Silent assumption broken:* **"the whole graph must be explored to know any single distance."**

# CHOSEN SEED

**SEED 1.** It is the only one of the three that breaks the preferred assumption ("a road can only be considered once its starting place is fully settled"), and it is the most literal: the native says the jewel *already reflects* and needs only its *first* flare — a jewel re-emits on contact, never after being certified. It is also the furthest from the known way, because the known way's entire skeleton is *settle-then-expand under a priority structure*, and SEED 1 has neither a settling step nor a priority structure — only cords whose own lengths park glints in time.

# ASSUMPTION BROKEN

**"A road can only be considered once its starting place is fully settled."**

In the world, the instant light touches a jewel it leaves along every cord. There is no moment of certification. The consequence in code: a node's out-edges are relaxed from a *tentative* distance, and if a shorter flare arrives later **within the same time-slice**, the node is simply re-lit and re-emits. This is exactly the thing heap-Dijkstra forbids, and it is what removes the heap: the queue is replaced by a small cyclic ring of time-slices of width `Δ`, indexed by *arrival time itself* (`⌊d/Δ⌋ mod NB`). Push = append to an array. Pop = walk an array. `Δ = w_max/K` is chosen so that only ~`1/K` of cords are "short" (`w < Δ`), which keeps the in-slice re-lighting subgraph subcritical (mean in-slice out-degree `< 1`), so re-emission is rare and the total work stays `O(m)` with an `O(1)`-per-edge constant. (This is a literal wavefront simulation; it lands on the same territory as Δ-stepping / Dial buckets, reached from the metaphor rather than from the textbook.)

**Two regimes, recognized in-world (required by step 5).** The known way explicitly names two: sparse/large (heap) and dense/small (plain `O(n²)` scan). The native's own instrument encodes both, because the chalk grid is a *physical object of a certain size*:

> When the grid scratched in the pavement is small enough — or so thickly corded that nearly every jewel touches every other — the crouching child does not need the cords' waiting at all. She sweeps her eyes across the **whole grid in one glance** and takes the palest unlit square, over and over.

That glance is the `O(n²)` min-scan; the ticking ring is the sparse path. The runtime test is on the grid's own size and cord density: `n <= 8192 && n² <= 8m + 8192`.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================================================================
   THE NET, literally:
     jewel               -> node
     cord                -> directed edge (CSR)
     length of the cord  -> weight = the delay the glint suffers on it
     glint               -> one relaxation in flight, carrying an arrival time
     chalk square        -> dist_out[] slot
     chalked number      -> the FIRST (hence shortest) arrival time
     one chalk tick      -> a time-slice of width delta (a bucket)
     blank + scratch     -> INFINITY
   A jewel re-emits the instant it is first lit.  Nothing is ever
   "settled" first; there is no priority structure anywhere.
   ================================================================== */

/* ---------- REGIME A: the grid is small / almost fully corded ----------
   The child sweeps her eyes over the WHOLE grid and takes the palest
   unlit square.  No cords waiting, no ticks: a plain O(n^2) glance,
   whose inner sweep is a flat, cache-resident, unrolled min-reduction. */
static void sweep_whole_grid(int n,
                            const int *restrict off,
                            const int *restrict edst,
                            const double *restrict ew,
                            int source,
                            double *restrict dist)
{
    double *restrict cand = (double *)malloc((size_t)n * sizeof(double));
    if (!cand) return;
    for (int i = 0; i < n; i++) { dist[i] = INFINITY; cand[i] = INFINITY; }
    dist[source] = 0.0;
    cand[source] = 0.0;

    for (int it = 0; it < n; it++) {
        double b0 = INFINITY, b1 = INFINITY, b2 = INFINITY, b3 = INFINITY;
        int i = 0;
        for (; i + 4 <= n; i += 4) {                 /* one glance, 4 lanes */
            double x0 = cand[i], x1 = cand[i + 1], x2 = cand[i + 2], x3 = cand[i + 3];
            if (x0 < b0) b0 = x0;
            if (x1 < b1) b1 = x1;
            if (x2 < b2) b2 = x2;
            if (x3 < b3) b3 = x3;
        }
        for (; i < n; i++) { double x = cand[i]; if (x < b0) b0 = x; }
        if (b1 < b0) b0 = b1;
        if (b3 < b2) b2 = b3;
        if (b2 < b0) b0 = b2;
        if (!(b0 < INFINITY)) break;                 /* only dark squares left */

        int u = 0;
        while (cand[u] != b0) u++;                   /* the palest square */
        cand[u] = INFINITY;

        double du = dist[u];
        int e1 = off[u + 1];
        for (int e = off[u]; e < e1; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; cand[v] = nd; }
        }
    }
    free(cand);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (m < 0) m = 0;
    if (source < 0 || source >= n) {
        for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
        return;
    }

    /* ---- thread every cord into the net, and learn the cords' lengths ---- */
    size_t ma = (size_t)(m > 0 ? m : 1);
    int    *pos  = (int *)calloc((size_t)n + 1, sizeof(int));
    int    *off  = (int *)malloc(((size_t)n + 1) * sizeof(int));
    int    *edst = (int *)malloc(ma * sizeof(int));
    double *ew   = (double *)malloc(ma * sizeof(double));
    if (!pos || !off || !edst || !ew) {
        free(pos); free(off); free(edst); free(ew);
        for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
        if (source >= 0 && source < n) dist_out[source] = 0.0;
        return;
    }

    double wmax = 0.0, wsum = 0.0;
    for (int i = 0; i < m; i++) {
        pos[src[i]]++;
        double w = weight[i];
        wsum += w;
        if (w > wmax) wmax = w;
    }
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + pos[i];
    for (int i = 0; i < n; i++) pos[i] = off[i];
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int p = pos[u]++;
        edst[p] = dst[i];
        ew[p]   = weight[i];
    }

    /* ---- which regime is the chalk grid in?  (runtime, in-metaphor) ---- */
    {
        double dnn = (double)n * (double)n;
        if (n <= 8192 && dnn <= 8.0 * (double)m + 8192.0) {
            sweep_whole_grid(n, off, edst, ew, source, dist_out);
            free(pos); free(off); free(edst); free(ew);
            return;
        }
    }

    /* ---- REGIME B: the ticking ring of time-slices ----
       delta = how much time one chalk tick covers.  Chosen so only about
       1/K of the cords are "short" (fire inside the tick): that keeps the
       in-tick re-lighting subgraph subcritical, so re-emission is rare. */
    double avgdeg = (m > 0) ? (double)m / (double)n : 1.0;
    int K = (int)(2.0 * avgdeg + 0.5);
    if (K < 8)   K = 8;
    if (K > 128) K = 128;

    double delta;
    if (wmax > 0.0) {
        delta = wmax / (double)K;
        double wmean = wsum / (double)(m > 0 ? m : 1);
        /* skewed cords (one enormous slack cord) must not widen the tick */
        if (wmean > 0.0 && delta > 4.0 * wmean) delta = 4.0 * wmean;
    } else {
        delta = 1.0;                 /* every cord taut to nothing */
    }

    long long need = (wmax > 0.0) ? (long long)(wmax / delta) + 4 : 2;
    int NB = 4;
    while ((long long)NB < need && NB < (1 << 14)) NB <<= 1;
    if ((long long)NB < need) { NB = 1 << 14; delta = wmax / (double)(NB - 4); }
    int    mask = NB - 1;
    double invd = 1.0 / delta;

    int **bd  = (int **)calloc((size_t)NB, sizeof(int *));   /* tick contents */
    int  *bs  = (int  *)calloc((size_t)NB, sizeof(int));     /* tick size     */
    int  *bc  = (int  *)calloc((size_t)NB, sizeof(int));     /* tick capacity */
    int  *qs  = (int  *)malloc((size_t)n * sizeof(int));     /* tick a jewel waits in, -1 = none */
    int   fcap = 256;
    int  *front = (int *)malloc((size_t)fcap * sizeof(int));

    if (!bd || !bs || !bc || !qs || !front) goto cleanup_fail;

    {
        double *restrict dist = dist_out;
        for (int i = 0; i < n; i++) { dist[i] = INFINITY; qs[i] = -1; }
        dist[source] = 0.0;

        long long pending = 0;

/* park a glint for jewel `nd_node` in tick slot `sl` */
#define PARK(nd_node, sl)                                                      \
        do {                                                                   \
            int _s = (sl);                                                     \
            if (bs[_s] == bc[_s]) {                                            \
                int _nc = bc[_s] ? bc[_s] * 2 : 64;                            \
                int *_p = (int *)realloc(bd[_s], (size_t)_nc * sizeof(int));    \
                if (!_p) goto cleanup;                                         \
                bd[_s] = _p; bc[_s] = _nc;                                     \
            }                                                                  \
            bd[_s][bs[_s]++] = (nd_node);                                      \
            qs[(nd_node)] = _s;                                                \
            pending++;                                                         \
        } while (0)

        PARK(source, 0);

        long long tick = 0;
        while (pending > 0) {
            int c = (int)(tick & (long long)mask);
            if (bs[c] == 0) { tick++; continue; }    /* nothing arrives this instant */

            /* drain this tick until it is empty, INCLUDING jewels re-lit
               inside it by short cords -- a jewel never waits to be settled */
            while (bs[c] > 0) {
                int cnt = bs[c];
                if (cnt > fcap) {
                    int nf = cnt + cnt / 2;
                    int *p = (int *)realloc(front, (size_t)nf * sizeof(int));
                    if (!p) goto cleanup;
                    front = p; fcap = nf;
                }
                memcpy(front, bd[c], (size_t)cnt * sizeof(int));
                bs[c] = 0;
                pending -= cnt;

                for (int t = 0; t < cnt; t++) {
                    int u = front[t];
                    if (qs[u] != c) continue;        /* a late flare: thrown away */
                    qs[u] = -1;
                    double du = dist[u];
                    int e1 = off[u + 1];
                    for (int e = off[u]; e < e1; e++) {
                        int v = edst[e];
                        double nd = du + ew[e];
                        if (nd < dist[v]) {          /* first/better light only */
                            dist[v] = nd;
                            int s = (int)((long long)(nd * invd) & (long long)mask);
                            if (qs[v] != s) PARK(v, s);
                        }
                    }
                }
            }
            tick++;
        }
#undef PARK
    }

cleanup:
    for (int i = 0; i < NB; i++) free(bd[i]);
    free(bd); free(bs); free(bc); free(qs); free(front);
    free(pos); free(off); free(edst); free(ew);
    return;

cleanup_fail:
    if (bd) { for (int i = 0; i < NB; i++) free(bd[i]); free(bd); }
    free(bs); free(bc); free(qs); free(front);
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    free(pos); free(off); free(edst); free(ew);
}
```

Why this is correct, stated plainly: all weights are `>= 0`, so a glint parked from tick `b` lands at absolute index `>= b` (monotone clock); the ring holds `NB >= ⌈w_max/Δ⌉ + 4` slices, so no two live absolute ticks alias onto the same slot; within a tick, draining to fixed point handles re-lighting; `qs[]` suppresses duplicate parks and discards stale ones; and termination is on `pending == 0`, so unreachable jewels keep their `INFINITY` and cost nothing.

# PREDICTION

**PREDICTION: speedup_vs_heap = 1.8**

Reasoning stated before measuring, so it can be scored honestly. On a large sparse graph the irreducible cost is the `m` random `dist[v]` probes — the heap and the ring both pay that. What the ring removes is the heap traffic: `~n` pops at `O(log n)` with poor locality plus the sift-ups. I estimate that at 35–50% of heap-Dijkstra's runtime, giving ~1.5–2.0×; I take 1.8. The ring adds one extra random probe (`qs[v]`) per *improving* relaxation only, and improving relaxations are `~1–2n`, not `m`, so that is noise. On dense or small inputs the whole-grid glance path should be much better than 1.8× (heap pays `m` pushes where the glance pays a cache-resident SIMD sweep), possibly 3–10×; on a tiny graph it should be ~1× or a bit better, never worse, since it has no allocation of a ring at all.

# MEASUREMENT

**Not measured. No tools were available in this session** — `dijkstra_bench` and `dijkstra_contract` were listed but the session was launched with no tool access, so I could not compile, could not verify against the reference, and could not run a single timing. I am reporting that as the failure it is rather than inventing numbers.

Consequently the "improve it at most four times" loop ran **zero** times. What is above is the first, unmeasured version. The specific things I would have looked at first, in order, and what each would tell me:

1. **Exactness against the reference.** The one real correctness risk is the ring-width invariant under floating-point rounding of `⌊nd·invd⌋`. I gave it 4 slices of slack and a power-of-two `NB`, which I believe is airtight, but this is precisely the kind of claim that deserves a test rather than an argument. A mismatch would show up as a *too-large* distance on a few nodes.
2. **Whether `Δ` is right.** Instrument the total number of parks and the total ticks traversed. If parks ≫ successful relaxations, `Δ` is too wide (in-tick re-lighting has gone supercritical) and `K` should rise; if ticks traversed ≫ `n`, `Δ` is too narrow and `K` should fall. `K = clamp(2·avgdeg, 8, 128)` is a guess informed by the subcritical-branching argument, not a fitted value.
3. **The dense/sparse threshold `n² <= 8m + 8192`.** The coefficient 8 is a cycle-cost estimate (sequential SIMD min-sweep vs. random-access edge relaxation), not a measurement. It is the single most likely thing to be off, and it is off in a *bounded* way: both branches are correct, only the choice of the faster one suffers.

# VERDICT

The native's mechanism translates with no slippage at all, and that is the interesting part: "a jewel already reflects every other jewel touching it and needs only its first true flare" is a complete, literal instruction to delete the settling step, and "a long cord keeps the glint waiting on the way" is a complete, literal instruction for what replaces the heap — the cord's own length files the glint into a time-slice, so the queue *is* the clock. I did not substitute the textbook method; there is no heap and no global min-scan anywhere on the sparse path.

**Stated risks, and how each is discharged rather than left hanging** (step 4):

- *"The ring only pays off on large sparse nets; on a small or thickly-corded grid its allocation and tick-walking are pure overhead."* — Guarded. `n <= 8192 && n² <= 8m + 8192` routes those inputs to the whole-grid glance, which allocates one `double[n]` and touches no ring at all. This is also the step-5 second regime, and it is reached through the metaphor's own object (the grid is a physical thing of a size the child can or cannot take in at a glance), not bolted on.
- *"A badly chosen `Δ` degenerates the tick into Bellman-Ford."* — Guarded twice: `Δ = w_max/K` with `K` scaled to average degree so the in-tick short-cord subgraph stays subcritical, **and** `Δ` clamped to `4·w_mean` so a single enormous slack cord cannot widen the tick.
- *"A huge `w_max/w_mean` ratio demands an unbounded number of slices."* — Guarded: `NB` capped at 16384 and `Δ` recomputed upward to fit, so memory is bounded and the invariant is preserved by construction.
- *"Thread parallelism over a tick's frontier."* — **Dropped, deliberately.** The metaphor does offer it (all glints in one tick are simultaneous), but concurrent relaxation races on `dist[]` and `qs[]`, and I will not ship a mechanism whose correctness I could not measure in this session. Per the instruction to prefer vectorization first, the only parallelism shipped is data-level: `restrict` on every hot pointer, CSR (separate `int`/`double` streams, both read sequentially), and a 4-accumulator min-sweep in the glance path so the reduction is not one serial dependency chain.

The honest bottom line: the artifact is complete, self-consistent, and both of its paths are guarded — but the `1.8` is a prediction with nothing behind it yet, and the three numbers I named (`K`, the `4·w_mean` clamp, the `8` in the density test) are reasoned constants, not fitted ones. Until `dijkstra_bench` is actually reachable, treat this as an untested build.