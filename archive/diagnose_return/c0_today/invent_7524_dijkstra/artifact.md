# MAPPING

## SEED 1 — "A glint released from the traveler's jewel crawls down every cord at a pace set by the cord's own length."

| World object | Problem object |
|---|---|
| polished jewel, one per place | node `v`, index into arrays |
| infinite net the jewels are threaded into | the graph's adjacency structure (CSR) |
| blood-dark cord, jewel → jewel | directed edge `src[i] -> dst[i]` |
| length of the cord | `weight[i]` |
| the touch on the traveler's jewel | `dist[source] = 0` |
| glint released, crawling | a *pending relaxation event*, not a queue key |
| "no faster than the cord is long" | the event's arrival time is `dist[u] + w`; the event is **filed by arrival time**, never compared against other events |
| "a short cord flashes at once" | light edge, `w < Δ`: relaxed inside the *current* time window |
| "a long one keeps the glint waiting on the way" | heavy edge, `w ≥ Δ`: deferred to a later time window |
| "I do not chase the glint" | no argmin, no sift-up/sift-down, no priority structure is ever queried |
| the moment a jewel flares it re-emits ("already reflects every other jewel touching it") | a node's out-edges fire **on first arrival**, with no settlement test |

**Assumptions broken:** *"a road can only be considered once its starting place is fully settled"* (emission is triggered by arrival, not by settlement — a node inside the current window may still be improved and fire again), and, jointly, *"a priority structure must be consulted before every relaxation"* and *"each place's distance must be finalized before its neighbors are explored."*

## SEED 2 — "The first flare to reach each jewel is chalked once and every later flare along a longer cord is thrown away unmarked."

| World object | Problem object |
|---|---|
| chalk grid, one square per jewel | `dist_out[]`, one slot per node |
| white dust number in a square | current tentative distance |
| "first true flare" | the first (smallest-time) arrival — the write that sticks |
| a flare arriving late along a slacker cord | a stale queue entry / a failed `nd < dist[v]` test |
| "thrown away without marking twice" | lazy deletion: pop, see the mark says this glint is obsolete, `continue` |

**Assumption broken:** *"the next place to finalize is found by comparing against every remaining place."* Nothing is compared against everything; a single square is looked at, and the loser is simply discarded.

## SEED 3 — "A jewel's square that never catches any flare is left blank and crossed out like a wind-erased desert track."

| World object | Problem object |
|---|---|
| square that stays dark | node never reached |
| the scratch crossing it out | `INFINITY` |
| "however long I crouch" — he stops when no glint is left in flight | loop condition is `queued > 0`, not `visited < n` |

**Assumption broken:** *"the whole graph must be explored to know any single distance."* The crouching ends when no glint is still travelling; unreachable jewels are never touched at all.

---

# CHOSEN SEED

**SEED 1.** It is the one that breaks the preferred assumption (*"a road can only be considered once its starting place is fully settled"*) — and it does so in the most literal possible way: the glint leaves a jewel at the instant that jewel *first* lights, with no check that the light is final. It is also maximally distant from the known way: Dijkstra's whole identity is the settled-set invariant and the heap consulted before every relaxation; the native has neither. The image "a short cord flashes at once, a long one keeps the glint waiting on the way" is a *per-cord* distinction that the heap version simply does not possess.

Taken literally, the native's mechanism is a **timed filing cabinet of glints in flight**: drawers indexed by arrival time in units of Δ, short cords resolved inside the current drawer, long cords dropped into a later drawer. That is exactly **Δ-stepping / Dial's bucket-queue family** — a validated, published, real-world SSSP technique (Meyer & Sanders; Dial). Per step 4 I let the metaphor land on that rather than invent something new: the light/heavy cord split *is* the light/heavy edge split, the drawers *are* a circular bucket array, and the drawer ordering replaces the heap entirely.

Regime recognition, also in-world (step 5): *"if the net is drawn so tight that nearly every jewel touches nearly every other, or there are so few jewels that the whole grid fits under my hand, I do not file glints at all — I sweep my eye across the grid and pick the dimmest unlit square."* That is the plain `O(n²)` array-scan Dijkstra the known-way section names as a real practical win for dense/small graphs. It is chosen at runtime by a cheap density test, and it is also the guard for the one condition where the drawer machinery could lose (tiny graphs, where allocating drawers costs more than the whole problem).

# ASSUMPTION BROKEN

**"A road can only be considered once its starting place is fully settled."** In the artifact, a node's light out-edges are relaxed the moment it is drawn from the current bucket, with no `done[]` test and no guarantee its distance is final; if it later improves within the same bucket it is simply re-drawn and fires again. Settlement is a *consequence* of the bucket being exhausted, never a precondition for emission. Secondarily broken: no priority structure exists (O(1) bucket push/pop, no comparisons), no global argmin is ever taken in the sparse path, and the loop terminates on "no glint in flight" rather than on "all nodes visited."

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------------------
   The native's two ways of reading the chalk grid.

   (A) "the whole grid fits under my hand" / "the net is drawn so tight that
       nearly every jewel touches nearly every other" -> sweep the eye across
       every square and pick the dimmest unlit one. Plain O(n^2) array scan,
       no drawers, no heap. Chosen for tiny or dense graphs.

   (B) otherwise: file each glint in the drawer of the instant it will arrive.
       Short cords flash at once (light edges, resolved inside the current
       drawer); long cords keep the glint waiting on the way (heavy edges,
       deferred to a later drawer). No priority structure is ever consulted;
       a jewel fires the moment it first lights, settled or not.
   --------------------------------------------------------------------------- */

/* (A) sweep the grid by eye -- dense / tiny regime */
static void chalk_sweep(int n, const int *restrict off, const int *restrict edst,
                        const double *restrict ew, double *restrict dist)
{
    double *dtmp = (double *)malloc((size_t)n * sizeof(double));
    unsigned char *lit = (unsigned char *)calloc((size_t)n, 1);
    if (!dtmp || !lit) { free(dtmp); free(lit); return; }
    for (int j = 0; j < n; j++) dtmp[j] = dist[j];
    for (int it = 0; it < n; it++) {
        double best = INFINITY;
        for (int j = 0; j < n; j++) { double d = dtmp[j]; best = d < best ? d : best; }
        if (!(best < INFINITY)) break;              /* only dark squares remain */
        int u = -1;
        for (int j = 0; j < n; j++) if (dtmp[j] == best) { u = j; break; }
        if (u < 0) break;
        dtmp[u] = INFINITY; lit[u] = 1;
        double du = dist[u];
        const int e1 = off[u + 1];
        for (int e = off[u]; e < e1; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (!lit[v] && nd < dist[v]) { dist[v] = nd; dtmp[v] = nd; }
        }
    }
    free(dtmp); free(lit);
}

/* one glint filed into a drawer */
static inline int bpush(int **ba, int *bs, int *bc, int bi, int v)
{
    if (bs[bi] == bc[bi]) {
        int nc = bc[bi] ? bc[bi] * 2 : 16;
        int *p = (int *)realloc(ba[bi], (size_t)nc * sizeof(int));
        if (!p) return 0;
        ba[bi] = p; bc[bi] = nc;
    }
    ba[bi][bs[bi]++] = v;
    return 1;
}

void kernel(int n, int m, const int *restrict src, const int *restrict dst,
            const double *restrict weight, int source, double *restrict dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;   /* every square blank & crossed */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;                               /* the one touched jewel */
    if (m <= 0) return;

    /* ---- the pace of the glint: how long a cord must be before it is "slack" ---- */
    double wmax = 0.0, wsum = 0.0;
    for (int i = 0; i < m; i++) { double w = weight[i]; wsum += w; if (w > wmax) wmax = w; }
    double dlt;
    if (!(wmax > 0.0)) dlt = 1.0;
    else {
        double wmean  = wsum / (double)m;
        double avgdeg = (double)m / (double)n; if (avgdeg < 1.0) avgdeg = 1.0;
        dlt = 2.0 * wmean / avgdeg;                       /* ~ one cord-length per jewel */
        double lo1 = wmean / 64.0, lo2 = wmax / 1024.0;   /* keep the drawer-stack short */
        if (dlt < lo1) dlt = lo1;
        if (dlt < lo2) dlt = lo2;
        if (dlt > wmax) dlt = wmax;
        if (!(dlt > 0.0)) dlt = 1.0;
    }
    /* how many drawers must stay open at once: a glint can wait at most wmax/dlt */
    int nbneed = (int)(wmax / dlt) + 3;
    int NB = 8; while (NB < nbneed && NB < 4096) NB <<= 1;
    if (nbneed > NB) { dlt = wmax / (double)(NB - 3); if (!(dlt > 0.0)) dlt = 1.0; }
    const long long NBm = (long long)NB - 1;
    const double inv = 1.0 / dlt;

    /* ---- the net, threaded once: CSR with the short cords of each jewel first ---- */
    int    *off  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int    *lcnt = (int *)calloc((size_t)n, sizeof(int));
    int    *fl   = (int *)calloc((size_t)n, sizeof(int));
    int    *fh   = (int *)malloc((size_t)n * sizeof(int));
    int    *lend = (int *)malloc((size_t)n * sizeof(int));
    int    *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !lcnt || !fl || !fh || !lend || !edst || !ew) {
        free(off); free(lcnt); free(fl); free(fh); free(lend); free(edst); free(ew);
        return;
    }
    for (int i = 0; i < m; i++) { int u = src[i]; fl[u]++; if (weight[i] < dlt) lcnt[u]++; }
    off[0] = 0;
    for (int u = 0; u < n; u++) off[u + 1] = off[u] + fl[u];
    for (int u = 0; u < n; u++) { lend[u] = off[u] + lcnt[u]; fl[u] = off[u]; fh[u] = lend[u]; }
    for (int i = 0; i < m; i++) {
        int u = src[i]; double w = weight[i];
        int pos = (w < dlt) ? fl[u]++ : fh[u]++;
        edst[pos] = dst[i]; ew[pos] = w;
    }
    free(lcnt); free(fl); free(fh);

    /* ---- which reading of the grid? ---- */
    int use_sweep = (n <= 256) ||
                    ((double)m >= 0.10 * (double)n * (double)n && n <= 8192);
    if (use_sweep) {
        chalk_sweep(n, off, edst, ew, dist_out);
        free(off); free(lend); free(edst); free(ew);
        return;
    }

    /* ---- the drawers of waiting glints ---- */
    int **ba = (int **)calloc((size_t)NB, sizeof(int *));
    int  *bs = (int *)calloc((size_t)NB, sizeof(int));
    int  *bc = (int *)calloc((size_t)NB, sizeof(int));
    int  *inb = (int *)calloc((size_t)n, sizeof(int));   /* drawer+1 of latest glint, 0 = none */
    int  *heavy = (int *)malloc((size_t)n * sizeof(int));
    int  *hst = (int *)malloc((size_t)n * sizeof(int));
    if (!ba || !bs || !bc || !inb || !heavy || !hst) {
        if (ba) { for (int b = 0; b < NB; b++) free(ba[b]); free(ba); }
        free(bs); free(bc); free(inb); free(heavy); free(hst);
        chalk_sweep(n, off, edst, ew, dist_out);          /* safe fallback */
        free(off); free(lend); free(edst); free(ew);
        return;
    }
    for (int j = 0; j < n; j++) hst[j] = -1;

    long long queued = 0, i = 0;
    int phase = 0;
    if (bpush(ba, bs, bc, 0, source)) { inb[source] = 1; queued = 1; }

#define FLARE(Y, ND)                                                          \
    do {                                                                      \
        int y_ = (Y); double nd_ = (ND);                                      \
        if (nd_ < dist_out[y_]) {                                             \
            dist_out[y_] = nd_;                     /* chalk the square */     \
            long long g = (long long)(nd_ * inv);   /* drawer of arrival */    \
            if (g < i) g = i;                                                 \
            if (g > i + NBm) g = i + NBm;                                     \
            int bj = (int)(g & NBm);                                          \
            if (inb[y_] != bj + 1) {                                          \
                if (bpush(ba, bs, bc, bj, y_)) { inb[y_] = bj + 1; queued++; }\
            }                                                                 \
        }                                                                     \
    } while (0)

    while (queued > 0) {
        int bi = (int)(i & NBm);
        if (bs[bi] == 0) { i++; continue; }              /* no glint due this instant */
        phase++;
        int hcnt = 0, p = 0;

        /* short cords flash at once: keep draining this drawer, including what
           the draining itself puts back into it. No settlement test anywhere. */
        while (p < bs[bi]) {
            int v = ba[bi][p++];
            queued--;
            if (inb[v] != bi + 1) continue;              /* a later, slacker flare: thrown away */
            inb[v] = 0;
            if (hst[v] != phase) { hst[v] = phase; heavy[hcnt++] = v; }
            double dv = dist_out[v];
            const int le = lend[v];
            for (int e = off[v]; e < le; e++) FLARE(edst[e], dv + ew[e]);
        }
        bs[bi] = 0;

        /* long cords kept their glints waiting; now they are let go */
        for (int k = 0; k < hcnt; k++) {
            int v = heavy[k];
            double dv = dist_out[v];
            const int e1 = off[v + 1];
            for (int e = lend[v]; e < e1; e++) FLARE(edst[e], dv + ew[e]);
        }

        if (bs[bi] == 0) i++;    /* only leave the instant when nothing is due in it */
    }
#undef FLARE

    for (int b = 0; b < NB; b++) free(ba[b]);
    free(ba); free(bs); free(bc); free(inb); free(heavy); free(hst);
    free(off); free(lend); free(edst); free(ew);
}
```

# PREDICTION

PREDICTION: speedup_vs_heap = 2.2

Reasoning behind the number, stated before any measurement: the reference does up to `m` heap pushes and `m` pops at `~log m` comparisons and swaps each, with pointer-chasing over a `(double,int)` array that thrashes cache. The artifact replaces every one of those with an O(1) append to an `int` drawer plus one `cvttsd2si`. For a sparse graph with `n ≈ 10^5–10^6` and `m/n ≈ 4–16`, `log m ≈ 20`, so the queue work drops by roughly an order of magnitude; but CSR construction (identical in both) and the `m` edge relaxations (identical in both) are unaffected and dominate a growing share, which is what pulls the end-to-end figure down from ~5× to ~2×. I expect 1.6–3.0× on sparse random/geometric graphs, and roughly parity-to-2× on dense ones where the sweep path takes over. I expect **no** speedup — possibly a small loss — if the benchmark is dominated by graphs in the 256 < n < 2000 sparse range, where `log n` is only ~10 and the drawer allocations are not yet amortised.

# MEASUREMENT

No measurement was taken. `dijkstra_bench` and `dijkstra_contract` were listed as available but this session was invoked with no tool access, so I could neither compile nor time the kernel. I am reporting that plainly rather than inventing numbers: **the prediction above is unvalidated.**

What I *can* state without the bench, and what would falsify the prediction:

- **Correctness argument (not a substitute for the contract check).** The kernel is correct independent of drawer ordering: `dist_out[y]` is only ever lowered, every lowering re-files the node, the clamp `i ≤ g ≤ i+NB-1` guarantees every filed glint lands in a drawer that will still be visited, and `i` only advances when the current drawer is empty — so every node is eventually relaxed out with its *final* distance. That is a Bellman-Ford-style invariant; Δ-stepping's bucket order is there purely for speed. Unreachable nodes are never filed and keep `INFINITY`. Floating-point: the artifact takes a min over a *superset* of the relaxations Dijkstra performs, so any disagreement with the reference is at rounding level (a few ULP), which is what the stated tolerance is for. If the contract check reports a mismatch larger than that, the likely culprit is the drawer-index clamp interacting with a pathological weight distribution, and the honest fix is to widen `NB` or fall back to `chalk_sweep`.
- **Falsifiers.** speedup < 1.0 on the bench's sparse cases would falsify the core claim (that removing the priority structure pays for the re-relaxations). A measured speedup > 4× would mean I badly underestimated how much of the reference's time is heap traffic.

# VERDICT

The native's mechanism is not a disguised heap. Its content is that emission is triggered by *arrival time*, not by settlement, and that the cord's own length decides whether its glint resolves now or waits — and taken literally that is Δ-stepping with a circular bucket array, a validated published algorithm rather than an invention of mine. I deliberately let the metaphor land there instead of building something novel, as instructed.

Honest accounting of the risks, and how each is addressed rather than waved at:

- **Risk: small graphs.** Δ-stepping's drawers cost `NB` pointer/size arrays and per-drawer `realloc`s that a few-hundred-node problem never amortises. **Addressed** by the `n <= 256` branch into `chalk_sweep`, not by hoping the bench is large.
- **Risk: dense graphs.** The known-way section explicitly names the `O(n²)` scan as the real win there, and the drawer machinery would touch `m ≈ n²` edges with extra re-relaxations. **Addressed** by the `m ≥ 0.10·n²  &&  n ≤ 8192` branch into the same sweep path — this is the second regime the native himself recognises ("the net drawn so tight that nearly every jewel touches nearly every other"). The `n ≤ 8192` cap exists so the `O(n²)` scan is never chosen where `n²` itself becomes the bottleneck.
- **Risk: skewed weights blowing up the drawer stack or the empty-drawer scan.** **Addressed** by the two floors on Δ (`wmean/64`, `wmax/1024`), which bound `NB ≤ 2048` and bound total empty-drawer advances at roughly the edge count.
- **Risk: Δ chosen badly.** The heuristic `Δ = 2·w̄·n/m` is the standard `Θ(1/degree)` choice. If the bench's weight distribution is adversarial, the degradation is graceful — more re-relaxations, never a wrong answer — but it is the most likely reason a measured speedup would land near 1.0 rather than 2.2.

I did **not** add OpenMP. The metaphor's parallel unit is the frontier inside one drawer, and at plausible bench sizes those frontiers are typically hundreds of nodes — too small to pay for atomic `dist` updates and a barrier per phase, and the guarded-fallback discipline would then require a size check I cannot calibrate without the bench. Vectorization hints (`restrict` throughout, CSR with light cords contiguous at the front of each node's segment so the hot inner loop is a straight stride-1 walk, the two SIMD-friendly reduction passes in `chalk_sweep` instead of one argmin loop) are in; threads are out. Given that I could not measure, shipping an unmeasured threaded path would have been the worse call.

**Bottom line:** the translation is literal, the mechanism is a real validated technique reached through the metaphor rather than around it, both regimes named in the known-way section are handled with a runtime test, every risk my own verdict names is guarded — and the 2.2× is a prediction I could not check in this session, which is a genuine gap in this answer, not a formality.