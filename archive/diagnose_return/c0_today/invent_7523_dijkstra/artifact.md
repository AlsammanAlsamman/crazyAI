## MAPPING

| World object (native's words) | Computational object | 
|---|---|
| a *place* / a *heap* | a graph node `v ∈ [0,n)` |
| a *road* | a directed edge `src[i] -> dst[i]` |
| the road's *notch* on the stick | `weight[i]` |
| the *ember* of a heap | `dist_out[v]`, the current tentative distance |
| ember "banked so low it is nearly out but never out" at the traveler | `dist_out[source] = 0.0` (finite, minimal) |
| "unlit heap, only cold ash" | `dist_out[v] = INFINITY` |
| a *runner* going down a road | one edge relaxation |
| the *stick he lays* on the far heap | the candidate value `dist_out[u] + w` |
| "heap already ringed by a garrison, refusing entry" | the test `nd < dist_out[v]` failing — the offer is rejected |
| *dropped stick fed to the hearth* | candidate discarded, zero further cost (it fuels the loop, i.e. it is the cheap-rejection fast path) |
| *the land-wide throw, one row read straight across* | a single lock-step pass in which **every** heap resolves its own minimum — a per-node reduction, never a cross-node comparison |
| *shortest sliver becomes the new ember, longer ones burned* | `dist_out[v] = min(dist_out[v], candidates)` |
| *garrison rides in when the ember matched the previous throw* | node did not improve this round ⇒ it is dropped from the next frontier; its out-edges are not walked again |
| *"its runners stop going out for good"* | node stays inactive **unless** a shorter stick later breaks the seal (my one honest amendment — see VERDICT) |
| *heap whose ash never once catches a spark* | never entered any frontier ⇒ unreachable ⇒ `INFINITY` |
| *the hooded figure reciting the embers in one pass* | the final contiguous `dist_out[0..n)` array |
| *counting roads against places before the first runner leaves* | runtime regime detection: density `m` vs `n²` |
| *abandoning the pyramid ritual when garrisons never cover the land* | work-budget bail-out to a classical Dijkstra |

### Per-seed mapping and the assumption each breaks

**SEED 1 — "a runner walks each road carrying its notch and lays a new stick equal to home ember + notch."**
| world | problem |
|---|---|
| runner leaves a heap that merely has *some* ember | edge relaxed from a **tentative** `dist[u]`, not a settled one |
| every road walked "at once" | edge-centric, not node-centric, traversal |

Breaks: *"a road can only be considered once its starting place is fully settled."*

**SEED 2 — "every heap throws its sticks together in one land-wide throw; the row read straight across picks each heap's shortest sliver."**
| world | problem |
|---|---|
| one throw for the whole land | one synchronous round over the whole active set |
| *a heap reads only its own sliver* | the reduction is strictly **per-node**, local |
| *the row is read across all heaps in the same breath* | all nodes resolved in one linear, prefetch-friendly, branch-light sweep |
| no fortune-teller ranks the heaps against each other | **no priority structure and no global argmin anywhere** |

Breaks: *"a priority structure must be consulted before every relaxation"* **and, directly, "the next place to finalize is found by comparing against every remaining place"* — nothing is ever compared against every remaining place; each heap only ever compares its own sticks to its own ember.

**SEED 3 — "a garrison seals a heap once its ember stops changing between throws."**
| world | problem |
|---|---|
| sealing | finalization detected *a posteriori* from a fixed point, not imposed in advance |
| runners stop going out | node leaves the active frontier |

Breaks: *"each place's distance must be finalized before its neighbors are explored."*

## CHOSEN SEED

**SEED 2.** It is the only one of the three that breaks the preferred assumption — *"the next place to finalize is found by comparing against every remaining place"* — and it is the furthest from the known way: it deletes the heap **and** the O(n) scan simultaneously, replacing both with a per-node local minimum taken in one lock-step sweep. SEEDs 1 and 3 fall out of it as consequences (a throw only happens if runners already ran; a garrison only rides in when a throw changed nothing), so choosing SEED 2 pulls the whole ritual along with it.

## ASSUMPTION BROKEN

**"The next place to finalize is found by comparing against every remaining place."** In this kernel there is never a global comparison of any kind — no heap `pop`, no `argmin` scan, no bucket. A node's ember is decided only against sticks laid on *that* node. Global progress comes from the ritual being repeated, not from any global ordering. Secondarily this also kills *"a priority structure must be consulted before every relaxation"* and *"a road can only be considered once its starting place is fully settled."*

Letting the mechanism land on a validated technique rather than an invention: what the native describes, taken literally, **is** frontier/active-set Bellman–Ford–Moore (the level-synchronous form used in every parallel and GPU SSSP library). I did not steer it there; the throw-seal-send loop is that algorithm. Its one well-known weakness — graphs with large shortest-path hop depth, where rounds multiply — is exactly the native's own "or until a heap's ash never once catches a spark across throw after throw," so the metaphor itself supplies the abandon-the-ritual clause, which I implement as a hard work budget with a fallback to the two known-way regimes (heap Dijkstra for sparse, O(n²) scan for dense).

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================= abandoned-ritual fallback A: binary-heap Dijkstra =========
   "call in the hooded figure with the ranked ledger" - sparse regime.        */
typedef struct { double d; int u; } HItem;

static void dij_heap(int n, int m, const int *restrict off, const int *restrict edst,
                     const double *restrict ew, int source, double *restrict dist)
{
    unsigned char *done;
    HItem *h;
    int hs;
    int i;
    for (i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    done = (unsigned char *)calloc((size_t)n, 1);
    h = (HItem *)malloc(((size_t)m + 2) * sizeof(HItem));
    if (!done || !h) { free(done); free(h); return; }
    h[0].d = 0.0; h[0].u = source; hs = 1;
    while (hs > 0) {
        HItem top = h[0];
        int u, e, e1;
        double du;
        --hs;
        if (hs) {                       /* sift the last item down from the root */
            HItem x = h[hs];
            int j = 0;
            for (;;) {
                int l = 2 * j + 1, r, s;
                if (l >= hs) break;
                r = l + 1;
                s = (r < hs && h[r].d < h[l].d) ? r : l;
                if (h[s].d >= x.d) break;
                h[j] = h[s]; j = s;
            }
            h[j] = x;
        }
        u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        du = top.d;
        e1 = off[u + 1];
        for (e = off[u]; e < e1; e++) {
            int v = edst[e];
            double nd;
            if (done[v]) continue;
            nd = du + ew[e];
            if (nd < dist[v]) {
                HItem x; int j;
                dist[v] = nd;
                x.d = nd; x.u = v;
                j = hs++;
                while (j > 0) { int p = (j - 1) >> 1; if (h[p].d <= x.d) break; h[j] = h[p]; j = p; }
                h[j] = x;
            }
        }
    }
    free(done); free(h);
}

/* ================= abandoned-ritual fallback B: O(n^2) array-scan Dijkstra ===
   dense regime: no heap at all, the min sweep vectorizes.                    */
static void dij_scan(int n, const int *restrict off, const int *restrict edst,
                     const double *restrict ew, int source, double *restrict dist)
{
    unsigned char *done = (unsigned char *)calloc((size_t)n, 1);
    int it, i;
    if (!done) return;
    for (i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    for (it = 0; it < n; it++) {
        double best = INFINITY;
        int u = -1, e, e1;
        for (i = 0; i < n; i++) {                   /* branchless min-reduction */
            double d = done[i] ? INFINITY : dist[i];
            best = d < best ? d : best;
        }
        if (!(best < INFINITY)) break;
        for (i = 0; i < n; i++) if (!done[i] && dist[i] == best) { u = i; break; }
        if (u < 0) break;
        done[u] = 1;
        e1 = off[u + 1];
        for (e = off[u]; e < e1; e++) {
            int v = edst[e];
            double nd = best + ew[e];
            if (nd < dist[v]) dist[v] = nd;
        }
    }
    free(done);
}

/* ============================ THE RITUAL ====================================
   one ember per place; runners down every road of every lit, unsealed heap;
   one land-wide throw per round resolving each heap's own shortest sliver;
   a garrison on every heap whose ember did not move.                         */
void kernel(int n, int m, const int *restrict src, const int *restrict dst,
            const double *restrict weight, int source, double *restrict dist_out)
{
    int *ibuf, *off, *edst, *fro, *nxt, *stamp, *cur;
    double *ew;
    int i, fn, round, bail, lg, t;
    unsigned long long work, budget;

    if (n <= 0) return;
    for (i = 0; i < n; i++) dist_out[i] = INFINITY;   /* cold ash everywhere   */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;                           /* the banked ember      */
    if (m <= 0) return;

    /* one arena for every integer structure, one for the notches */
    ibuf = (int *)malloc(((size_t)m + 4u * (size_t)n + 8u) * sizeof(int));
    ew   = (double *)malloc((size_t)m * sizeof(double));
    if (!ibuf || !ew) { free(ibuf); free(ew); return; }
    off   = ibuf;
    edst  = off  + ((size_t)n + 1);
    fro   = edst + (size_t)m;
    nxt   = fro  + (size_t)n;
    stamp = nxt  + (size_t)n;

    /* roads laid out by their home place (CSR) */
    memset(off, 0, ((size_t)n + 1) * sizeof(int));
    for (i = 0; i < m; i++) off[src[i] + 1]++;
    for (i = 0; i < n; i++) off[i + 1] += off[i];
    cur = fro;                                        /* scratch cursor, n ints */
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (i = 0; i < m; i++) {
        int p = cur[src[i]]++;
        edst[p] = dst[i];
        ew[p]   = weight[i];
    }
    for (i = 0; i < n; i++) stamp[i] = -1;

    /* "count the roads against the places": how long may the ritual run before
       it is abandoned?  ~ the cost of the ranked-ledger way, so we can never
       lose by more than the restart.                                         */
    lg = 1; t = n; while (t > 1) { t >>= 1; lg++; }
    if (lg > 24) lg = 24;
    budget = ((unsigned long long)m + (unsigned long long)n) * (unsigned long long)(lg + 1);
    work = 0;

    fn = 0; fro[fn++] = source;
    round = 0; bail = 0;

    while (fn > 0) {
        int nn = 0, k;
        for (k = 0; k < fn; k++) {                    /* runners down every road */
            int u = fro[k];
            double du = dist_out[u];
            int e0 = off[u], e1 = off[u + 1], e;
            work += (unsigned long long)(unsigned int)(e1 - e0);
            for (e = e0; e < e1; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                double dv;
                if (e + 8 < e1) __builtin_prefetch(&dist_out[edst[e + 8]], 1, 1);
                dv = dist_out[v];
                /* the throw, resolved locally per heap: shortest sliver wins,
                   every longer sliver is burned.  A heap already holding a
                   shorter ember is garrisoned and refuses entry.             */
                if (nd < dv) {
                    dist_out[v] = nd;
                    if (stamp[v] != round) { stamp[v] = round; nxt[nn++] = v; }
                }
            }
        }
        if (work > budget) { bail = 1; break; }
        { int *tmp = fro; fro = nxt; nxt = tmp; }
        fn = nn;                 /* heaps whose ember did not move are sealed:
                                    their runners do not go out next round.   */
        round++;
    }

    if (bail) {                  /* the ash never caught: abandon the ritual   */
        double dn = (double)n;
        if ((double)m > 0.0625 * dn * dn) dij_scan(n, off, edst, ew, source, dist_out);
        else                              dij_heap(n, m, off, edst, ew, source, dist_out);
    }

    free(ibuf); free(ew);
}
```

## PREDICTION

PREDICTION: speedup_vs_heap = 2.2

Reasoning behind the number, stated before any measurement: on graphs with small shortest-path hop depth (random, scale-free, dense) the ritual touches each edge roughly 2–4 times with a load, an add, a compare and an occasional store — a handful of cycles — while the reference performs up to `m` heap pushes at ~log₂n pointer-chasing swaps each, on a 16-byte-per-item structure that is ~2× my memory footprint. That is a 4–8× advantage on the *search* phase. It is diluted because CSR construction (two scattered passes over `m`) is paid identically by both and is a large share of a fast kernel's total, so I expect the end-to-end figure to land near 2×. Downside case if the benchmark uses a grid or road-like graph with deep hop structure: the budget trips, we restart, and the measured figure is ~0.85–0.95×.

## MEASUREMENT

**Not measured.** `dijkstra_bench` and `dijkstra_contract` were not available in this session — no tools were exposed at all — so I ran zero of the four permitted improvement rounds against real numbers. I am reporting that plainly rather than inventing a table. Everything above is a prediction and a static argument; the pipeline's compile-and-measure step is the first real evidence. The specific numbers to check: (1) speedup vs the reference heap kernel per graph shape, (2) whether `bail` ever fires (if it does on the main benchmark graph, the chosen seed loses and the fallback is what is being timed), and (3) exact agreement with the reference, which should be bit-identical, not merely within tolerance — see below.

## VERDICT

**Exactness.** Both kernels compute, for each node, the minimum over paths of the left-to-right accumulated sum along that path. Per path the summation order is identical, so per-path values are bit-identical; with all weights ≥ 0, `a + w >= a` holds in IEEE round-to-nearest, so Dijkstra's pruning is sound and its answer equals the full min. The two minima are taken over the same finite set of identical doubles. I expect exact equality, not tolerance-level agreement. If the harness reports a mismatch, the hypothesis is wrong and something in my edge ordering or the graph contains a negative weight.

**The one place I did not stay literal, stated plainly.** The native says a sealed heap's ember "will not shorten again no matter what still arrives" and its runners stop "for good." Taken at face value that is an incorrect algorithm: a heap can be unchanged in one throw and then receive a genuinely shorter stick in the next, because one of its upstream neighbours only just shortened. I kept the seal as *deactivation* — a sealed heap sends no runners — but I let a shorter stick break the seal and put the heap back in the next round's frontier. This is the smallest amendment that preserves the ritual and returns correct distances. I did not quietly swap in the textbook method to get there; the rest of the loop is the native's, including the absence of any global comparison.

**Risk named and guarded.** The stated weakness of this mechanism is graphs with deep shortest-path hop structure, where rounds multiply and total relaxations exceed heap Dijkstra's work. Per the rule that a self-named risk must be guarded, it is: `work` is accumulated over every edge walked and the ritual is abandoned the moment it exceeds `(m+n)·(log₂n + 1)`, which is the order of the heap kernel's own cost. Worst case is therefore a restart — roughly 2× the fallback, never unbounded. I did not leave this as a caveat in prose.

**Two regimes, as required.** The known-way section names sparse-with-heap and dense-with-plain-scan. The abandon branch chooses between them at runtime by density (`m > n²/16` → the scan, else the heap). I will be honest that this branch is rarely taken: dense graphs converge in two or three throws and almost never blow the budget, so the scan path is a real but cold path. I kept it because it is the regime answer the brief demands and it costs nothing when unused; I would not claim it as a measured win.

**No thread parallelism.** Per the instruction to prefer vectorization hints first, I used `restrict`, a single-arena layout, a two-stream CSR (12 bytes/edge rather than the reference's 16), and a prefetch on the scattered `dist_out[edst[e]]` access — which is the actual bottleneck of the inner loop. The throw's scatter is data-dependent and would need atomics or per-thread candidate buffers to parallelize; without a measurement of the benchmark's actual sizes I have no evidence the units of work justify that, so I left it out rather than ship an unguarded guess.

**What would falsify this.** If the measured speedup is below 1.0 on the main benchmark graph and `bail` never fires, then the per-edge relaxation count is far higher than the 2–4× I assumed, and the seal rule is not pruning enough — the honest next move would be Δ-stepping (bucketed throws), which reintroduces a weak ordering and therefore partially concedes the assumption I set out to break.