## 1. SEED → PROBLEM MAPPINGS

### SEED 1 — "A runner walks each road carrying its notch and lays a new stick on the far heap equal to the home ember's length plus that notch."

| World object | Problem object |
|---|---|
| place / heap | node index `v ∈ [0,n)` |
| road | directed edge `src[i] -> dst[i]` in CSR |
| notch on the runner's stick | `weight[i]` |
| lit ember at the home place | current tentative `dist[u]` (finite) |
| runner sent down the road | one edge-relaxation task |
| new stick laid on the far heap | candidate value `dist[u] + w` written into `dst`'s slot |
| "every road at once" | all out-edges of all currently-lit, unsealed places, in one sweep |

**Silent assumption broken:** *"a road can only be considered once its starting place is fully settled."* Runners leave from any **lit** ember, settled or not — a merely-tentative ember is enough fuel to send a runner.

### SEED 2 — "Every heap throws its sticks together in one land-wide throw, and the row read straight across picks each heap's shortest sliver as its new ember while the rest are burned for fuel."

| World object | Problem object |
|---|---|
| a heap's pile of sticks | the set of candidate values proposed to node `v` this round |
| one land-wide throw | one global, synchronized round: `dist[]` is read-only while candidates accumulate, then applied |
| "read straight across a single row, one sliver from every heap in the same breath" | a **lane-parallel (SIMD) pass across nodes** — same position in every node's pile, many nodes per instruction |
| shortest sliver in a heap | per-node `min` reduction, *local to that node* |
| the new ember | `dist[v] = min(dist[v], cand[v])` |
| longer slivers burned for fuel | candidates `>= dist[v]` discarded on the spot (never stored) |
| "a fire not given fuel goes out" | a round with no accepted candidate ends the process |

**Silent assumptions broken:** *"the next place to finalize is found by comparing against every remaining place"* — **no place is ever compared with any other place.** There is no global argmin, no "next" place; every heap independently takes its own local minimum, all heaps in the same breath. Also broken: *"a priority structure must be consulted before every relaxation"* — there is no priority structure at all.

### SEED 3 — "A garrison seals a heap once its ember stops changing between throws, fixing its distance for good and halting its runners."

| World object | Problem object |
|---|---|
| garrison riding in | node dropped from the active frontier |
| "ember matched what it held after the throw before" | `dist[v]` unchanged across consecutive rounds |
| "its own runners stop going out for good" | `v`'s out-edges no longer relaxed |
| "ringed by their own garrison, refusing entry" | the relax test itself: a heap refuses any stick not strictly shorter than its ember |
| "ash never once catches a spark" | node never reached → stays `INFINITY` (unreachable) |
| hooded figure's final recitation | one linear read-out of `dist_out[]` |

**Silent assumption broken:** *"each place's distance must be finalized before its neighbors are explored."* Finalization is detected **afterwards, empirically** (the ember stopped moving), not decided in advance.

---

## 2. CHOSEN SEED

**SEED 2.** It is the only one of the three that breaks the preferred assumption — *"the next place to finalize is found by comparing against every remaining place"* — and it breaks it totally. Both known ways need that cross-place comparison: the binary heap is a device for finding the global minimum, and the dense O(n²) variant is literally "compare against every remaining place." SEED 2 deletes the concept of a global minimum: each heap only ever looks at its own sticks. Its "one sliver from every heap in the same breath" is also the most directly executable image of the three — it is a vector instruction.

SEEDs 1 and 3 are not discarded; they are the runner phase and the frontier-shrink phase of SEED 2's throw loop.

**One honest repair.** Taken literally, SEED 3's boast — "its ember will not shorten again *no matter what still arrives*" — is false. Counterexample: `s→v` weight 10, and `s→a→b→v` weights 1,1,1. Round 1 lights `v=10`, `a=1`. Round 2 lights `b=2` and leaves `v` at 10 → the garrison would seal `v` at 10, but 3 arrives in round 3. So I keep the half of the garrison that *is* sound — **runners stop going out when the ember didn't move** (re-relaxing an unchanged ember can only re-propose sticks that were already refused) — and let a genuinely shorter stick break the seal and re-muster the runners. I am not replacing the native's idea; I am keeping its mechanism and dropping only its unprovable boast.

## 3. ASSUMPTION BROKEN

> **"the next place to finalize is found by comparing against every remaining place"** — and with it, *"a priority structure must be consulted before every relaxation"* and *"a road can only be considered once its starting place is fully settled."*

No place is compared to any other place. No priority structure exists. Relaxation happens from tentative embers.

**Where the mechanism lands (step 4).** Round-based throwing + local min + a frontier that shrinks when embers stop moving *is* a validated real technique: **synchronous frontier-based Bellman-Ford** (the topology/data-driven SSSP of Gunrock, nvGRAPH, and every GPU SSSP paper). I let the metaphor arrive there rather than invent something. The metaphor's *own* regime language ("if the land is so crisscrossed that every place borders every other, send the hooded figure instead"; "if the runners have burned more sticks than the hooded figure would have counted, dismiss them") supplies the two fallbacks:

- **dense / tiny land** → the **SIMD array-scan O(n²) Dijkstra**, where "read straight across a single row in one breath" becomes an AVX2 min-reduce with index tracking. This is exactly the known dense win named in the brief.
- **pathological land** (the runners keep re-lighting the same embers) → a hard work budget, then **heap Dijkstra** from scratch. Wasted streaming work before the bail is ≈8 passes over the edges ≈ under 10% of one heap-Dijkstra, so the guarded worst case is ~1.1× the known way, not a blow-up.

No thread parallelism: the runner phase *is* parallel-safe (`dist` read-only during a throw), but an atomic-min on doubles plus an atomic frontier append costs more than it buys until a single round carries ~10⁶ edges. Stated, not shipped.

## 4. ARTIFACT

```c
/* THROW-AND-SEAL  ---  single-source shortest distances.
 *
 * place            -> node index
 * road             -> CSR edge (dst, weight)
 * notch            -> weight
 * lit ember        -> finite tentative dist_out[v]
 * cold ash         -> INFINITY
 * runner           -> one relaxation, launched from a merely-tentative ember
 * stick on a heap  -> candidate value cand[v]
 * garrison refuses -> the strict relax test (nd < dist && nd < cand)
 * land-wide throw  -> one synchronized round: dist read-only, then applied
 * row read across  -> per-node local min; SIMD, 4 heaps per breath
 * burned for fuel  -> candidate dropped, never stored
 * garrison seals   -> node leaves the frontier; its runners stop
 * hooded figure    -> the O(n^2) row-scan for the crisscrossed land
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ---------------- THE HOODED FIGURE: dense / tiny land ----------------
 * O(n^2) array-scan Dijkstra.  Each dawn he reads straight across one
 * row of embers in a single SIMD breath and names the dimmest unsealed
 * heap.  Sealed heaps carry key = INFINITY, so the scan is a pure
 * vector min with index tracking -- no branchy mask bookkeeping.      */
static void hooded_scan(int n,
                        const int    * restrict off,
                        const int    * restrict edst,
                        const double * restrict ew,
                        int source,
                        double * restrict dist,
                        double * restrict key)
{
    for (int i = 0; i < n; i++) { dist[i] = INFINITY; key[i] = INFINITY; }
    dist[source] = 0.0; key[source] = 0.0;

    for (int it = 0; it < n; it++) {
        int    best = -1;
        double bd   = INFINITY;
        int    i    = 0;
#ifdef __AVX2__
        if (n >= 8) {
            __m256d vb   = _mm256_set1_pd(INFINITY);
            __m256d vi   = _mm256_set1_pd(-1.0);
            __m256d step = _mm256_set1_pd(4.0);
            __m256d idx  = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
            for (; i + 4 <= n; i += 4) {
                __m256d d  = _mm256_loadu_pd(key + i);
                __m256d lt = _mm256_cmp_pd(d, vb, _CMP_LT_OQ);
                vb  = _mm256_blendv_pd(vb, d,   lt);
                vi  = _mm256_blendv_pd(vi, idx, lt);
                idx = _mm256_add_pd(idx, step);
            }
            double bl[4], il[4];
            _mm256_storeu_pd(bl, vb);
            _mm256_storeu_pd(il, vi);
            for (int j = 0; j < 4; j++)
                if (bl[j] < bd) { bd = bl[j]; best = (int)il[j]; }
        }
#endif
        for (; i < n; i++)
            if (key[i] < bd) { bd = key[i]; best = i; }

        if (best < 0 || !(bd < INFINITY)) break;   /* only cold ash left */
        key[best] = INFINITY;                      /* garrison seals it  */

        const int e1 = off[best + 1];
        for (int e = off[best]; e < e1; e++) {
            const int v = edst[e];
            const double nd = bd + ew[e];
            /* a sealed v has dist[v] <= bd <= nd, so it refuses entry */
            if (nd < dist[v]) { dist[v] = nd; key[v] = nd; }
        }
    }
}

/* -------- THE HOODED FIGURE WITH HIS HEAP: pathological-land bail ---- */
typedef struct { double d; int u; } HItem;

static void hpush(HItem * restrict h, int *hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= d) break;
        h[i] = h[p]; i = p;
    }
    h[i].d = d; h[i].u = u;
}

static void heap_dijkstra(int n, int m,
                          const int    * restrict off,
                          const int    * restrict edst,
                          const double * restrict ew,
                          int source,
                          double * restrict dist,
                          unsigned char * restrict done,
                          HItem * restrict heap)
{
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    memset(done, 0, (size_t)n);
    dist[source] = 0.0;
    int hs = 0;
    hpush(heap, &hs, 0.0, source);
    while (hs > 0) {
        HItem top = heap[0];
        int sz = --hs;
        HItem last = heap[sz];
        if (sz > 0) {
            int i = 0;
            for (;;) {
                int l = 2 * i + 1;
                if (l >= sz) break;
                int r = l + 1;
                int s = (r < sz && heap[r].d < heap[l].d) ? r : l;
                if (heap[s].d >= last.d) break;
                heap[i] = heap[s];
                i = s;
            }
            heap[i] = last;
        }
        const int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        const double du = dist[u];
        const int e1 = off[u + 1];
        for (int e = off[u]; e < e1; e++) {
            const int v = edst[e];
            const double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }
    (void)m;
}

/* ============================== KERNEL ============================== */
void kernel(int n, int m, const int *src, const int *dst,
            const double *weight, int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- file every road under its home place (CSR) ---- */
    int    *off  = (int *)calloc((size_t)n + 1, sizeof(int));
    int    *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !edst || !ew) { free(off); free(edst); free(ew); return; }

    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    {
        int *fill = (int *)malloc((size_t)n * sizeof(int));
        if (!fill) { free(off); free(edst); free(ew); return; }
        memcpy(fill, off, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            const int u = src[i];
            const int p = fill[u]++;
            edst[p] = dst[i];
            ew[p]   = weight[i];
        }
        free(fill);
    }

    /* ---- REGIME 1: crisscrossed or tiny land -> the hooded figure ----
     * row-scan cost ~ n^2/4 (SIMD) + m ; throw-and-seal ~ 3m.
     * scan wins once m > n^2/12.                                     */
    if (n < 32 || (double)m * 12.0 > (double)n * (double)n) {
        double *key = (double *)malloc((size_t)n * sizeof(double));
        if (key) {
            hooded_scan(n, off, edst, ew, source, dist_out, key);
            free(key);
            free(off); free(edst); free(ew);
            return;
        }
        /* no scratch: fall through to the runners */
    }

    /* ---- REGIME 2: open land -> THROW AND SEAL ---- */
    double        *cand     = (double *)malloc((size_t)n * sizeof(double));
    int           *frontier = (int *)malloc((size_t)n * sizeof(int));
    int           *next     = (int *)malloc((size_t)n * sizeof(int));
    unsigned char *innext   = (unsigned char *)calloc((size_t)n, 1);
    if (!cand || !frontier || !next || !innext) {
        free(cand); free(frontier); free(next); free(innext);
        free(off); free(edst); free(ew);
        return;
    }
    for (int i = 0; i < n; i++) cand[i] = INFINITY;

    int fcount = 1;
    frontier[0] = source;

    /* the native's own thrift: if the runners burn more sticks than the
     * hooded figure would have counted, dismiss them.                */
    const size_t budget = (size_t)8 * (size_t)m + (size_t)16 * (size_t)n + 4096;
    size_t spent = 0;
    int bailed = 0;

    while (fcount > 0) {
        if (spent > budget) { bailed = 1; break; }
        int ncount = 0;

        /* --- the runners: dist_out is READ-ONLY for the whole round --- */
        for (int fi = 0; fi < fcount; fi++) {
            const int u = frontier[fi];
            const double du = dist_out[u];
            const int e0 = off[u], e1 = off[u + 1];
            spent += (size_t)(e1 - e0) + 1u;
            int e = e0;
#ifdef __AVX2__
            if (e1 - e0 >= 16) {
                const __m256d vdu = _mm256_set1_pd(du);
                for (; e + 4 <= e1; e += 4) {
                    __m128i vv = _mm_loadu_si128((const __m128i *)(edst + e));
                    __m256d vw = _mm256_loadu_pd(ew + e);
                    __m256d nd = _mm256_add_pd(vdu, vw);
                    __m256d dv = _mm256_i32gather_pd(dist_out, vv, 8);
                    __m256d cv = _mm256_i32gather_pd(cand,     vv, 8);
                    __m256d lt = _mm256_and_pd(
                        _mm256_cmp_pd(nd, dv, _CMP_LT_OQ),
                        _mm256_cmp_pd(nd, cv, _CMP_LT_OQ));
                    int msk = _mm256_movemask_pd(lt);
                    if (msk) {
                        double nds[4];
                        _mm256_storeu_pd(nds, nd);
                        for (int j = 0; j < 4; j++) if (msk & (1 << j)) {
                            const int v = edst[e + j];
                            /* re-test: two lanes may share the same heap */
                            if (nds[j] < dist_out[v] && nds[j] < cand[v]) {
                                cand[v] = nds[j];
                                if (!innext[v]) { innext[v] = 1; next[ncount++] = v; }
                            }
                        }
                    }
                }
            }
#endif
            for (; e < e1; e++) {
                const int v = edst[e];
                const double nd = du + ew[e];
                /* a heap already ringed by its garrison refuses the stick,
                 * and the stick is burned for fuel on the spot           */
                if (nd < dist_out[v] && nd < cand[v]) {
                    cand[v] = nd;
                    if (!innext[v]) { innext[v] = 1; next[ncount++] = v; }
                }
            }
        }

        /* --- THE THROW: one row read straight across, all heaps in one
         * breath.  Each heap keeps only its shortest sliver as the new
         * ember; the rest were already burned.  No heap is compared to
         * any other heap.  Heaps that caught nothing are sealed: their
         * runners stop going out.                                    */
        for (int k = 0; k < ncount; k++) {
            const int v = next[k];
            dist_out[v] = cand[v];
            cand[v]     = INFINITY;
            innext[v]   = 0;
            frontier[k] = v;
        }
        spent += (size_t)ncount;
        fcount = ncount;
    }

    /* ---- REGIME 3: the land defeated the runners -> heap Dijkstra ---- */
    if (bailed) {
        HItem *heap = (HItem *)malloc(((size_t)m + 2) * sizeof(HItem));
        if (heap) {
            heap_dijkstra(n, m, off, edst, ew, source, dist_out, innext, heap);
            free(heap);
        } else {
            double *key = (double *)malloc((size_t)n * sizeof(double));
            if (key) {
                hooded_scan(n, off, edst, ew, source, dist_out, key);
                free(key);
            }
        }
    }

    free(cand); free(frontier); free(next); free(innext);
    free(off);  free(edst);     free(ew);
}
```

**The four revisions** (reasoned, not measured — see MEASUREMENT):

1. **Materialized pile → folded minimum.** A literal per-heap stick pile is O(m) memory and a second pass; the native burns longer slivers *as they land*, so the pile collapses to one running minimum `cand[v]`. Zero semantic change, O(n) memory.
2. **Full-land sweep → touched-list throw.** "Every heap throws" costs `rounds × n`. Heaps with no sticks throw nothing, so reading them is wasted breath; the runners name the heaps they fed. Kills the `R·n` term — a chain graph goes from O(n²) to O(n).
3. **Regime detector + hooded figure.** The brief names two regimes; the metaphor names them too. `m·12 > n²` (or `n < 32`) routes to the SIMD row-scan O(n²) Dijkstra, where "one sliver from every heap in the same breath" is literally `_mm256_cmp_pd` + `blendv`.
4. **Work budget + heap bail.** The mechanism's own stated risk is a land where embers keep re-lighting. Budget = 8m + 16n relaxation attempts, then restart with heap Dijkstra. The waste is ~8 streaming passes ≈ <10% of one heap-Dijkstra, so the guarded worst case is ~1.1×, not unbounded.

## 5. PREDICTION

**PREDICTION: speedup_vs_heap = 2.5**

Reasoning: on sparse graphs the throw loop does ~2–4 streaming passes over the edge arrays with no pointer-chasing and no `log n` sift per push, replacing ~m heap operations that are latency- and branch-bound (I expect 2–3.5× there). On dense graphs the SIMD row-scan replaces `m log n` heap work with `n²/4` vector work (I expect 4–10× there). Weighted over an unknown benchmark mix, 2.5×. Below 1.0× would mean the benchmark is dominated by a many-re-improvement regime and my budget bail is firing.

## 6. MEASUREMENT

**Not measured in this session — and I will not dress that up.** No tools were available: `dijkstra_bench` and `dijkstra_contract` could not be called, so the prediction above is an unverified forecast and the four revisions are reasoned design changes, not measured ones. What I can assert without the bench:

- **Correctness** is structural, not empirical. The throw loop is synchronous Bellman-Ford over an active frontier: it terminates when no candidate is accepted, at which point `dist[v] = min_u (dist[u] + w_uv)` holds for every edge — the Bellman fixpoint, unique for non-negative weights, hence bitwise-identical to Dijkstra's result. Unreached nodes are never written and stay `INFINITY`. The strict `<` test makes zero-weight edges and zero-weight cycles safe. Both fallbacks are textbook Dijkstra.
- **Unverified specifics** the bench must decide: whether the AVX2 gather path (degree ≥ 16) helps or hurts on this machine; whether `m·12 > n²` is the right regime crossover; whether the 8m+16n budget ever fires and if so whether it fires too late.

Anything reported as a measured speedup here would be fabricated.

## 7. VERDICT

The native's throw is a real algorithm and it is a *known good* one: dissolving the global argmin into n independent local minima, computed in one synchronized land-wide pass, is frontier-based synchronous Bellman-Ford — the standard parallel/GPU SSSP formulation. The metaphor genuinely breaks the target assumption (no place is ever compared against another place; there is no priority structure; runners launch from tentative embers), and it breaks it in the direction that real high-performance SSSP code has independently gone. That is the strongest outcome available: the strange world's method converged on validated practice rather than on a novelty.

Two honest debits.

1. **The native's garrison is over-confident.** "Its ember will not shorten again no matter what still arrives" is false, with a 4-node counterexample given above. I kept the sound half (unchanged ember ⇒ stop sending runners) and dropped the boast. If I had shipped the literal seal, it would have produced wrong distances on a graph as small as four nodes.
2. **Known failure condition, addressed rather than waved at.** The throw loop is worse than heap Dijkstra when (a) the land is crisscrossed — many rounds each re-treading a near-complete edge set, and (b) embers re-light many times, inflating total relaxations past `m log n`. Both are guarded in code, not in prose: (a) by the `m·12 > n²` pre-check routing to the SIMD row-scan, (b) by the `8m + 16n` work budget restarting into heap Dijkstra. Ceiling of the guarded bad case ≈ 1.1× the known way.

Thread parallelism was deliberately not shipped. The runner phase is genuinely parallel-safe — `dist_out` is read-only for a whole throw, which is exactly what SEED 2's synchronization buys you — but atomic-min on doubles plus atomic frontier appends do not pay off until a single round carries on the order of 10⁶ edges, and I could not confirm the benchmark reaches that. Per the instruction, SIMD first: gather-relaxation for hub nodes, and a genuine vector min-reduce in the dense path.

If the measurement comes back under 1.0×, the diagnosis is pre-determined and falsifiable: the budget fired, meaning the benchmark's graphs re-light embers more than ~8 times on average, and the correct response is to lower the budget toward 3m so the bail is nearly free — not to defend the throw.