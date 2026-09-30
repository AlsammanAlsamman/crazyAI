## MAPPING

### SEED 1 — "A runner walks each road carrying its notch and lays a new stick on the far heap equal to the home ember's length plus that notch."

| World object | Problem object |
|---|---|
| place / heap of sticks | node index `v ∈ [0,n)` |
| lit ember, "banked low but never out" | `dist[v]` finite; source's ember = `0.0` |
| cold ash, unlit heap | `dist[v] == INFINITY` |
| round wooden disc, one character | `source` |
| road | directed edge `u → v` |
| notch cut into the stick | `weight[e]` |
| runner | one relaxation attempt |
| "home already holds a lit ember" | `dist[u] < INFINITY` (only reached nodes emit) |
| new stick laid on the far heap | candidate `dist[u] + w` offered to `v` |
| "runners down every road **at once**" | edges relaxed in arbitrary order, no ordering structure |
| stick dropped by the wayside → fed to the hearth | rejected candidate discarded in place, zero allocation |

**Silent assumption broken:** *"a priority structure must be consulted before every relaxation"* and *"a road can only be considered once its starting place is fully settled."* A runner leaves on a merely-lit ember, never a settled one.

### SEED 2 — "Every heap throws its sticks together in one land-wide throw, and the row read straight across picks each heap's shortest sliver as its new ember while the rest are burned for fuel."

| World object | Problem object |
|---|---|
| a heap's pile of sticks | the multiset of candidate labels arriving at `v` during one round |
| **the throw** | one synchronous round barrier — a Bellman-Ford sweep, not a pop |
| "one throw for the **whole land at once**" | data-parallel commit over the whole label array; no per-node serialization |
| "read straight across a **single row**, one sliver from every heap **in the same breath**" | one vector lane per place; many places' minima resolved per instruction (SIMD row read) |
| whichever sliver is shortest becomes the ember | `dist[v] ← min(dist[v], candidates)` |
| every longer sliver **burned for fuel** | non-minimal candidates dropped in place — the min is *fused into the landing*, never stored |
| the winner kept as a **coal banked forward** | `dist[]` carried unchanged into the next throw |
| fortune-tellers at the pyramid | one global, order-free, simultaneous decision |

**Silent assumptions broken:** *"each place's distance must be finalized before its neighbors are explored"* — nothing is finalized during a throw — **and** *"the next place to finalize is found by comparing against every remaining place."* There is no "next place." The selection step is **deleted**: comparison is local to a heap and every heap does it in the same breath.

### SEED 3 — "A garrison seals a heap once its ember stops changing between throws, fixing its distance for good and halting its runners."

| World object | Problem object |
|---|---|
| garrison riding in | mark node inactive |
| "ember matched exactly what it held after the throw before" | label did not decrease in the last throw |
| **"its own runners stop going out"** | `v`'s out-edges are not scanned next round → **the frontier** |
| roads into garrisoned places "refuse entry" | relaxation into a settled node rejected |
| "every heap wears its garrison" | frontier empty ⇒ converged |
| "a heap's ash never once catches a spark across throw after throw" | node never reached ⇒ stays `INFINITY` |
| the hooded figure's final recitation, one pass | linear write-out of `dist_out[]` |

**Silent assumption broken:** *"the whole graph must be explored to know any single distance"* — work is confined to the places whose embers actually moved.

**Honest defect, stated up front:** the *permanence* of the seal is **unsound**. In a synchronous throw, a place's ember can sit unchanged for one throw and still shorten later, because an upstream neighbour improved during that same throw. Taken literally, permanent sealing would return wrong distances and violate the exact-match contract. I keep the sound half of the native's mechanism — *unchanged ⇒ runners stop this throw* (that is exactly the frontier, and it is correct) — and drop the permanence. I do **not** replace it with Dijkstra's finalization; the garrison can be dismissed and ride again.

## CHOSEN SEED

**Seed 2 — the land-wide throw and the row read straight across.**

It is the most literal (it names a computational object outright: a *row*, read across all heaps in one breath, with a per-heap min and in-place discard of the losers), and it is the furthest from binary-heap Dijkstra: no priority structure, no extraction, no settling order. Seeds 1 and 3 are its own machinery — the relaxation and the frontier — so I implement all three; they are one mechanism.

Seed 2 **does** break the preferred assumption ("the next place to finalize is found by comparing against every remaining place"), not by making that comparison cheaper but by removing the comparison entirely. Seed 3 also touches it; Seed 2 is chosen because its mapping lands on a concrete instruction-level object.

## ASSUMPTION BROKEN

Primary: **"the next place to finalize is found by comparing against every remaining place."** There is no selection at all — every place updates simultaneously.
Also broken: **"each place's distance must be finalized before its neighbors are explored"**, **"a priority structure must be consulted before every relaxation"**, **"a road can only be considered once its starting place is fully settled"**, and **"the whole graph must be explored to know any single distance."**

### The native's own regime test (required, since `known_way` names two regimes)

The description already contains both readings, and I implement both:

- *Before the first throw, the native counts the roads against the places.* If the land is a **thicket** — roads more numerous than the places multiplied by themselves and divided among four-and-twenty — or so small a single breath covers it, the runners would only trample one another. Then the hooded figure walks the whole land each throw and calls out the single dimmest unsealed ember. → **AVX2 row-reading argmin array-scan Dijkstra, O(n²/4 + m), no heap.** This is the "row read straight across" applied to the *embers* instead of the *slivers*, and it is the acknowledged dense win.
- *And if, throw after throw, the runners wear out more sandals than there are roads four times over and heaps still go unsealed* — a long, thin land where each throw lights barely a spark — the native calls the runners home and hands every lit ember to the hooded figure's ordered walk. → **relaxation budget `4m + 8n`; on exceedance, seed a binary heap with every finite label and finish as Dijkstra.** This bounds the worst case at roughly `4m + (n+m)log n`, so the mechanism can never blow up on high-diameter graphs.

Thread parallelism is deliberately **not** used in the throws. The metaphor's "runners at once" is inviting, but the ember writes are a scatter-min: unsynchronized threads lose updates and break exact match, and atomic CAS-min at the real benchmark frontier sizes costs more than it buys. Per the instruction, I stopped at vectorization hints (`restrict`, one 16-byte interleaved road stream, explicit AVX2 argmin, unlikely-branch hint).

## ARTIFACT

Four improvements, in order, all reasoned on paper:
1. Literal `cand[]` "sticks lying on heaps" + a full-land commit sweep each throw → cost `n` per throw, discarded.
2. Fuse the per-heap min **into the landing** ("the heap compares each stick as it lands") — same minimum, no `cand[]`, no full-land sweep; frontier compaction via a per-throw stamp so each heap appears once.
3. Interleave far-heap and notch into one 16-byte `Road` (one memory stream, both operands on one cache line); `restrict` on every hot pointer; improvement branch marked unlikely.
4. Regime recognition: thicket → AVX2 argmin array scan; sandal budget → handover to the ordered heap walk.

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif

/* A road as the runner carries it: the far heap and the notch in ONE 16-byte
   object, so both arrive together on a single cache line, one stream. */
typedef struct { double w; int v; int pad; } Road;

/* The hooded figure's ordered walk -- only for the thicket aid and the
   long-thin-land handover, never for the throws themselves. */
typedef struct { double d; int u; } HeapItem;

static void hpush(HeapItem *h, int *hs, double d, int u)
{
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}

static HeapItem hpop(HeapItem *h, int *hs)
{
    HeapItem top = h[0];
    int sz = --(*hs);
    if (sz > 0) {
        HeapItem last = h[sz];
        int i = 0;
        for (;;) {
            int l = 2 * i + 1;
            if (l >= sz) break;
            int r = l + 1;
            int s = (r < sz && h[r].d < h[l].d) ? r : l;
            if (h[s].d >= last.d) break;
            h[i] = h[s];
            i = s;
        }
        h[i] = last;
    }
    return top;
}

/* "read straight across a single row, one sliver from every heap in the same
   breath" -- applied to the embers: the dimmest unsealed ember in the land,
   four lanes per instruction. Returns -1 when the whole land is cold ash. */
static int land_argmin(const double * restrict key, int n)
{
    int best = -1;
    double bv = INFINITY;
#if defined(__AVX2__)
    if (n >= 8) {
        const __m256i lane = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256d vmin = _mm256_set1_pd(INFINITY);
        __m256i vidx = _mm256_set1_epi64x(-1);
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d v  = _mm256_loadu_pd(key + i);
            __m256d lt = _mm256_cmp_pd(v, vmin, _CMP_LT_OQ);
            __m256i id = _mm256_add_epi64(_mm256_set1_epi64x((long long)i), lane);
            vmin = _mm256_min_pd(vmin, v);
            vidx = _mm256_blendv_epi8(vidx, id, _mm256_castpd_si256(lt));
        }
        {
            double mv[4]; long long mi[4];
            int k;
            _mm256_storeu_pd(mv, vmin);
            _mm256_storeu_si256((__m256i *)mi, vidx);
            for (k = 0; k < 4; k++)
                if (mv[k] < bv) { bv = mv[k]; best = (int)mi[k]; }
        }
        for (; i < n; i++)
            if (key[i] < bv) { bv = key[i]; best = i; }
        return best;
    }
#endif
    {
        int i;
        for (i = 0; i < n; i++)
            if (key[i] < bv) { bv = key[i]; best = i; }
    }
    return best;
}

/* ONE LAND-WIDE THROW.
   Every heap whose ember moved last throw sends its runners. Each stick is
   compared against the far heap's own ember as it lands (the per-heap minimum,
   fused into the landing -- losers are burned, never stored). A heap enters the
   next throw at most once, stamped. Heaps whose embers did not move send
   nothing: their garrison stands, dismissible. */
static int one_throw(const int * restrict off, const Road * restrict R,
                     double * restrict dist, int * restrict stamp,
                     const int * restrict frontier, int fn,
                     int * restrict nextf, int round, long long * restrict work)
{
    int nn = 0;
    int k;
    for (k = 0; k < fn; k++) {
        int u  = frontier[k];
        double du = dist[u];
        int e0 = off[u], e1 = off[u + 1], e;
        *work += (long long)(e1 - e0);
        for (e = e0; e < e1; e++) {
            int v = R[e].v;
            double nd = du + R[e].w;
            if (UNLIKELY(nd < dist[v])) {
                dist[v] = nd;
                if (stamp[v] != round) { stamp[v] = round; nextf[nn++] = v; }
            }
        }
    }
    return nn;
}

void kernel(int n, int m, const int *src, const int *dst,
            const double *weight, int source, double *dist_out)
{
    const int    * restrict SRC = src;
    const int    * restrict DST = dst;
    const double * restrict W   = weight;
    double       * restrict dist = dist_out;
    int *off, *cur;
    Road *R;
    int i, thicket;

    if (n <= 0) return;
    for (i = 0; i < n; i++) dist[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist[source] = 0.0;                    /* the ember banked nearly out, never out */
    if (m <= 0) return;

    /* ---- lay out the roads leaving each place (CSR, counting sort) ---- */
    off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    cur = (int *)malloc((size_t)(n + 1) * sizeof(int));
    R   = (Road *)malloc((size_t)m * sizeof(Road));
    if (!off || !cur || !R) { free(off); free(cur); free(R); return; }
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (i = 0; i < m; i++) off[SRC[i] + 1]++;
    for (i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)(n + 1) * sizeof(int));
    for (i = 0; i < m; i++) {
        int u = SRC[i];
        int p = cur[u]++;
        R[p].v = DST[i];
        R[p].w = W[i];
        R[p].pad = 0;
    }
    free(cur);

    /* ---- the native counts the roads against the places ----
       a thicket (or a land one breath wide): runners would trample each other,
       so the hooded figure reads the whole land each throw instead. */
    thicket = (n <= 256) || ((double)m * 24.0 >= (double)n * (double)n);

    if (thicket) {
        double *key    = (double *)malloc((size_t)n * sizeof(double));
        char   *sealed = (char *)calloc((size_t)n, 1);
        int it;
        if (!key || !sealed) { free(key); free(sealed); free(off); free(R); return; }
        for (i = 0; i < n; i++) key[i] = dist[i];
        for (it = 0; it < n; it++) {
            int u = land_argmin(key, n);
            int e, e1;
            double du;
            if (u < 0) break;              /* only cold ash left: forever unlit */
            key[u] = INFINITY;             /* the garrison seals this heap */
            sealed[u] = 1;
            du = dist[u];
            e1 = off[u + 1];
            for (e = off[u]; e < e1; e++) {
                int v = R[e].v;
                double nd = du + R[e].w;
                if (nd < dist[v] && !sealed[v]) { dist[v] = nd; key[v] = nd; }
            }
        }
        free(key); free(sealed); free(off); free(R);
        return;
    }

    /* ---- the throws ---- */
    {
        int *frontier = (int *)malloc((size_t)n * sizeof(int));
        int *nextf    = (int *)malloc((size_t)n * sizeof(int));
        int *stamp    = (int *)malloc((size_t)n * sizeof(int));
        int fn, round = 0, handover = 0;
        long long work = 0;
        long long budget = 4LL * (long long)m + 8LL * (long long)n;

        if (!frontier || !nextf || !stamp) {
            free(frontier); free(nextf); free(stamp); free(off); free(R); return;
        }
        for (i = 0; i < n; i++) stamp[i] = 0;
        fn = 0;
        frontier[fn++] = source;

        while (fn > 0) {
            int nn;
            int *t;
            round++;
            nn = one_throw(off, R, dist, stamp, frontier, fn, nextf, round, &work);
            t = frontier; frontier = nextf; nextf = t;
            fn = nn;
            /* more sandals worn than four times the roads, heaps still unsealed:
               a long thin land. Call the runners home. */
            if (work > budget && fn > 0) { handover = 1; break; }
        }

        if (handover) {
            char *sealed  = (char *)calloc((size_t)n, 1);
            HeapItem *heap = (HeapItem *)malloc(((size_t)n + (size_t)m + 2) * sizeof(HeapItem));
            if (sealed && heap) {
                int hs = 0;
                for (i = 0; i < n; i++)
                    if (dist[i] < INFINITY) hpush(heap, &hs, dist[i], i);
                while (hs > 0) {
                    HeapItem top = hpop(heap, &hs);
                    int u = top.u, e, e1;
                    double du;
                    if (sealed[u]) continue;
                    if (top.d > dist[u]) continue;
                    sealed[u] = 1;
                    du = dist[u];
                    e1 = off[u + 1];
                    for (e = off[u]; e < e1; e++) {
                        int v = R[e].v;
                        double nd = du + R[e].w;
                        if (nd < dist[v]) { dist[v] = nd; hpush(heap, &hs, nd, v); }
                    }
                }
            } else {
                /* no memory for the ordered walk: keep throwing to convergence.
                   Slower, still exact. */
                while (fn > 0) {
                    int nn, *t;
                    round++;
                    nn = one_throw(off, R, dist, stamp, frontier, fn, nextf, round, &work);
                    t = frontier; frontier = nextf; nextf = t;
                    fn = nn;
                }
            }
            free(sealed); free(heap);
        }

        free(frontier); free(nextf); free(stamp);
    }

    free(off); free(R);
    /* dist_out already holds the whole land's embers, read off in one pass. */
}
```

**Why the seeded handover is exact:** every label in `dist[]` at handover is the length of a real path from the source (labels only ever descend by `dist[u] + w`), so every label is an upper bound on the true distance; `dist[source] = 0` is exact; and *every* finite-label node is pushed. Those are precisely Dijkstra's preconditions with non-negative weights, so the ordered walk completes correctly from the partial state. Unreachable nodes are never pushed and keep `INFINITY`.

**Why sealing in the thicket path is exact:** `key[u] = INFINITY` plus `sealed[u]` is Dijkstra's own finalization (extract-min over the array), not the native's unsound unchanged-ember seal. In the throws, the garrison only stops runners for one throw and is dismissed the moment a shorter stick lands.

## PREDICTION

PREDICTION: speedup_vs_heap = 2.5

Decomposed, so it can be falsified separately:
- sparse random graph (avg degree 4–16, n ≳ 10⁵): **2–3×**. The throws do ~2–3× as many relaxations as Dijkstra, but each is a sequential stream over one 16-byte array with no priority structure, against Dijkstra's ~n cache-missing sift chains.
- dense / small (`24m ≥ n²` or `n ≤ 256`): **8–20×**. Array scan is `n²/4` SIMD-lane comparisons against `m ≈ n²` heap pushes at `log n` each.
- grid / road-like, high diameter: **1.0–1.3×**, floored by the sandal budget handover; this is the regime where I expect the mechanism to have no edge at all.

## MEASUREMENT

**Not measured. I could not measure it.** `dijkstra_bench` and `dijkstra_contract` were listed but no tools are callable in this session, so no number here is mine. I am reporting that plainly rather than inventing a measurement: the prediction above stands unverified, and the pipeline's numbers are the first real evidence that will exist.

What I would look at first, and what would falsify me:
- If sparse speedup < 1.0, the throws are doing far more than 3× Dijkstra's relaxations — check whether `work / m` at convergence exceeds ~4 (then the budget is firing and the handover is eating the gain, and the budget should rise or the frontier should be bucketed by distance).
- If the dense path is *slower*, the `24m ≥ n²` threshold is mis-set for the machine's cache; the crossover is where `n²/4 ≈ 3m`.
- If any distance mismatches the reference, suspect the seeded handover first, not the throws.

## VERDICT

The native's mechanism translates literally and completely: places are node indices, roads are CSR edges, embers are labels, cold ash is `INFINITY`, a runner is a relaxation, the throw is a synchronous round with a fused per-heap minimum, and the garrison is the frontier. Nothing in the description had to be reinterpreted as Dijkstra — the resulting kernel contains no priority structure in its main path, which is exactly what the seed promised.

Two things in the description are **wrong** and I say so rather than papering over them:
1. **Permanent sealing on an unchanged ember is unsound.** A label can hold still for one throw and shorten later. I kept the sound half (unchanged ⇒ runners rest this throw = the frontier) and dropped the permanence. This is the one place I did not follow the native literally, and the reason is correctness, not preference for the textbook.
2. **"Roads into garrisoned places refuse entry" buys nothing.** Under the sound reading, the `nd < dist[v]` test already rejects those sticks for free; an extra `sealed[v]` branch in the hot loop would cost more than it saves. Dropped, per the instruction not to ship a mechanism whose own risk I can't justify.

Stated risks, and how each is addressed rather than left as a caveat:
- *Risk: catastrophic on high-diameter graphs (O(n·m) worst case).* **Guarded** by the `4m + 8n` relaxation budget with a correctness-preserving handover to a seeded binary-heap Dijkstra, bounding the worst case at ~`4m + (n+m)log n` — at most a small constant factor worse than the reference, never unbounded.
- *Risk: overhead on dense or tiny graphs.* **Guarded** by the runtime thicket test (`n ≤ 256 || 24m ≥ n²`) selecting the SIMD argmin array scan, which is the known dense win and needs no heap.
- *Risk: thread races on ember writes corrupting exact match.* **Dropped** — no thread parallelism in the throws. Vectorization hints only, as instructed; the frontier at realistic benchmark sizes does not justify atomic scatter-min.

Honest overall assessment: I expect this to beat heap Dijkstra clearly in the two regimes the native's own test can recognize, and to be a wash in the third, where the budget makes it merely safe rather than fast. Whether the predicted 2.5× survives contact with `dijkstra_bench` I genuinely do not know, and I have not pretended to.

---

*Note on session setup: the `claude.ai` and `PubMed` MCP servers require authorization and are unavailable here — claude.ai connectors via your claude.ai connector settings, others via `claude mcp` or `/mcp` in an interactive session. Also, no shell was used this turn, so I could not check whether the `brainny` CLI is present; the ambient capture routines were skipped for that reason, not ignored.*