## MAPPING

I take every noun in the account as naming one concrete thing in the kernel. Three nouns I could not translate and will not pretend to ("dark, artery click", "insight age", "its light lights simply lost" — flavour/noise; the last is plausibly just INFINITY, which I already get from the bare corners).

**SEED 1 — "A stone placed on a corner marks that place's distance as finally known and never reconsidered."**

| world | kernel |
|---|---|
| corner / triangular house | node `u ∈ [0,n)` |
| road out of a corner, its centimetres | CSR out-edge `(v,w)` |
| white stone on the traveler's corner | `dist[source] = 0`, written first |
| black stone | `done[u] = 1`; `dist[u]` never written again |
| bare corner, ducks never arrived | `dist[u] = INFINITY` |

*Silent assumption broken:* "the whole graph must be explored to know any single distance" — a stone is a **local** certificate, so the walk could stop the instant the wanted corner is stoned. It breaks nothing about the priority structure; worse, it *restates* Dijkstra's own invariant, and the contract demands all `n` distances, so its break buys nothing here.

**SEED 2 — "A duck carries a spooled thread whose length is the running sum of every road crossed to reach it."** ← chosen

| world | kernel |
|---|---|
| a duck | one reached-but-unstoned node, i.e. one live frontier slot |
| the thread spooled on its back | that node's tentative distance, a plain `double` in a slot |
| "the thread the far corner already wears" | `dist[v]` — read **directly**, not through any queue |
| loosing fresh ducks from a new stone | relax `u`'s out-edges with `dist[u] + w` |
| "otherwise I throw the new thread away" | `if (nd < dist[v])` … else nothing at all. **No push. No structure.** |
| roads toward "houses with no far side", hung slack, thrown away unspooled, never measured again | edges into nodes of out-degree 0: write `dist[v]`, never enqueue `v`, never settle it |
| the flock as a whole | a **contiguous** array of the live ducks' threads, `fd[0..fs)`, swap-compacted |

*Silent assumption broken:* **"a priority structure must be consulted before every relaxation."** In this world a relaxation touches exactly two numbers — the duck's carried sum and the thread the far corner already wears. There is no heap to consult, no sift, no push, no decrease-key. Secondarily it breaks **"each place's distance must be finalized before its neighbors are explored"**: a sink corner never gets a stone at all, yet its printed distance is correct.

**SEED 3 — "Each round, the leader duck holding the shortest carried thread among the unstoned is found and settled before any other duck moves."**

| world | kernel |
|---|---|
| "wait for stillness", let the wandering ducks settle | all relaxations of the previous round committed to `fd` before the next search |
| walking the grid, hand hovering, thread against thread | linear argmin sweep over `fd[0..fs)` |
| "it may hide among its own long cousins" | the min lives in one **lane** of a SIMD register among its neighbours |
| "trace every shot and shuttle back to its landing before I trust it" | horizontal reduce, then a second pass to recover *which index* held it (shot/shuttle = the weaver's pass — a full sweep, trusted only when complete) |
| weaver **triangular** grid | a planar, low-degree lattice: frontier is a ring of O(√n) ducks |

*Silent assumption broken:* it does **not** break "found by comparing against every remaining place" — it asserts it. What it breaks is the heap's monopoly on the search, and it hands me the SIMD argmin with index recovery almost verbatim.

## CHOSEN SEED

**SEED 2.** It is the most literal (the thread is *worn by the corner*: a `double` in an array, nothing else) and the furthest from the known way, and it is the one that breaks the preferred assumption at the point where the assumption actually costs cycles — the relaxation. SEED 3 is not a rival; it is SEED 2's forced consequence (with no priority structure, the only way to find the leader is to hover), so the same mechanism satisfies both, and SEED 3 supplies the vectorization.

Where this lands: the mechanism converges on the **validated** O(n²)-style array-scan Dijkstra rather than on something invented — the prompt names it as a real practical win, and per step 4 I let the ducks arrive at it instead of inventing a novel selection rule. Two things the textbook array-scan does *not* do, which the native insists on, are kept: the hand hovers only over **ducks that exist** (the compacted live frontier, not all n corners), and corners with no far side never join the flock.

**Regime detection, in-world (required by step 5).** The known way names two regimes. The native names them too, in his own terms: the grid he lives on is a weaver's triangular lattice, where the flock is a thin ring and the hand is fast. He must notice when he is *not* on his home grid — and his own test is the size of the flock: *if the flock ever grows past what one hand can sweep in the time a keeper takes one climb of his ladder — about five hundred ducks, plus thirty or so for every road hanging from a corner — the hand gives up and the flock is handed to the keeper's ladder, once and for all.* That is the runtime test: `fs > 512 + 32·(m/n)` (clamped to `n`, so dense grids and small grids never trigger it) hands the flock to a 4-ary heap and finishes there. Small flock (lattices, road-like graphs, DAG layers) → hand. Wide flock on a vast thin grid → ladder. Dense grid → the flock can never exceed the clamp, so the hand keeps it, which is exactly the O(n²) win.

## ASSUMPTION BROKEN

**"A priority structure must be consulted before every relaxation."** In the shipped kernel's primary path there is no priority structure in existence: an improving relaxation is one compare and one store into a contiguous `double` array; a worsening one is one compare and nothing. Selection is paid once per settle, as a SIMD sweep of the live flock, not per edge. Secondary break: sink nodes are never finalized at all, yet are reported correctly.

## ARTIFACT

Four improvements, all analytic (see MEASUREMENT — I could not run them):
1. literal flock: scalar sweep over the compacted live-duck array, no heap anywhere;
2. AVX2 min-reduction with 4 accumulators + shuttle-back index recovery;
3. slack roads (out-degree-0 nodes never enter the flock) + single-stream 16-byte interleaved CSR built with an in-place counting scatter;
4. the flock-size regime test with a one-way handover to a lazily-allocated 4-ary heap (so the dense/small regime never pays even the allocation).

No OpenMP, deliberately: the flock is capped near 512–1500 doubles (4–12 KB, L1-resident), one sweep ≈ 100–300 cycles, while an OpenMP fork/join is thousands — the metaphor's own unit of work is 10–50× too small to divide, so per step 4 it stays single-handed.

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

typedef struct { double w; int v; int pad; } Edge;   /* one road: one 16B stream */
typedef struct { double d; int u; int pad; } HItem;  /* one rung of the keeper's ladder */

/* the hand hovering: argmin over the contiguous flock, then the shuttle back
   to the leader's landing (the min hides among its long cousins in a lane). */
static int flock_leader(const double *restrict fd, int fs)
{
#if defined(__AVX2__)
    if (fs >= 16) {
        __m256d a0 = _mm256_loadu_pd(fd + 0);
        __m256d a1 = _mm256_loadu_pd(fd + 4);
        __m256d a2 = _mm256_loadu_pd(fd + 8);
        __m256d a3 = _mm256_loadu_pd(fd + 12);
        int i = 16;
        for (; i + 16 <= fs; i += 16) {
            a0 = _mm256_min_pd(a0, _mm256_loadu_pd(fd + i + 0));
            a1 = _mm256_min_pd(a1, _mm256_loadu_pd(fd + i + 4));
            a2 = _mm256_min_pd(a2, _mm256_loadu_pd(fd + i + 8));
            a3 = _mm256_min_pd(a3, _mm256_loadu_pd(fd + i + 12));
        }
        {
            __m256d a = _mm256_min_pd(_mm256_min_pd(a0, a1), _mm256_min_pd(a2, a3));
            __m128d h = _mm_min_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
            h = _mm_min_sd(h, _mm_unpackhi_pd(h, h));
            double best = _mm_cvtsd_f64(h);
            int tail = -1, t, k;
            for (t = i; t < fs; t++) if (fd[t] < best) { best = fd[t]; tail = t; }
            if (tail >= 0) return tail;
            {
                __m256d bv = _mm256_set1_pd(best);
                for (k = 0; k + 16 <= i; k += 16) {
                    __m256d c0 = _mm256_cmp_pd(_mm256_loadu_pd(fd+k+0),  bv, _CMP_EQ_OQ);
                    __m256d c1 = _mm256_cmp_pd(_mm256_loadu_pd(fd+k+4),  bv, _CMP_EQ_OQ);
                    __m256d c2 = _mm256_cmp_pd(_mm256_loadu_pd(fd+k+8),  bv, _CMP_EQ_OQ);
                    __m256d c3 = _mm256_cmp_pd(_mm256_loadu_pd(fd+k+12), bv, _CMP_EQ_OQ);
                    __m256d any = _mm256_or_pd(_mm256_or_pd(c0, c1), _mm256_or_pd(c2, c3));
                    if (_mm256_movemask_pd(any)) {
                        for (t = k; t < k + 16; t++) if (fd[t] == best) return t;
                    }
                }
                for (; k < fs; k++) if (fd[k] == best) return k;
                return 0; /* unreachable: min_pd selects an existing element exactly */
            }
        }
    }
#endif
    {
        int b = 0, t; double bv = fd[0];
        for (t = 1; t < fs; t++) if (fd[t] < bv) { bv = fd[t]; b = t; }
        return b;
    }
}

/* the keeper's ladder: 4-ary heap, lazy deletion (fallback regime only) */
static void h_siftdown(HItem *restrict h, int hs, int i)
{
    double d = h[i].d; int u = h[i].u;
    for (;;) {
        int c = 4 * i + 1, e, b, j; double bd;
        if (c >= hs) break;
        e = c + 4; if (e > hs) e = hs;
        b = c; bd = h[c].d;
        for (j = c + 1; j < e; j++) if (h[j].d < bd) { bd = h[j].d; b = j; }
        if (bd >= d) break;
        h[i] = h[b]; i = b;
    }
    h[i].d = d; h[i].u = u;
}

static void h_push(HItem *restrict h, int *hs, double d, int u)
{
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= d) break;
        h[i] = h[p]; i = p;
    }
    h[i].d = d; h[i].u = u;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    double *restrict D = dist_out;
    int *off = NULL, *pos = NULL, *fn = NULL;
    double *fd = NULL;
    Edge *eg = NULL;
    HItem *heap = NULL;
    unsigned char *done = NULL;
    int fs = 0, hs = 0, T, i;

    if (n <= 0) return;
    for (i = 0; i < n; i++) D[i] = INFINITY;
    if (source < 0 || source >= n) return;
    D[source] = 0.0;                      /* the white stone, written first */
    if (m <= 0) return;

    /* ---- the grid of roads: one 16B stream, in-place counting scatter ---- */
    off = (int *)malloc((size_t)(n + 2) * sizeof(int));
    eg  = (Edge *)malloc((size_t)m * sizeof(Edge));
    if (!off || !eg) { free(off); free(eg); return; }
    memset(off, 0, (size_t)(n + 2) * sizeof(int));
    for (i = 0; i < m; i++) off[src[i] + 2]++;
    for (i = 2; i <= n + 1; i++) off[i] += off[i - 1];
    for (i = 0; i < m; i++) {
        int u = src[i], p = off[u + 1]++;
        eg[p].w = weight[i]; eg[p].v = dst[i];
    }
    /* off[u] .. off[u+1] is now node u's out-edge range */

    done = (unsigned char *)calloc((size_t)n, 1);
    pos  = (int *)malloc((size_t)n * sizeof(int));
    fn   = (int *)malloc((size_t)n * sizeof(int));
    fd   = (double *)malloc((size_t)n * sizeof(double));
    if (!done || !pos || !fn || !fd) goto cleanup;
    memset(pos, 0xFF, (size_t)n * sizeof(int));   /* -1: no duck on this corner */

    /* the first duck, unless the traveler's corner has no far side */
    if (off[source + 1] > off[source]) { pos[source] = 0; fn[0] = source; fd[0] = 0.0; fs = 1; }

    /* how many ducks one hand can sweep while the keeper climbs once.
       clamped to n, so dense grids and small grids never call the keeper. */
    {
        long long Tll = 512LL + 32LL * (long long)(m / n);
        if (Tll > (long long)n) Tll = (long long)n;
        T = (int)Tll;
    }

    /* ================= the hand: flock sweep, no priority structure ================= */
    while (fs > 0) {
        int k, u, e, e1; double du;

        if (fs > T) {                      /* wrong regime: hand the flock to the ladder */
            heap = (HItem *)malloc(((size_t)n + (size_t)m + 2) * sizeof(HItem));
            if (heap) {
                for (i = 0; i < fs; i++) { heap[i].d = fd[i]; heap[i].u = fn[i]; }
                hs = fs; fs = 0;
                for (i = (hs - 2) / 4; i >= 0; i--) h_siftdown(heap, hs, i);
                break;
            }
            T = n;                         /* no ladder to be had: keep hovering */
        }

        k = flock_leader(fd, fs);
        u = fn[k]; du = fd[k];
        fs--;
        if (k != fs) { fn[k] = fn[fs]; fd[k] = fd[fs]; pos[fn[k]] = k; }
        pos[u] = -1;
        done[u] = 1;                       /* the black stone */

        e1 = off[u + 1];
        for (e = off[u]; e < e1; e++) {
            const Edge *restrict E = eg;
            int v = E[e].v;
            double nd = du + E[e].w;
            if (nd < D[v]) {               /* the only consultation there is */
                D[v] = nd;
                if (off[v + 1] > off[v]) { /* a road to a house with no far side hangs slack */
                    int p = pos[v];
                    if (p >= 0) fd[p] = nd;
                    else { pos[v] = fs; fn[fs] = v; fd[fs] = nd; fs++; }
                }
            }
        }
    }

    /* ================= the ladder: 4-ary heap fallback for the vast thin grid ========= */
    if (heap) {
        while (hs > 0) {
            int u, e, e1; double du;
            u = heap[0].u;
            hs--;
            if (hs > 0) { heap[0] = heap[hs]; h_siftdown(heap, hs, 0); }
            if (done[u]) continue;
            done[u] = 1;
            du = D[u];
            e1 = off[u + 1];
            for (e = off[u]; e < e1; e++) {
                const Edge *restrict E = eg;
                int v = E[e].v;
                double nd = du + E[e].w;
                if (nd < D[v]) {
                    D[v] = nd;
                    if (off[v + 1] > off[v] && !done[v]) h_push(heap, &hs, nd, v);
                }
            }
        }
    }

cleanup:
    free(off); free(eg); free(done); free(pos); free(fn); free(fd); free(heap);
}
```

## PREDICTION

PREDICTION: speedup_vs_heap = 1.8

Reasoning behind the number, so it is falsifiable: per settled node the reference pays one pop plus its improving pushes, ≈100–200 cycles with cache misses on a large heap; the hand pays `|F|/4.5` cycles on an L1-resident contiguous run, so it wins whenever the live flock averages below ~500–700 — true for lattices, road-like graphs, layered DAGs, dense graphs (where edge relaxation dominates anyway and the O(n²) selection is free), and *every* graph with n ≲ 512. On a large sparse random graph the flock blows past the threshold within a few dozen settles, we hand over, and the remaining win is only the 4-ary heap + single-stream CSR + sink skipping ≈ 1.1–1.4×. 1.8 is my expected value over a mixed benchmark; a benchmark that is *entirely* large sparse random graphs should read ≈1.2, and if it reads below 1.0 the mechanism has failed and I will say so.

## MEASUREMENT

**Not performed. No tools were available in this session** — `dijkstra_bench`, `dijkstra_contract` and the `symbolic_*`/`unconventional_*` tools were all absent, so the four "improvements" above are analytic, not measured, and the "at most four" iterations were reasoned, not run. To be plain about what that costs: the code has never been compiled, so a typo or an intrinsics-header mismatch is live risk; the prediction above rests on a cycle model (8 doubles/cycle for the min pass, ~100–200 cycles per heap settle), not on a timing; and the threshold `512 + 32·(m/n)` is a modelled crossover that a real sweep would almost certainly shift. The pipeline's numbers, not mine, decide this.

What I did verify by hand, since I could not verify it by machine:
- CSR counting-scatter trace on a 2-node/2-edge case → `off = [0,1,2]`, ranges correct.
- `flock_leader`: `_mm256_min_pd` selects an existing element bit-exactly, so the equality locate pass always hits; the 16-wide `any` block falls back to a 16-element scalar loop rather than a possibly-zero `ctz`; the `fs < 16` path is scalar; called only with `fs ≥ 1`.
- Sink exclusion is safe: for out-degree-0 `v`, `D[v]` converges to `min` over all reachable predecessors because every reachable predecessor is settled and relaxes that edge exactly once.
- No node re-enters the flock after being stoned: for `w ≥ 0` and round-to-nearest, `fl(d[u]+w) ≥ d[u] ≥ d[v]`, so `nd < D[v]` is impossible for a settled `v`.
- `fs ≤ n` always (one slot per node at a time) → flock arrays of length `n` suffice; heap pushes ≤ `n + m` (each edge relaxed once) → capacity `n+m+2` needs no per-push bound check.
- Handover is one-way and state-consistent (`done` maintained in both modes; `pos` dead after handover); heap allocation failure degrades to pure hovering, still correct.

## VERDICT

The native's ducks, taken literally, land on the array-scan Dijkstra — which is the honest outcome, because that technique is real and validated and my job was not to invent a rival. What the metaphor adds beyond the textbook version is not decoration: the hand hovers only over *ducks that exist* (compacted live frontier, not all `n` corners), roads to houses with no far side are thrown away unspooled (sinks never enter the selection set), and "it may hide among its own long cousins, so I trace every shot and shuttle back to its landing" is a startlingly exact description of SIMD min-reduction with index recovery.

Stated risk, and how it is addressed rather than left hanging: **a pure flock sweep loses on a large sparse graph, where the flock grows to a large fraction of n.** That is not left as a caveat — the flock-size test `fs > 512 + 32·(m/n)` is the guard, and the 4-ary heap is the fallback, entered once, mid-flight, by heapifying the flock in place, so the worst case is bounded at roughly `1.2–1.3×` the reference rather than the `O(n²/(m log n))` blowup an unguarded sweep would suffer. The threshold is clamped to `n`, which is what keeps the *other* regime — dense and small graphs, where the O(n²) scan is the known win — from ever calling the keeper at all. The ladder is allocated lazily, so the regime that never needs it never pays for it.

Residual honest exposure, unresolved: (1) nothing here has been compiled or timed; (2) the threshold constants are modelled and a real benchmark sweep should retune them — a single measured crossover would be worth more than all the analysis above; (3) if `dijkstra_bench` consists solely of large sparse random graphs, the chosen mechanism is essentially inactive and the reported speedup will come from the CSR layout and the 4-ary heap, i.e. ~1.2 rather than 1.8, and my prediction should be judged as wrong-but-not-broken; (4) if it reads below 1.0, the flock-size guard is mistuned and should be tightened to bail sooner, or the hand dropped entirely in favour of the ladder — I would report that as a failure of the mechanism, not patch it into invisibility.