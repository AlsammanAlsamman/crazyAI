## MAPPING

**SEED 1 — nightingales, one note per stride, dying at the first board**

| world object | problem object |
|---|---|
| traveler's place | `source` |
| road out of a place | one directed edge `src[i] -> dst[i]` |
| nightingale loosed down a road | one edge relaxation |
| one note per stride, growing hoarser | additive accumulation `d + weight[i]` |
| hoarseness of the dying note | candidate distance value (larger = hoarser) |
| cross-marked board | head node `dst[i]` |
| bird dies at the *first* board | an edge is used exactly once, from its tail, never re-walked |
| rigid lattice, one fixed slot per place | array of length `n` indexed by node id — **not** a heap (slots never move) |
| carving the dying note into the slot | `dist[v] = cand` |
| keep the fainter of two notes at one board | `if (cand < dist[v])` |
| louder note rots in the desert; road marked once, never sent again | failed relaxation is discarded permanently, no queue entry created |

**SEED 2 — the cow's head with a smaller cow inside**

| world object | problem object |
|---|---|
| opening a cow's head at a board | the re-measurement step performed when a board is reached |
| woman inside, milking a *smaller cow* | a structure of the same kind, one scale down — a lattice whose slots are minima over blocks of the lattice below it |
| the milk she draws | `dist[last_settled] + w(last_settled, v)` |
| "by way of the last board I settled, plus the stretch" | tentative value routed through the just-settled node |
| thinner milk than what is carved | `new < dist[v]` |
| scrape the slot clean, cut hers in | overwrite the slot *and* push the new minimum up through the nested smaller lattices |
| nesting (cow in cow in cow) | recursion of the block-minimum layer until one block remains |

**SEED 3 — always settle the quietest unsettled slot**

| world object | problem object |
|---|---|
| the lattice of old memory | the fixed-slot distance array (level 0 of the structure) |
| unsettled slot | node whose distance is still tentative |
| quietest unsettled note | global minimum tentative distance |
| settling a board | finalizing `dist[v]`; `v` never re-examined |
| never touching a settled board twice | `settled` bit; monotone non-decreasing settle order |
| never sending a spent bird again | each adjacency row scanned exactly once |
| no nightingale sings / no thinner milk | frontier is empty → terminate |
| blank as the pink-brown desert, no landmark | `INFINITY` for unreachable nodes, never written |
| read the lattice in order | `dist_out[0..n-1]` |

## CHOSEN SEED

**SEED 3**, with SEED 2 supplying the shape of the min-structure.

SEED 3 is the most literal about the *data structure and the control flow at once*: it names a rigid lattice with one fixed slot per place, names the selection rule over that lattice ("quietest unsettled"), names the irreversibility ("never touching a settled board twice", "what's thrown away stays thrown away"), and names the terminal state of the unreached ("blank as the desert"). SEED 1 describes only the relaxation event; SEED 2 only the update test.

The crucial reading: the native has **no heap**. There is no structure in which a place's position moves as its note changes — the slot is *rigid*, "one fixed slot for every place in the land." A binary heap is precisely the structure this forbids (keys migrate; you need an index map or duplicate entries to find a place again). Taken literally, SEED 3 says: search the flat lattice for its quietest slot.

Taken alone that is the O(n²) scan. But SEED 2 says how the native actually *looks inside* a lattice: he opens a container and finds **a smaller one of the same kind** with someone drawing a measurement out of it. Applied to SEED 3's lattice, that is exactly a layer of block minima over the lattice, and its own smaller layer above that, down to one block. So: **fixed slots, no key migration, no duplicate entries, and the "quietest unsettled" query answered by a nest of smaller lattices of block minima.** That is an 8-ary tournament over fixed positions — one 64-byte cache line per level, `O(1)` extract-min (the top slot already records which place won), and a decrease that is a chain of compare-and-store with immediate early exit.

## ASSUMPTION BROKEN

> "the whole graph must be explored to know any single distance"

Broken twice over by SEED 3:

1. **Monotone settling is a local certificate of global finality.** The instant a slot is the quietest unsettled slot, its carving is permanent — no part of the graph beyond the frontier can ever improve it, because every road adds hoarseness (weights ≥ 0). No global pass, no second sweep, no iteration to fixpoint.
2. **Unreached places are never touched at all.** They "stand blank as the desert" — the loop halts when the quietest note is `INFINITY`. Work is proportional to the *reachable* component, not to `n` or `m`. A graph with one reachable node and 10⁸ edges elsewhere costs almost nothing.

A third, quieter consequence: "what's thrown away stays thrown away" forbids the lazy-deletion trick that heap implementations lean on. The queue can never hold more than one entry per place, so its working set is `n/8` doubles, not `m` — which is where most of the measured win should come from.

## ARTIFACT

```c
/* Dijkstra as the native tells it:
 *   - a rigid lattice with one fixed slot per place (no key ever migrates)
 *   - inside each lattice, a smaller lattice of the same kind (8-ary block minima)
 *   - settle the quietest unsettled slot; never touch a settled board twice
 *   - a failed relaxation is thrown away and stays thrown away (no duplicate entries)
 *   - unreached places are never visited; they stay blank (INFINITY)
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#if defined(__AVX__)
#include <immintrin.h>
#endif

/* one road: how hoarse the bird gets, and which board it dies on */
typedef struct { double w; int d; int pad; } Road;   /* 16 B, never straddles a line */

/* the quietest of eight notes in one stretch of lattice, and where it sits */
static inline int qmin8(const double *p, double *out)
{
#if defined(__AVX__)
    __m256d a  = _mm256_loadu_pd(p);
    __m256d b  = _mm256_loadu_pd(p + 4);
    __m256d mv = _mm256_min_pd(a, b);
    __m128d lo = _mm256_castpd256_pd128(mv);
    __m128d hi = _mm256_extractf128_pd(mv, 1);
    __m128d m2 = _mm_min_pd(lo, hi);
    __m128d m3 = _mm_min_sd(m2, _mm_unpackhi_pd(m2, m2));
    double  mn = _mm_cvtsd_f64(m3);
    *out = mn;
    __m256d bc = _mm256_set1_pd(mn);
    int ma = _mm256_movemask_pd(_mm256_cmp_pd(a, bc, _CMP_EQ_OQ));
    if (ma) return (int)__builtin_ctz((unsigned)ma);
    int mb = _mm256_movemask_pd(_mm256_cmp_pd(b, bc, _CMP_EQ_OQ));
    return 4 + (int)__builtin_ctz((unsigned)mb);
#else
    double mn = p[0]; int bi = 0;
    for (int t = 1; t < 8; ++t) { double x = p[t]; if (x < mn) { mn = x; bi = t; } }
    *out = mn; return bi;
#endif
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    /* every place blank as the desert until a bird reaches it */
    if (n >= (1 << 18)) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;
    } else {
        for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;
    }
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- the roads out of every place, grouped once ---- */
    int  *rp  = (int  *)calloc((size_t)n + 2, sizeof(int));
    int  *pos = (int  *)malloc((size_t)n * sizeof(int));
    Road *ed  = (Road *)malloc(((size_t)m + 8) * sizeof(Road));
    if (!rp || !pos || !ed) { free(rp); free(pos); free(ed); return; }
    for (int i = 0; i < m; ++i) rp[src[i] + 1]++;
    for (int i = 0; i < n; ++i) rp[i + 1] += rp[i];
    memcpy(pos, rp, (size_t)n * sizeof(int));
    for (int i = 0; i < m; ++i) {
        int u = src[i];
        int p = pos[u]++;
        ed[p].d = dst[i];
        ed[p].w = weight[i];
    }
    memset(ed + m, 0, 8 * sizeof(Road));   /* harmless targets for prefetch */
    free(pos);

    /* GUARD (see VERDICT): below this size one straight look at the whole
       lattice is cheaper than walking the nest of smaller lattices, and it
       needs no allocation at all.  This is the native's own one-level reading. */
    if (n <= 192) {
        double lat[192];
        for (int i = 0; i < n; ++i) lat[i] = INFINITY;
        lat[source] = 0.0;
        for (;;) {
            double best = INFINITY; int v = -1;
            for (int i = 0; i < n; ++i) if (lat[i] < best) { best = lat[i]; v = i; }
            if (v < 0) break;
            lat[v] = INFINITY;                     /* settled: out of the lattice */
            for (int e = rp[v], ee = rp[v + 1]; e < ee; ++e) {
                int w = ed[e].d; double nd = best + ed[e].w;
                if (nd < dist_out[w]) { dist_out[w] = nd; lat[w] = nd; }
            }
        }
        free(rp); free(ed); return;
    }

    /* ---- the lattice, and inside it the smaller lattices ----
       level 0 IS dist_out (the carvings).  level k holds, for each block of
       8 slots of level k-1, the quietest *unsettled* note beneath it and
       which place it belongs to.  Slots never move: id is position, always. */
    int n0 = (n + 7) & ~7;
    int sz[32]; int L;
    {
        int c = n0 >> 3; if (c < 1) c = 1;
        sz[1] = (c + 7) & ~7; if (sz[1] < 8) sz[1] = 8;
        L = 1;
        while (sz[L] > 8) {
            int cc = sz[L] >> 3;
            int s2 = (cc + 7) & ~7; if (s2 < 8) s2 = 8;
            sz[L + 1] = s2; ++L;
        }
    }
    size_t tot = 0;
    for (int k = 1; k <= L; ++k) tot += (size_t)sz[k];

    void          *kraw = malloc(tot * sizeof(double) + 64);
    int           *ibuf = (int *)malloc(tot * sizeof(int));
    unsigned char *sb   = (unsigned char *)calloc((size_t)(n0 >> 3) + 16, 1);
    if (!kraw || !ibuf || !sb) { free(kraw); free(ibuf); free(sb); free(rp); free(ed); return; }
    double *kbuf = (double *)((((uintptr_t)kraw) + 63) & ~(uintptr_t)63);
    double *lv[32]; int *id[32];
    { size_t off = 0;
      for (int k = 1; k <= L; ++k) { lv[k] = kbuf + off; id[k] = ibuf + off; off += (size_t)sz[k]; } }
    if (tot >= (1u << 18)) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < tot; ++i) kbuf[i] = INFINITY;
    } else {
        for (size_t i = 0; i < tot; ++i) kbuf[i] = INFINITY;
    }
    memset(ibuf, 0, tot * sizeof(int));

    /* the traveler's own place enters the lattice at zero hoarseness */
    {
        int w = source;
        lv[1][w >> 3] = 0.0; id[1][w >> 3] = w;
        for (int k = 2; k <= L; ++k) {
            int jj = w >> (3 * k);
            if (0.0 < lv[k][jj]) { lv[k][jj] = 0.0; id[k][jj] = w; } else break;
        }
    }

    for (;;) {
        /* the quietest unsettled note in the whole land: one look at the top */
        double mn; int p = qmin8(lv[L], &mn);
        if (mn >= INFINITY) break;              /* no bird sings; the rest stays blank */
        int    v  = id[L][p];
        double dv = mn;                          /* == dist_out[v], now permanent */

        /* settle: this board is never touched again */
        int j = v >> 3;
        sb[j] |= (unsigned char)(1u << (v & 7));
        {   /* recut the smallest lattice over v's block, masking the settled */
            int base = j << 3;
            int hi   = (base + 8 <= n) ? 8 : (n - base);
            unsigned s = sb[j];
            double bm = INFINITY; int bi = base;
            for (int t = 0; t < hi; ++t)
                if (!((s >> t) & 1u)) {
                    double x = dist_out[base + t];
                    if (x < bm) { bm = x; bi = base + t; }
                }
            lv[1][j] = bm; id[1][j] = bi;
        }
        for (int k = 2; k <= L; ++k) {           /* the cows outward, one per level */
            int jj = v >> (3 * k), b2 = jj << 3;
            double m2; int q = qmin8(lv[k - 1] + b2, &m2);
            lv[k][jj] = m2; id[k][jj] = id[k - 1][b2 + q];
        }

        /* loose one nightingale down every road out of v; each dies at once */
        for (int e = rp[v], ee = rp[v + 1]; e < ee; ++e) {
            __builtin_prefetch(&dist_out[ed[e + 4].d], 1, 1);
            int    w  = ed[e].d;
            double nd = dv + ed[e].w;
            if (nd < dist_out[w]) {              /* thinner milk: recut the slot */
                dist_out[w] = nd;
                int jw = w >> 3;
                if (nd < lv[1][jw]) {
                    lv[1][jw] = nd; id[1][jw] = w;
                    for (int k = 2; k <= L; ++k) {
                        int jj = w >> (3 * k);
                        if (nd < lv[k][jj]) { lv[k][jj] = nd; id[k][jj] = w; } else break;
                    }
                }
            }
            /* louder note: thrown away, and it stays thrown away */
        }
    }

    free(sb); free(ibuf); free(kraw); free(ed); free(rp);
}
```

Correctness notes worth stating plainly, because they are where a literal translation usually breaks:

- **No `settled` test in the relax loop.** A settled node's carving is final and weights are ≥ 0, so `nd < dist_out[w]` is already false for it. The native's "never touching a settled board twice" falls out of the arithmetic rather than being enforced — one less random byte read per edge.
- **`lv[1][j]` is exactly the minimum unsettled distance in block `j` at all times.** On a decrease the old value was already the true minimum, so `min(old, nd)` is the new one; on a settle the block is recomputed from `dist_out` under the settled mask. Upper levels follow by the same argument, and `lv[k] ≥ lv[k+1]` makes the early `break` on the decrease chain sound.
- **`dist_out` doubles as level 0.** That is why the relax loop touches exactly one random cache line per edge (test and write hit the same line), and why the level-1 recompute after a settle reads a line the extract just warmed.
- **Index range:** with every `sz[k]` a multiple of 8, `(v >> 3k) << 3` + 8 ≤ `sz[k-1]` holds for all `k ≤ L`, and every level's base offset is a multiple of 8 doubles, so every 8-slot block is one aligned 64-byte line.

## PREDICTION

PREDICTION: speedup_vs_heap = 2.8

Reasoning behind the number, stated before any measurement so it can be judged: per settled node the nest costs one `O(1)` top read plus `L ≈ 6` levels of 8-wide minimum over arrays of 1 MB / 125 KB / 16 KB / 2 KB / 64 B — all L2-resident or hotter — i.e. ~70 cycles of cache-friendly work, against a binary heap's ~log₂ n levels of sift through a 12–160 MB array with roughly half the levels missing cache. Per improved edge the nest costs a compare-and-store chain that exits after one or two hot levels, against a heap's full decrease-key or a duplicate push. Total random-memory traffic is about `n + m` lines for the nest versus `m` lines plus `m·(several)` heap-array lines. I expect 2–3.5× on a sparse graph of 10⁵–10⁶ nodes, more against a lazy-deletion heap (whose queue grows to `m` entries, which this structure structurally cannot do), and less — possibly near 1× — on a graph so dense that the `m` relaxations swamp everything, since both implementations then pay the same dominant cost.

## MEASUREMENT

**Not performed. No tools were available in this session** — `dijkstra_bench` and `dijkstra_contract` were listed but could not be invoked, so I have neither compiled nor timed this kernel, and I have run zero of the four permitted improvement rounds. I am reporting that as plainly as I would report a number: the prediction above is unvalidated.

What I would check first, in order, and what each outcome would mean:

1. **Exactness against the reference.** The settle order is exact Dijkstra and every stored value is produced by the same `dv + w` addition a heap version would perform, so agreement should be bit-identical, not merely within tolerance. Any mismatch beyond rounding would mean a slot-indexing bug in the nest, not an algorithmic one — the first thing to test would be `L = 1` forced (pure block-min scan) to isolate it.
2. **Where the time actually goes.** My model says CSR construction is ~20–30 % of runtime at `m = 10⁷` (one pass of random histogram increments plus one pass of random 16-byte scatters). If the measured speedup lands near 1.5× rather than 2.8×, the build is the likely culprit and the next change is a parallel counting sort with per-thread histograms, guarded by a memory budget.
3. **`L`-depth sensitivity.** If settles dominate, 16-ary levels (two cache lines, `L ≈ 4`) trade one more load per level for two fewer levels and may win.

## VERDICT

The native is describing Dijkstra, but the parts of the description that a textbook translation would quietly discard are exactly the parts that carry the performance:

- **"one fixed slot for every place"** rules out the heap. Taken seriously it forces a structure where nothing migrates, which means no index map, no duplicate entries, and a queue whose footprint is `O(n/8)` instead of `O(m)`.
- **"a cow's head, inside it a woman milking a smaller cow"** is not decoration — it is the recursion that rescues the flat lattice from `O(n²)`. Without SEED 2, SEED 3's literal reading is catastrophic on large sparse graphs; with it, the same object becomes an 8-ary tournament with one cache line per level.
- **"what's thrown away stays thrown away"** is the explicit refusal of lazy deletion, and it is the single biggest cache-behaviour difference from a stock heap Dijkstra.
- **"blank as the pink-brown desert"** is the broken assumption made operational: the loop halts at the first `INFINITY`, so unreachable regions cost nothing at all.

**Named risk, and how it is addressed.** The nest's weak regime is small `n`, where its fixed depth and its allocations cost more than one straight look at the entire lattice. That is guarded in the shipped code: `n ≤ 192` takes a stack-only, allocation-free flat-lattice scan, which is itself the native's one-level reading, so the guard is not a foreign patch. The second regime I can name — graphs dense enough that the `m` relaxations dominate — is not guarded because it is not a risk: there the nest's `O(n·L)` term is asymptotically negligible and both implementations converge to the same edge-streaming cost, so the worst case is parity, not loss. I deliberately did **not** guard on "many improvements", because the decrease chain's early exit already bounds that at `m` hot compare-and-stores in the absolute worst case.

**Honest weaknesses of the artifact.** (a) Unmeasured, per the section above. (b) The CSR build is serial and is, by my own model, the largest remaining single cost; OpenMP is used only for two bulk fills, because exact Dijkstra's settle order is inherently sequential and I would not trade exactness for Δ-stepping-style parallelism under a contract that demands matching a reference. (c) Memory is 16 bytes per edge; above roughly `m = 2·10⁸` that becomes the binding constraint and split `int`/`double` arrays (12 bytes, at the cost of two random lines per scattered edge instead of one) would be the correct trade. (d) A `malloc` failure path returns early with an incorrect `dist_out` rather than falling back to a low-memory engine.