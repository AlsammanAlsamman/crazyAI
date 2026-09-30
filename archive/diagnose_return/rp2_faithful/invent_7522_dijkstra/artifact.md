# MAPPING

**SEED 1 — "A stone placed on a corner marks that place's distance as finally known and never reconsidered."**

| world object | problem object |
|---|---|
| corner / house | node index `u ∈ [0,n)` |
| white stone (artery click) | `dist[source] = 0`, written before anything else |
| black stone | permanent removal of `u` from the candidate set; `dist_out[u]` final |
| "never touched again" | no `done[]` test is even needed — a later `nd` can never beat a final `dist` |
| bare corner | `dist_out[v] == INFINITY`, never entered the flock |

*Assumption broken:* none. This seed **affirms** "each place's distance must be finalized before its neighbors are explored." It is the Dijkstra invariant itself, not a break.

**SEED 2 — "A duck carries a spooled thread whose length is the running sum of every road crossed to reach it."**

| world object | problem object |
|---|---|
| duck | an entry in a *compact* frontier array `cand[i]` |
| spooled thread length, carried *on the duck's back* | `cd[i]` — the tentative distance stored **next to the duck, contiguously**, not fetched from the house |
| "spooled to exactly that road's centimetres" | `nd = dist[u] + w(u,v)` |
| road hung slack, thrown away unspooled | edge whose `nd` fails `nd < dist[v]`: no structure is touched at all |

*Assumption broken:* **"a priority structure must be consulted before every relaxation."** A relaxation here is one `double` store into a flat array (plus, on first arrival, an append). Nothing is sifted, nothing is compared but the two thread lengths.

**SEED 3 — "Each round, the leader duck holding the shortest carried thread among the unstoned is found and settled before any other duck moves."**

| world object | problem object |
|---|---|
| "walk the grid with my hand hovering, comparing thread against thread" | one contiguous min-reduction over `cd[0..nc)` |
| "among all not yet stoned" | settled nodes are swap-removed from `cand`/`cd`, so the walk never passes them |
| "it may hide among its own long cousins, so I trace every shot and shuttle back to its landing before I trust it" | **no early exit**: complete the full vector reduction over all lanes, *then* resolve which lane/index holds the winner (SIMD horizontal reduce + index blend) |
| stillness / settle | the round boundary: one and only one node is finalized per walk |
| "roads toward houses with no far side" | `off[u]==off[u+1]`; also destinations already stoned, discarded unmeasured |

*Assumption broken:* **"a priority structure must be consulted before every relaxation"** (there is no priority structure at all) and, secondarily, "a road can only be considered once its starting place is fully settled" is *kept*, while the heap's amortization is discarded.

# CHOSEN SEED

**SEED 3**, with SEED 2 supplying its data layout. SEED 3 is the seed that breaks the preferred assumption, and it is the most literal: "hand hovering along the grid comparing thread against thread" is, word for word, a linear scan, and "trace every shot and shuttle back to its landing before I trust it" is word for word a full (non-early-exiting) vector reduction followed by index recovery.

Honesty note: this lands near the O(n²) array-scan Dijkstra the prompt names as a known practical win, so it is *not* maximally different from all known methods. But two things in the native's account are not the textbook O(n²) scan: (a) the hand walks **ducks**, not corners — the scan is over a compacted frontier carrying its own thread lengths, so it costs Σ|frontier| rather than n per round, and untouched corners are never walked past at all; (b) the walk is explicitly *complete-then-resolve*, which is exactly a 4-accumulator SIMD argmin. I build that, not the textbook loop.

# ASSUMPTION BROKEN

*A priority structure must be consulted before every relaxation.* In this kernel a relaxation is: one compare, one store to `dist_out`, one store to `cd`. Cost per edge is O(1) with no log factor and no pointer chasing — the entire log n is moved out of the m-sized inner loop and paid once per round instead, in the most cache- and SIMD-friendly shape that exists (a sequential array reduction).

Second regime (required, because the known_way names two): the native must *feel* when walking is the wrong trade. Encoded as: **"every so often I measure the length of my walk against the number of fresh threads I would have to lay; if my hand has hovered over far more ducks than there are threads to spool, I stop walking and pile the ducks into nested pens instead."** That is a windowed runtime comparison of mean frontier size against `~4·(1+m/n)·log₂n`; if the walk is losing, the flock is poured into a binary heap and the run *resumes* from the exact same invariant. Small grids (`n ≤ 4096`) never switch — walking is unconditionally cheap there.

# ARTIFACT

Which code implements which part of the metaphor:

- `dist_out[source]=0` before all else → the white stone's "dark, artery click… written first".
- `cand[]`/`cd[]`/`slot[]` → the **flock**: `cand[i]` is which corner duck *i* landed on, `cd[i]` is the thread on its back (SEED 2: the length travels with the duck, contiguously), `slot[v]` is where a corner's duck stands so a shorter thread can be re-spooled in O(1).
- `duck_leader()` → "walk the grid with my hand hovering… trace every shot and shuttle back to its landing before I trust it": 4×`__m256d` running minima with parallel index vectors, `_CMP_LT_OQ` + `blendv`, **no early exit**, then a 16-lane shuttle-back to recover the landing index.
- the swap-remove after the scan → "that corner gets a black stone… never touched again"; it is also why the hand never walks past a stoned corner.
- the relax loop `if (nd < dist_out[v])` → "only if that sum is shorter than the thread the far corner already wears; otherwise I throw the new thread away". Note this single test *also* silently discards every road into an already-stoned house — the native's "roads that lead nowhere useful… thrown away unspooled, never measured again" — because a stoned house's thread is already globally minimal.
- untouched `dist_out[v] == INFINITY` → "a corner the ducks never reach I leave bare… its light simply lost."
- `win_scanned / win_rounds` vs `walk_thresh` → the regime sense; `heap`/`hpush`/`hpop` → the "nested pens", entered only when the walk is provably losing, resumed from the same stones.
- No thread parallelism: the metaphor's unit of work per round is one frontier scan (microseconds at any size where the walk is still the right strategy), far below OpenMP barrier cost. Vectorization + cache layout (16-byte packed `DuckRoad` records: one stream, one load yields both `v` and `w`) only, per instruction.

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* one road: 16 bytes, single stream -> one load gives far side + centimetres */
typedef struct { double w; int v; int pad; } DuckRoad;
typedef struct { double d; int u; } HItem;

static void hpush(HItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) >> 1; if (h[p].d <= h[i].d) break;
        HItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HItem hpop(HItem *h, int *hs) {
    HItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    for (;;) { int l = 2*i+1, r = l+1, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HItem t = h[s]; h[s] = h[i]; h[i] = t; i = s; }
    return top;
}

/* THE WALK: hand hovering along the flock, thread against thread.
   No early exit -- every shot is traced, then we shuttle back to the landing. */
static inline int duck_leader(const double *restrict cd, int nc)
{
    int best = 0, i = 0;
    double bv = cd[0];
#if defined(__AVX2__)
    if (nc >= 16) {
        const __m256d INFV = _mm256_set1_pd(INFINITY);
        __m256d m0 = INFV, m1 = INFV, m2 = INFV, m3 = INFV;
        __m256d p0 = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
        __m256d p1 = _mm256_setr_pd(4.0, 5.0, 6.0, 7.0);
        __m256d p2 = _mm256_setr_pd(8.0, 9.0, 10.0, 11.0);
        __m256d p3 = _mm256_setr_pd(12.0, 13.0, 14.0, 15.0);
        __m256d k0 = p0, k1 = p1, k2 = p2, k3 = p3;
        const __m256d step = _mm256_set1_pd(16.0);
        for (; i + 16 <= nc; i += 16) {
            __m256d a0 = _mm256_loadu_pd(cd + i);
            __m256d a1 = _mm256_loadu_pd(cd + i + 4);
            __m256d a2 = _mm256_loadu_pd(cd + i + 8);
            __m256d a3 = _mm256_loadu_pd(cd + i + 12);
            __m256d c0 = _mm256_cmp_pd(a0, m0, _CMP_LT_OQ);
            __m256d c1 = _mm256_cmp_pd(a1, m1, _CMP_LT_OQ);
            __m256d c2 = _mm256_cmp_pd(a2, m2, _CMP_LT_OQ);
            __m256d c3 = _mm256_cmp_pd(a3, m3, _CMP_LT_OQ);
            m0 = _mm256_blendv_pd(m0, a0, c0); k0 = _mm256_blendv_pd(k0, p0, c0);
            m1 = _mm256_blendv_pd(m1, a1, c1); k1 = _mm256_blendv_pd(k1, p1, c1);
            m2 = _mm256_blendv_pd(m2, a2, c2); k2 = _mm256_blendv_pd(k2, p2, c2);
            m3 = _mm256_blendv_pd(m3, a3, c3); k3 = _mm256_blendv_pd(k3, p3, c3);
            p0 = _mm256_add_pd(p0, step); p1 = _mm256_add_pd(p1, step);
            p2 = _mm256_add_pd(p2, step); p3 = _mm256_add_pd(p3, step);
        }
        double vb[16], kb[16];
        _mm256_storeu_pd(vb,      m0); _mm256_storeu_pd(vb + 4,  m1);
        _mm256_storeu_pd(vb + 8,  m2); _mm256_storeu_pd(vb + 12, m3);
        _mm256_storeu_pd(kb,      k0); _mm256_storeu_pd(kb + 4,  k1);
        _mm256_storeu_pd(kb + 8,  k2); _mm256_storeu_pd(kb + 12, k3);
        bv = vb[0]; best = (int)kb[0];
        for (int t = 1; t < 16; t++) {
            int idx = (int)kb[t];
            if (vb[t] < bv || (vb[t] == bv && idx < best)) { bv = vb[t]; best = idx; }
        }
    }
#endif
    for (; i < nc; i++) if (cd[i] < bv) { bv = cd[i]; best = i; }
    return best;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- the grid of roads, in one stream ---- */
    int *off = (int *)malloc(((size_t)n + 1) * sizeof(int));
    int *cur = (int *)malloc((size_t)n * sizeof(int));
    DuckRoad *E = (DuckRoad *)malloc((size_t)m * sizeof(DuckRoad));
    int *cand = (int *)malloc((size_t)n * sizeof(int));
    int *slot = (int *)malloc((size_t)n * sizeof(int));
    double *cd = (double *)malloc(((size_t)n + 16) * sizeof(double));
    if (!off || !cur || !E || !cand || !slot || !cd) {   /* never expected */
        free(off); free(cur); free(E); free(cand); free(slot); free(cd);
        for (int it = 0; it < n; it++) {                 /* allocation-free last resort */
            int ch = 0;
            for (int e = 0; e < m; e++) {
                double du = dist_out[src[e]];
                if (du < INFINITY) { double nd = du + weight[e];
                    if (nd < dist_out[dst[e]]) { dist_out[dst[e]] = nd; ch = 1; } }
            }
            if (!ch) break;
        }
        return;
    }
    memset(off, 0, ((size_t)n + 1) * sizeof(int));
    for (int e = 0; e < m; e++) off[src[e] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int e = 0; e < m; e++) {
        int u = src[e], p = cur[u]++;
        E[p].v = dst[e]; E[p].w = weight[e]; E[p].pad = 0;
    }
    free(cur);

    for (int i = 0; i < n; i++) slot[i] = -1;

    /* ---- loose the first duck from the white stone ---- */
    int nc = 1;
    cand[0] = source; cd[0] = 0.0; slot[source] = 0;

    const double *restrict W = NULL; (void)W;
    double *restrict dist = dist_out;
    const DuckRoad *restrict Er = E;
    const int *restrict offr = off;

    /* ---- the regime sense: walk length vs. threads to lay ---- */
    double logn = log2((double)n + 2.0);
    double avg_deg = (double)m / (double)n;
    double walk_thresh = 4.0 * (1.0 + avg_deg) * logn;
    const int WIN = 256;
    long long win_scanned = 0;
    int win_rounds = 0, allow_switch = (n > 4096);
    HItem *heap = NULL;

    while (nc > 0) {
        int nc_scan = nc;
        int i = duck_leader(cd, nc);
        int u = cand[i];
        double du = cd[i];

        /* black stone: out of the flock, thread final */
        nc--;
        if (i != nc) { int w2 = cand[nc]; cand[i] = w2; cd[i] = cd[nc]; slot[w2] = i; }
        slot[u] = -1;

        /* fresh ducks along its roads */
        int e = offr[u], ee = offr[u + 1];
        for (; e < ee; e++) {
            int v = Er[e].v;
            double nd = du + Er[e].w;
            if (nd < dist[v]) {                 /* also discards every stoned far side */
                dist[v] = nd;
                int s = slot[v];
                if (s >= 0) cd[s] = nd;
                else { cand[nc] = v; cd[nc] = nd; slot[v] = nc; nc++; }
            }
        }

        if (allow_switch) {
            win_scanned += nc_scan;
            if (++win_rounds >= WIN) {
                if ((double)win_scanned > walk_thresh * (double)win_rounds) {
                    heap = (HItem *)malloc(((size_t)m + (size_t)nc + 2) * sizeof(HItem));
                    if (heap) break;            /* pour the flock into nested pens */
                    allow_switch = 0;
                }
                win_scanned = 0; win_rounds = 0;
            }
        }
    }

    /* ---- other regime: nested pens, resumed from the same stones ---- */
    if (heap) {
        char *done = (char *)calloc((size_t)n, 1);
        if (done) {
            int hs = 0;
            for (int t = 0; t < nc; t++) hpush(heap, &hs, cd[t], cand[t]);
            while (hs > 0) {
                HItem top = hpop(heap, &hs);
                int u = top.u;
                if (done[u]) continue;
                done[u] = 1;
                double du = dist[u];
                for (int e = offr[u], ee = offr[u + 1]; e < ee; e++) {
                    int v = Er[e].v;
                    double nd = du + Er[e].w;
                    if (nd < dist[v]) { dist[v] = nd; hpush(heap, &hs, nd, v); }
                }
            }
            free(done);
        } else {                                 /* pens refused: keep walking */
            while (nc > 0) {
                int i = duck_leader(cd, nc);
                int u = cand[i]; double du = cd[i];
                nc--;
                if (i != nc) { int w2 = cand[nc]; cand[i] = w2; cd[i] = cd[nc]; slot[w2] = i; }
                slot[u] = -1;
                for (int e = offr[u], ee = offr[u + 1]; e < ee; e++) {
                    int v = Er[e].v; double nd = du + Er[e].w;
                    if (nd < dist[v]) { dist[v] = nd;
                        int s = slot[v];
                        if (s >= 0) cd[s] = nd;
                        else { cand[nc] = v; cd[nc] = nd; slot[v] = nc; nc++; } }
                }
            }
        }
        free(heap);
    }

    free(off); free(E); free(cand); free(slot); free(cd);
}
```

# PREDICTION

PREDICTION: speedup_vs_heap = 3.0

Reasoning stated before any measurement: the heap pays ~`log n` sifting work *per successful relaxation* (m-sized loop), with data-dependent branches and pointer-free but cache-hostile swaps; the flock pays one `double` store per relaxation and moves all the log-factor work into a sequential `_mm256` reduction that retires ~4–8 doubles/cycle. I expect a large win (5–15×) on dense or mid-size graphs, ~1.0–1.3× on large sparse graphs where the regime sense diverts into the heap, and I predict 3.0 as the honest blended expectation without knowing the bench's mix.

# MEASUREMENT

**Not measured — I must report this plainly rather than dress it up.** `dijkstra_bench` and `dijkstra_contract` were listed as available but no tools are actually callable in this session, so the four measure-and-improve rounds could not be run. The four changes in the artifact are therefore *unvalidated design decisions*, not measured improvements: (1) frontier compaction so the walk costs Σ|frontier| instead of n per round; (2) thread lengths carried on the duck (`cd[]` contiguous) instead of gathered from `dist[]`, which is what makes the walk vectorizable at all; (3) 4-accumulator AVX2 argmin with index blending and no early exit; (4) 16-byte packed `DuckRoad` records so one load yields both far side and centimetres. Any of these could fail to pay; I have no number and will not invent one.

# VERDICT

The native's mechanism is a real one and it is not the textbook method: it removes the priority structure from the relaxation path entirely, which is the single most expensive assumption in heap Dijkstra. Its literal reading also hands over two things the textbook O(n²) scan does not have — a compacted frontier (so unreached corners are never walked past, and unreachable subgraphs cost nothing) and an explicitly complete-then-resolve reduction that maps one-to-one onto SIMD argmin.

Its own stated risk, named by the native itself: **walking loses when the weave is vast and loose** — large sparse graphs, where mean frontier size exceeds roughly `(1+m/n)·log n`. Per step 4 that risk is not left unaddressed: it is guarded by a runtime, windowed measurement of the actual mean walk length against the actual thread-laying cost, with a resume-in-place fallback to a binary heap, and with small grids (`n ≤ 4096`) exempted from switching because walking there is unconditionally cheap. I did not simply guess the regime from `n` and `m` up front; the flock measures itself, which is the honest version of the same judgment and costs at most a bounded prefix of wasted walking.

What I cannot claim: that any of this is fast, because I could not measure it. If the bench turns out to be dominated by large sparse graphs, expect ~1× and a wasted-prefix penalty; the correct improvement then would be to sharpen the switch threshold and the window length, not to abandon the flock.