# MAPPING

## SEED 1 — "The nightingales always land first on the stone carrying the smallest owed letter."

| World object | Problem object |
|---|---|
| casting tower at my own house, center of the garden | `source` node; the only node seeded with a real label |
| the garden of nightingale | the whole directed graph (n nodes) |
| garden-stone with a chalked letter | one slot of the tentative-distance array `dist_out[v]` |
| "nothing owed" under my feet | `dist_out[source] = 0.0` |
| "unreadable, far number" chalked on every other stone | `INFINITY` |
| "scratch it out, chalk the smaller sum… the stone stays exactly where it stood" | decrease-key as a **single in-place store**, no reordering, no re-insertion, no sift — O(1) |
| nightingales sent up **after** the casting, not during | min-extraction is **deferred and batched**, not consulted per relaxation |
| birds "circle the unlocked stones" | a linear sweep over the set of unlocked stones that carry a *readable* letter (finite tentative dist) held in one contiguous array |
| bird drops on the smallest, sings its name down | `argmin` over that contiguous array → next node to settle |
| stone's letter "set, locked, never chalked over again" | node settled; `done` is implied by the stone leaving the sweep set |
| **Breaks** | *"a priority structure must be consulted before every relaxation"* — there is no priority structure at all. Relaxation only writes chalk. Priority is *recomputed by flight* when it is actually needed. Secondarily it re-reads *"the next place to finalize is found by comparing against every remaining place"* as comparing only against every remaining **reached, unlocked** place. |

## SEED 2 — "Threads cast from the tower price each road by what its thief is owed, one direction at a time."

| World object | Problem object |
|---|---|
| thread flung from tower to a house the roads actually touch | one out-edge traversal from the currently settled node |
| "each house the roads actually touch" | CSR adjacency: only real out-neighbours enumerated, never all n |
| the thief guarding a road, asked what he is owed | `weight[i]` |
| "directional, since a road paid one way is not paid the other" | directed edge `src[i] -> dst[i]`; the edge list is bucketed by `src` only |
| casting threads only from the house just locked | outward relaxation from a finalized node |
| **Breaks** | nothing in the list. This is the standard edge-relaxation order; it *confirms* "a road can only be considered once its starting place is fully settled". Its only real content is the CSR build. |

## SEED 3 — "A locked garden letter is never rebuilt again, and threads that reach nowhere are dropped and thrown away."

| World object | Problem object |
|---|---|
| "no thief anywhere charges a negative toll and nothing cheaper can reach it now" | non-negativity proof that `dist_locked[v] ≤ du ≤ du + w`, so `nd < dist_out[v]` is **already false** for a locked v |
| therefore "never chalk over it again" | the `if (done[v]) continue;` test per edge is **provably redundant and can be deleted** — one load and one branch removed from the inner loop |
| thread into bramble / over the tide with no house at the far end | edge with an endpoint outside `[0,n)` — dropped at build time |
| thread looping back around a garden wall to nowhere new | self-loop `u == u` — dropped at build time |
| stones that keep their unreadable debt forever | unreachable nodes keep `INFINITY`; they are **never entered into the sweep set**, so the birds never fly over them |
| "the traveler was never going to need them" | termination when the sweep set empties, not when the graph is exhausted |
| **Breaks** | *"the whole graph must be explored to know any single distance"* — and it shrinks the per-iteration sweep from `n` to `|reached ∧ unlocked|`. |

# CHOSEN SEED

**SEED 1**, augmented by SEED 3 (which is the same mechanism's own pruning rule, not a different mechanism).

Plainly, as required: **none of the three seeds breaks "each place's distance must be finalized before its neighbors are explored."** The native locks with a nightingale *first* and only then casts from that house ("From that newly locked house I cast threads again"), so that assumption is preserved by all three. So I fall back to the most literal seed. Seed 1 is both the most literal (every noun has a slot: stone = array cell, letter = double, lock = removal from sweep, bird = argmin) and the most different from binary-heap Dijkstra: the heap's entire raison d'être — an ordered structure touched on every relaxation — simply does not exist in the garden.

# ASSUMPTION BROKEN

**"A priority structure must be consulted before every relaxation."**

In the garden, relaxation is a chalk stroke on a stone that does not move. Ordering is not maintained incrementally; it is *manufactured on demand* by one sweep of birds over the unlocked, readable stones. Consequences, all literal:

1. **Decrease-key is one store.** A binary heap pays `O(log)` per improvement, or duplicates entries (as the reference kernel does — up to `m` heap entries). The native's "knot of three threads… detached and called back on — recast toward the same stone with the lower number" is exactly an in-place update of the *same* record: the sweep set holds **at most one entry per node, ever**, bounded by `n`, never by `m`.
2. **The sweep is contiguous and branch-free** → this is the one place SIMD belongs: `_mm256_min_pd` over a packed `double` array at a fraction of a cycle per element, versus pointer-chasing sift-downs.
3. **Unreadable stones are not flown over** (SEED 3) → the sweep is over the *frontier*, not over `n`. This is what turns the classic `O(n²)` scan Dijkstra into something also viable on graphs that are not dense.

This mechanism arrives at a **known, validated** technique, as required by step 4: Dijkstra with an *unordered candidate list* + linear min-select (the textbook `O(n²)`/array-scan variant, here with the standard "only scan reached nodes" refinement). I did not invent a new data structure; the seed's own objects landed on the established one, and the established one is exactly what the `known_way` section names as "a real, well-known practical win" for dense or small graphs.

**Two regimes (step 5), recognized in-world.** The native paces the wall before releasing the birds: if the garden is very wide but the threads very few, the birds would tire circling emptiness. Then he uses the other thing a garden has — a **tiered roost**, stones stacked so the lightest letter is always on top: a binary heap. And mid-flight he counts wingbeats: if total circling exceeds `1.5·(n+m)·log₂n` bird-stone visits, he calls them down and moves every unlocked readable stone into the roost, finishing there. Nothing already locked is redone, so only the *selection* work is forfeited — bounded at a few percent of the heap's own cost. Regime test at entry: `n² ≤ 512·(n+m)·log₂(n+2)` → sweep; else roost directly.

**No thread parallelism.** The metaphor's unit of work is one bird sweep, ~`|frontier|/4` vector ops ≈ hundreds of cycles at benchmark sizes — far below OpenMP fork cost, and it happens `n` times. SIMD + `restrict` + packed layout only. This is deliberate, per step 4.

# ARTIFACT

```c
/* The garden of nightingale: in-place chalk (O(1) decrease-key), no priority
   structure touched during relaxation, priority manufactured on demand by a
   SIMD sweep over the reached-and-unlocked stones only.
   Regime fallback: a tiered roost (binary heap) chosen up front for wide/thin
   gardens, or mid-flight when the birds' wingbeats exceed their budget. */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX__)
#include <immintrin.h>
#endif

typedef struct { double d; int u; } HeapItem;

/* ---- the tiered roost: lightest letter on top (hole-based sift, no swaps) ---- */
static void roost_push(HeapItem *restrict h, int *restrict hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}
static HeapItem roost_pop(HeapItem *restrict h, int *restrict hs) {
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

/* ---- the nightingale sweep: drop on the smallest owed letter ---- */
static int bird_drop(const double *restrict d, int nn) {
#if defined(__AVX__)
    if (nn >= 16) {
        __m256d a = _mm256_loadu_pd(d);
        __m256d b = _mm256_loadu_pd(d + 4);
        int i = 8;
        for (; i + 8 <= nn; i += 8) {
            a = _mm256_min_pd(a, _mm256_loadu_pd(d + i));
            b = _mm256_min_pd(b, _mm256_loadu_pd(d + i + 4));
        }
        a = _mm256_min_pd(a, b);
        double t[4];
        _mm256_storeu_pd(t, a);
        double mv = t[0];
        if (t[1] < mv) mv = t[1];
        if (t[2] < mv) mv = t[2];
        if (t[3] < mv) mv = t[3];
        for (; i < nn; i++) if (d[i] < mv) mv = d[i];
        __m256d vv = _mm256_set1_pd(mv);
        int j = 0;
        for (; j + 4 <= nn; j += 4) {
            int msk = _mm256_movemask_pd(
                _mm256_cmp_pd(_mm256_loadu_pd(d + j), vv, _CMP_EQ_OQ));
            if (msk) return j + (int)__builtin_ctz((unsigned)msk);
        }
        for (; j < nn; j++) if (d[j] == mv) return j;
        return 0;
    }
#endif
    {
        int best = 0;
        double bv = d[0];
        for (int i = 1; i < nn; i++) { double x = d[i]; if (x < bv) { bv = x; best = i; } }
        return best;
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    double *restrict dd = dist_out;
    for (int i = 0; i < n; i++) dd[i] = INFINITY;
    if ((unsigned)source >= (unsigned)n) return;
    dd[source] = 0.0;
    if (m < 0) m = 0;

    const int *restrict es = src;
    const int *restrict ed = dst;
    const double *restrict ww = weight;

    /* ---- cast the roads into buckets by their starting house; let dead
            threads drop (endpoint outside the garden, or a wall-loop) ---- */
    int *restrict off = (int *)calloc((size_t)n + 1u, sizeof(int));
    if (!off) return;
    for (int i = 0; i < m; i++) {
        int u = es[i], v = ed[i];
        if ((unsigned)u < (unsigned)n && (unsigned)v < (unsigned)n && u != v) off[u + 1]++;
    }
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    int mk = off[n];
    size_t am = (size_t)(mk > 0 ? mk : 1);

    int *restrict cur = (int *)malloc((size_t)n * sizeof(int));
    int *restrict av  = (int *)malloc(am * sizeof(int));
    double *restrict aw = (double *)malloc(am * sizeof(double));
    int *restrict pos = (int *)malloc((size_t)n * sizeof(int));
    int *restrict cid = (int *)malloc((size_t)n * sizeof(int));
    double *restrict cdv = (double *)malloc((size_t)n * sizeof(double));
    if (!cur || !av || !aw || !pos || !cid || !cdv) {
        free(off); free(cur); free(av); free(aw); free(pos); free(cid); free(cdv);
        return;
    }
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = es[i], v = ed[i];
        if ((unsigned)u < (unsigned)n && (unsigned)v < (unsigned)n && u != v) {
            int p = cur[u]++;
            av[p] = v;
            aw[p] = ww[i];
        }
    }

    /* ---- pace the wall: which regime is this garden in? ---- */
    double lg = log2((double)n + 2.0);
    double heap_scale = (double)(n + mk) * lg;
    int use_sweep = ((double)n * (double)n <= 512.0 * heap_scale);
    double budget = 1.5 * heap_scale + 4096.0;   /* bird-stone visits allowed */
    double wing = 0.0;
    int ncand = 0;

    if (use_sweep) {
        for (int i = 0; i < n; i++) pos[i] = -1;
        cid[0] = source; cdv[0] = 0.0; pos[source] = 0; ncand = 1;
        while (ncand > 0) {
            if (wing > budget) break;            /* call the birds down */
            wing += (double)ncand;
            int bi = bird_drop(cdv, ncand);
            int u = cid[bi];
            double du = cdv[bi];
            int last = --ncand;                  /* the stone leaves the sweep: locked */
            if (bi != last) {
                cdv[bi] = cdv[last];
                int mv = cid[last];
                cid[bi] = mv;
                pos[mv] = bi;
            }
            pos[u] = -1;
            int e1 = off[u + 1];
            for (int e = off[u]; e < e1; e++) {
                int v = av[e];
                double nd = du + aw[e];
                /* no lock test: no thief charges a negative toll, so a locked
                   stone can never fail this compare (dd[v] <= du <= nd). */
                if (nd < dd[v]) {
                    dd[v] = nd;
                    int p = pos[v];
                    if (p >= 0) {
                        cdv[p] = nd;             /* detached and called back on */
                    } else {
                        p = ncand++;
                        cid[p] = v; cdv[p] = nd; pos[v] = p;
                    }
                }
            }
        }
    }

    if (!use_sweep || ncand > 0) {
        HeapItem *restrict h =
            (HeapItem *)malloc(((size_t)mk + (size_t)n + 2u) * sizeof(HeapItem));
        if (h) {
            int hs = 0;
            if (use_sweep) {
                for (int k = 0; k < ncand; k++) roost_push(h, &hs, cdv[k], cid[k]);
            } else {
                roost_push(h, &hs, 0.0, source);
            }
            while (hs > 0) {
                HeapItem t = roost_pop(h, &hs);
                int u = t.u;
                double du = dd[u];
                if (t.d > du) continue;          /* stale letter; the stone is set */
                int e1 = off[u + 1];
                for (int e = off[u]; e < e1; e++) {
                    int v = av[e];
                    double nd = du + aw[e];
                    if (nd < dd[v]) { dd[v] = nd; roost_push(h, &hs, nd, v); }
                }
            }
            free(h);
        }
    }

    free(off); free(cur); free(av); free(aw); free(pos); free(cid); free(cdv);
}
```

# PREDICTION

PREDICTION: speedup_vs_heap = 1.8

Reasoning behind the number, stated before any measurement: on small/dense gardens the sweep path replaces `O(m log n)` duplicate-heavy heap traffic with `O(Σ|frontier|)` contiguous AVX min-reduction at ~0.3 cycles/element plus an O(1) decrease-key, which should be 2.5–4×. On large sparse gardens the entry-time regime test sends control straight to the roost, where I expect 1.05–1.3× over the reference heap from three sources only: hole-based sift (no 3-move swaps), no `done[]` array (staleness read off `dist_out`), and no redundant lock test in the inner loop. Geometric mean across a mixed suite: ~1.8×. If the suite is *entirely* large sparse graphs, expect ~1.1× and my prediction is too high.

# MEASUREMENT

Honest report: **no measurement was taken.** `dijkstra_bench` and `dijkstra_contract` were not available in this session — the tool surface I was given contains no benchmark or contract tool, so the "improve it at most four times" loop could not be run even once. The prediction above is therefore unvalidated, and I am not going to dress up an estimate as a measurement. The artifact is written against the fixed contract as given (argument order unchanged) and is ready for the pipeline to compile with `gcc -O3 -march=native -fopenmp -lm` and measure.

What I did verify, by hand rather than by instrument:
- **Correctness of dropping the lock test** (the only non-obvious deviation): for locked `v`, `dd[v] ≤ du`; weights are `≥ 0`; IEEE round-to-nearest is monotone and `du` is representable, so `fl(du+w) ≥ du ≥ dd[v]`, hence `nd < dd[v]` is false. No settled label can ever be overwritten.
- **No duplicate sweep entries**: a node is inserted only when `pos[v] < 0`, and a locked node can never re-enter by the argument above; so `|sweep| ≤ n` always and the `n`-sized allocations cannot overflow.
- **Heap capacity**: pushes ≤ (successful relaxations ≤ `mk`) + (migration seeds ≤ `n`); allocation is `mk+n+2`.
- **Hand-off correctness**: at migration, `dist_out` holds valid tentative labels and the sweep set is exactly the reached-unlocked set, which is precisely the heap's required initial state; settled nodes are never revisited because the `nd < dd[v]` test cannot fire for them.
- **Unreachable nodes** never enter the sweep or the heap and retain `INFINITY`; both loops terminate on an empty structure, not on graph exhaustion.

# VERDICT

The seed survives literal translation. Every object mapped onto something real: the stone that "stays exactly where it stood" is an in-place `double` store (an O(1) decrease-key, strictly better than the reference's duplicate-push), the birds that "circle the unlocked stones" are an AVX `min_pd` reduction over a packed frontier array, the letter nobody re-chalks is a deleted branch in the hottest loop, and the threads "let drop and thrown away" are the build-time filter plus the `INFINITY` that is never swept. Nothing was quietly swapped for the textbook method — and the pleasing part is that the garden, followed exactly, *lands on* the textbook method the `known_way` section itself calls a real practical win (unordered-list Dijkstra with linear min-select), rather than on something novel and untested.

Risks I named, and what I did about each, since naming a risk and shipping it anyway is not allowed:

1. **"The sweep loses badly on large sparse graphs."** Guarded twice: an entry-time regime test (`n² ≤ 512·(n+m)·log₂(n+2)`) routes wide/thin gardens straight to the binary heap, and a mid-flight wingbeat budget (`1.5·(n+m)·log₂n` bird-stone visits) aborts the sweep and migrates to the heap if the frontier fattens unexpectedly. Forfeited work is selection-only — settled labels carry over — so the worst case is bounded at a small constant fraction of heap cost, not a factor.
2. **"OpenMP would help."** It would not, at these sizes: one sweep is hundreds of cycles and happens `n` times. Dropped deliberately rather than shipped as a guarded maybe.
3. **"The equality scan for the argmin index is a second pass."** Accepted: both passes are contiguous and vectorized, and it avoids the blend-based index tracking that makes SIMD argmin slower than two clean reductions.

Where this could still be *worse* than the reference and is not fully guarded: if the benchmark's graphs sit exactly on the regime boundary, the sweep may run for its full budget and then hand off, costing a few percent. That is the residual, and it is bounded by construction — I would rather state it than hide it. And the headline number is unverified: **the prediction of 1.8× is a prediction, not a result**, because no benchmark tool existed in this session to turn it into one.