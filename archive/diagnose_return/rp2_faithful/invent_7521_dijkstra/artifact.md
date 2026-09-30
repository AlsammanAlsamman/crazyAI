# MAPPING

### SEED 1 — "The nightingales always land first on the stone carrying the smallest owed letter."

| World object | Problem object |
|---|---|
| garden of nightingale | the whole graph, `n` nodes |
| garden-stone | node `v` |
| letter chalked on a stone | `dist_out[v]` (tentative distance) |
| "nothing owed" stone under the tower | `dist_out[source] = 0.0` |
| "unreadable, far number" | `INFINITY` |
| **unlocked stones the birds circle** | **the set of chalked-but-unsettled nodes — a compact frontier array, not a heap** |
| nightingales circling, dropping on the smallest | **linear argmin sweep over that frontier's keys** (vectorizable, contiguous) |
| bird sings the place-name down | the argmin returns the node id |
| locked letter | node settled; `pos[v] = -2`, never re-chalked |
| "no thief charges a negative toll" | `w >= 0` ⇒ correctness of the lock |
| "knot of three threads … detached and called back on" | **3-ary heap entry with in-place decrease-key** (`pos[]`-indexed), the *other* flight pattern |
| "the stone stays exactly where it stood" | key rewritten in place, element not moved/reordered — `fkey[pos[v]] = nd`, an O(1) decrease-key |

Breaks: **"the next place to finalize is found by comparing against every remaining place"** *and* **"a priority structure must be consulted before every relaxation"** — the bird compares only against *chalked, unlocked* stones (never the unreadable ones), and an improvement costs one double store, no structure at all.

### SEED 2 — "Threads cast from the tower price each road by what its thief is owed, one direction at a time."

| World object | Problem object |
|---|---|
| casting tower at my own house | the `source` node, root of expansion |
| thread flung from tower to a house | one directed edge traversal `u -> v` |
| "the roads the house actually touch" | CSR adjacency row `off[u] .. off[u+1]` |
| the thief who guards the road, what he is owed | `weight[e]` |
| "directional — a road paid one way is not paid the other" | edges are stored one-way only; no reverse row |
| asking the thief his price | `nd = dist[u] + ew[e]` |

Breaks: **"a road can only be considered once its starting place is fully settled"** — only weakly: the tower's letter is set by fiat ("I am already here") and threads are cast from it before any bird ever flies. Everything after that is still lock-then-cast.

### SEED 3 — "A locked garden letter is never rebuilt again, and threads that reach nowhere are simply dropped and thrown away."

| World object | Problem object |
|---|---|
| locked letter, never chalked over | settled node; no `done[]` re-check needed, because a locked node can never satisfy `nd < dist[v]` when `w >= 0` |
| thread into bramble / over the tide / round a wall to nowhere new | edge to an already-locked node, or a self-loop — failed relaxation, silently discarded |
| "let the thread drop and throw it away" | no push, no allocation, no heap entry for a failed relaxation |
| stones that keep their unreadable debt forever | unreachable nodes stay `INFINITY`; **the loop never visits them at all** |
| "the traveler was never going to need them" | work is proportional to the reachable component, not to `n` |

Breaks: **"the whole graph must be explored to know any single distance."**

# CHOSEN SEED

**SEED 1.** It is the most literal (the bird's landing rule *is* the extract-min, and the text even specifies the two flight patterns and the in-place key rewrite) and the most different from the known way: the known way's selection device is a lazy binary heap of up to `m+2` entries; the native's is a flock sweeping a small compact set of chalked stones with a free decrease-key. SEED 3's "drop the thread" and SEED 2's CSR/directional pricing are folded in as they are consistent with, and required by, SEED 1's machine.

**Plainly: none of the three seeds breaks "each place's distance must be finalized before its neighbors are explored."** The native is explicitly finalize-then-expand — "Wherever a nightingale lands, that stone's letter is set — permanently … From that newly locked house I cast threads again." I am not going to pretend otherwise, so per the instruction I fell back to the most literal seed.

# ASSUMPTION BROKEN

Primary: **"the next place to finalize is found by comparing against every remaining place."** The nightingales circle only the *unlocked, already-chalked* stones — never the ones wearing unreadable debt. The selection set is the frontier, not `n`.

Secondary: **"a priority structure must be consulted before every relaxation"** (a cheaper letter arriving is one chalk stroke — "the stone stays exactly where it stood"), and **"the whole graph must be explored"** (dropped threads cost nothing and unreached stones are never enumerated).

# ARTIFACT

Which code implements which part of the native's mechanism:

* **The garden / stones** — `dist_out[]` is the chalk on every stone; `hpos[v]` is the stone's state: `-1` unreadable-debt (never chalked), `>= 0` chalked & unlocked (its place in the flock), `-2` **locked**.
* **The tower & "nothing owed"** — `dist_out[source] = 0.0`, inserted as the single first chalked stone.
* **Threads, thieves, one direction at a time (SEED 2)** — the CSR build (`off/edst/ew`), one-way rows only; `nd = du + ew[e]` is asking that thief his directional price.
* **The nightingale flight (SEED 1 core)** — `flock_argmin()`: an AVX2 min+index sweep, two accumulators, over the *compact frontier keys* `fkey[0..k-1]`. This is the bird circling and dropping on the smallest.
* **"The stone stays exactly where it stood"** — `fkey[p] = nd` in phase 1: decrease-key in one store, no reordering, no structure consulted.
* **Locking** — `hpos[u] = -2` plus swap-removal of the landed stone from the flock. No `done[]` array is needed anywhere: "no thief charges a negative toll and nothing cheaper can reach it now" is exactly the proof that `nd < dist_out[v]` is unsatisfiable for a locked `v`.
* **Dropped threads (SEED 3)** — a failed `nd < dist_out[v]` does nothing at all: no push, no entry, no memory. Unreachable stones are never enumerated; the loop ends when the flock empties (`k == 0`), not when `n` nodes are done.
* **"A knot of three threads … called back on, so the bird changes its mind" (regime #2)** — the 3-ary indexed heap `sift_up`/`sift_down` over the *same* `hnode`/`hpos` arrays, with true decrease-key (no lazy duplicates, heap size ≤ n).
* **Reading which garden he is in (step 5, in-metaphor regime detection)** — the native watches the flock: while the unlocked chalked stones are few enough for the birds to circle (`k <= FLOCK_CAP`), he uses the flock sweep; the moment the garden gets crowded he **knots the threads into threes** — `heapify` in place (`for (i = k/3; i >= 0; i--) sift_down(...)`, which reuses the very same arrays because a heap is just a permutation of the flock) — and finishes with the knot. One-way switch, so no thrashing. This is the runtime regime test and the fallback path, and it covers both regimes the known-way section names: small/dense (sweep, never switches) and large/sparse-with-big-frontier (switches to the knot).
* **No thread parallelism.** The metaphor's unit of work per bird flight is a sweep of at most `FLOCK_CAP = 2048` doubles (16 KB, L1-resident, ~a few hundred cycles). An OpenMP barrier costs more than the flight it would split, and the sweep is executed `n` times, so threads would be pure loss at these sizes. Per step 4 I dropped it rather than shipping a risk I can't justify.

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* how many chalked, unlocked stones the flock can still circle faster than
   the knot-of-three can be walked. 2048 doubles = 16 KB, L1-resident. */
#define FLOCK_CAP 2048

/* ---- the nightingale flight: index of the smallest letter among k stones ---- */
static inline int flock_argmin(const double *restrict key, int k)
{
#if defined(__AVX2__)
    if (k >= 8) {
        __m256d b0 = _mm256_set1_pd(INFINITY), b1 = _mm256_set1_pd(INFINITY);
        __m256d x0 = _mm256_castsi256_pd(_mm256_set1_epi64x(-1)), x1 = x0;
        __m256i c0 = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i c1 = _mm256_setr_epi64x(4, 5, 6, 7);
        const __m256i st = _mm256_set1_epi64x(8);
        int i = 0;
        for (; i + 8 <= k; i += 8) {
            __m256d v0 = _mm256_loadu_pd(key + i);
            __m256d v1 = _mm256_loadu_pd(key + i + 4);
            __m256d m0 = _mm256_cmp_pd(v0, b0, _CMP_LT_OQ);
            __m256d m1 = _mm256_cmp_pd(v1, b1, _CMP_LT_OQ);
            b0 = _mm256_min_pd(b0, v0);
            b1 = _mm256_min_pd(b1, v1);
            x0 = _mm256_blendv_pd(x0, _mm256_castsi256_pd(c0), m0);
            x1 = _mm256_blendv_pd(x1, _mm256_castsi256_pd(c1), m1);
            c0 = _mm256_add_epi64(c0, st);
            c1 = _mm256_add_epi64(c1, st);
        }
        double bv[8]; long long bi[8];
        _mm256_storeu_pd(bv, b0);
        _mm256_storeu_pd(bv + 4, b1);
        _mm256_storeu_si256((__m256i *)bi, _mm256_castpd_si256(x0));
        _mm256_storeu_si256((__m256i *)(bi + 4), _mm256_castpd_si256(x1));
        double best = INFINITY; int arg = -1;
        for (int t = 0; t < 8; t++) if (bv[t] < best) { best = bv[t]; arg = (int)bi[t]; }
        for (; i < k; i++) if (key[i] < best) { best = key[i]; arg = i; }
        return arg;
    }
#endif
    {
        double best = INFINITY; int arg = -1;
        for (int i = 0; i < k; i++) if (key[i] < best) { best = key[i]; arg = i; }
        return arg;
    }
}

/* ---- the knot of three threads: 3-ary indexed heap, true decrease-key ---- */
static inline void knot_up(int *restrict hn, int *restrict hp,
                           const double *restrict d, int i)
{
    int u = hn[i];
    double du = d[u];
    while (i > 0) {
        int p = (i - 1) / 3;
        int w = hn[p];
        if (d[w] <= du) break;
        hn[i] = w; hp[w] = i; i = p;
    }
    hn[i] = u; hp[u] = i;
}

static inline void knot_down(int *restrict hn, int *restrict hp,
                             const double *restrict d, int k, int i)
{
    int u = hn[i];
    double du = d[u];
    for (;;) {
        int c = 3 * i + 1;
        if (c >= k) break;
        int bestc = c;
        double bd = d[hn[c]];
        int lim = c + 3; if (lim > k) lim = k;
        for (int t = c + 1; t < lim; t++) {
            double x = d[hn[t]];
            if (x < bd) { bd = x; bestc = t; }
        }
        if (!(bd < du)) break;
        int w = hn[bestc];
        hn[i] = w; hp[w] = i; i = bestc;
    }
    hn[i] = u; hp[u] = i;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    /* ---- every stone starts with an unreadable, far debt ---- */
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;

    /* ---- threads, priced one direction at a time: CSR, single offset array ---- */
    int *off = (int *)malloc((size_t)(n + 2) * sizeof(int));
    size_t mm = (size_t)(m > 0 ? m : 1);
    int *edst = (int *)malloc(mm * sizeof(int));
    double *ew = (double *)malloc(mm * sizeof(double));
    int *hnode = (int *)malloc((size_t)n * sizeof(int));   /* flock ids / heap  */
    int *hpos  = (int *)malloc((size_t)n * sizeof(int));   /* stone -> slot     */
    double *fkey = (double *)malloc((size_t)n * sizeof(double)); /* flock letters */
    if (!off || !edst || !ew || !hnode || !hpos || !fkey) {
        free(off); free(edst); free(ew); free(hnode); free(hpos); free(fkey);
        return;
    }

    memset(off, 0, (size_t)(n + 2) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 2]++;
    for (int i = 3; i <= n + 1; i++) off[i] += off[i - 1];
    for (int i = 0; i < m; i++) {
        int p = off[src[i] + 1]++;
        edst[p] = dst[i];
        ew[p] = weight[i];
    }
    /* after the destructive fill, off[0..n] are exactly the CSR row starts */

    const int *restrict EO = off;
    const int *restrict ED = edst;
    const double *restrict EW = ew;
    double *restrict D = dist_out;

    for (int i = 0; i < n; i++) hpos[i] = -1;          /* nothing chalked yet */

    /* ---- "nothing owed": I am already here, the road from me to me is free ---- */
    D[source] = 0.0;
    hnode[0] = source; fkey[0] = 0.0; hpos[source] = 0;
    int k = 1;
    int knotted = 0;

    /* ================= regime 1: send the nightingales up ================= */
    while (k > 0 && !knotted) {
        int j = flock_argmin(fkey, k);
        if (j < 0) break;
        int u = hnode[j];
        double du = fkey[j];

        /* the bird landed: lock this letter, take the stone out of the flock */
        k--;
        if (j < k) {
            fkey[j] = fkey[k];
            int w = hnode[k];
            hnode[j] = w; hpos[w] = j;
        }
        hpos[u] = -2;                 /* locked: never rebuilt again */

        int e = EO[u], ee = EO[u + 1];
        for (; e < ee; e++) {
            if (e + 8 < ee) __builtin_prefetch(&D[ED[e + 8]], 1, 1);
            int v = ED[e];
            double nd = du + EW[e];
            if (nd < D[v]) {          /* else: the thread reached nowhere, drop it */
                D[v] = nd;
                int p = hpos[v];
                if (p >= 0) fkey[p] = nd;        /* stone stays where it stood */
                else { fkey[k] = nd; hnode[k] = v; hpos[v] = k; k++; }
            }
        }

        /* the garden got crowded: knot the threads into threes */
        if (k > FLOCK_CAP) {
            for (int i = k / 3; i >= 0; i--) knot_down(hnode, hpos, D, k, i);
            knotted = 1;
        }
    }

    /* ================= regime 2: walk the knot of three ================= */
    while (k > 0) {
        int u = hnode[0];
        double du = D[u];
        k--;
        if (k > 0) {
            int w = hnode[k];
            hnode[0] = w; hpos[w] = 0;
            knot_down(hnode, hpos, D, k, 0);
        }
        hpos[u] = -2;

        int e = EO[u], ee = EO[u + 1];
        for (; e < ee; e++) {
            if (e + 8 < ee) __builtin_prefetch(&D[ED[e + 8]], 1, 1);
            int v = ED[e];
            double nd = du + EW[e];
            if (nd < D[v]) {
                D[v] = nd;
                int p = hpos[v];
                if (p >= 0) knot_up(hnode, hpos, D, p);   /* called back on */
                else { hnode[k] = v; hpos[v] = k; knot_up(hnode, hpos, D, k); k++; }
            }
        }
    }

    free(off); free(edst); free(ew); free(hnode); free(hpos); free(fkey);
}
```

# PREDICTION

PREDICTION: speedup_vs_heap = 2.0

Written before any measurement (no bench tool was reachable in this session — see MEASUREMENT). Breakdown of where that number comes from:

* small graphs (n ≲ 2000): pure flock sweep, L1-resident, free decrease-key, no heap at all → **3–5×**
* large graphs with a naturally small frontier (grids, road-like): never crosses `FLOCK_CAP` → **1.5–3×**
* large random sparse: knots early, so this is a 3-ary indexed heap (≤ n entries, no lazy duplicates, fewer pops, shallower) vs. the reference's lazy binary heap of up to m+2 → **1.2–1.6×**
* dense: edge-work bound in both → **1.1–1.3×**

# MEASUREMENT

**Not measured. I have to report this plainly rather than dress it up:** this session was started with no tool access — `dijkstra_bench` and `dijkstra_contract` were not callable, so the numbers above are an unvalidated cycle-count estimate, not a result. Nothing in the MEASUREMENT section should be read as evidence. The specific claims a real run must adjudicate:

1. Does `FLOCK_CAP = 2048` sit on the right side of the sweep-vs-knot crossover on this machine? (Estimated crossover ~700–2000 depending on how many successful decrease-keys per pop the instance generates; if the bench is mostly large random sparse, lowering to 768 should be tried first.)
2. Is `flock_argmin` actually issuing ~2 vectors/cycle, or is the index-blend chain the bottleneck?
3. Do the two `__builtin_prefetch` calls help or cost at the bench's degree distribution? They are the single easiest thing to delete if the measurement says so.
4. Exact agreement with the reference: ties, zero-weight edges, self-loops, unreachable nodes left at `INFINITY`. The kernel is deliberately arithmetic-identical to the reference (same `dist[u] + w` order, same strict `<`), so it should match bit-for-bit, not merely within tolerance — any tolerance failure means a real bug, not rounding.

# VERDICT

The native's machine is not the textbook one, and it is not a worse version of it either. Three of his claims are load-bearing and all three are true:

* **"The birds circle the unlocked stones"** — the selection set is the frontier, not the node set. This is the one real algorithmic idea here, and it is why the flock sweep is not the textbook O(n²) scan: it never looks at a stone wearing unreadable debt.
* **"The stone stays exactly where it stood"** — decrease-key as a single store. A heap cannot do this; the native's flock can, and on a sparse graph there are several successful decrease-keys per pop, so this is where most of the win lives.
* **"No thief charges a negative toll, so nothing cheaper can reach it now"** — this eliminates the `done[]` array *and* the stale-entry check entirely. The reference's `if (done[u]) continue;` exists only because it pushes duplicates; the native never does.

Stated risk and how it is addressed (step 4): the flock sweep is *worse* than the known way when the frontier is large, because its cost per landing is linear in the flock. I did not ship that risk — `FLOCK_CAP` is a runtime check on exactly the quantity that makes it bad, and the fallback is an in-place heapify of the same arrays into the native's own knot-of-three. The switch is one-way, so it cannot thrash. I also declined thread parallelism outright rather than guarding it: the metaphor's unit of work (one bird flight over ≤ 16 KB) is smaller than an OpenMP barrier, so it could only lose. The vectorization stayed (SIMD sweep, `restrict`, contiguous frontier keys, single-array CSR).

The honest weak spot: on a large random sparse graph the kernel spends almost all its time in regime 2, which is a good heap but still a heap — the native's distinctive idea does not get to run there, and the predicted 1.2–1.6× is mostly credit for the indexed 3-ary layout rather than for the nightingales. If the benchmark is dominated by that case, the measured overall speedup will land nearer 1.4 than 2.0, and I would rather say that now than explain it afterwards.