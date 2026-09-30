# MAPPING

### SEED 1 — "The nightingales always land first on the stone carrying the smallest owed letter."

| World object | Computational object |
|---|---|
| garden stone | node `v` (its identity is its index) |
| letter chalked on a stone | `dist_out[v]` — a double, addressed by node id |
| "nothing owed" beneath my feet | `dist_out[source] = 0.0` |
| unreadable, far number | `INFINITY` |
| *unlocked* stone with a readable letter | entry in a **contiguous candidate array** `fd[0..fs)` (owed sums) paired with `fu[0..fs)` (place-names) |
| the flock circling the unlocked stones | one linear pass over `fd[0..fs)` — a **SIMD min-reduction with index** (a flock, not one bird: 4–8 lanes circling at once) |
| the bird dropping onto the smallest | `argmin(fd, fs)` |
| singing its place-name down | `u = fu[k]` |
| locking | remove from the candidate array, `pos[u] = LOCKED`, `dist_out[u]` never written again |
| "no thief charges a negative toll" | `w >= 0`, the greedy-lock correctness certificate |

**Silent assumption broken:** *"the next place to finalize is found by comparing against every remaining place."* The birds circle only the **unlocked stones with readable letters** — not every remaining stone. There is no heap (so no "priority structure" either), but there is also no full-array `n²` scan: the sweep length is the *reached-and-unlocked frontier*, which on geometric/layered graphs is `O(√n)` or `O(n^{2/3})`, and on dense graphs is exactly the classic dense-Dijkstra scan.

### SEED 2 — "Threads cast from the tower price each road by what its thief is owed, one direction at a time."

| World object | Computational object |
|---|---|
| casting tower at the locked house | the currently locked node `u`, source of an out-adjacency sweep |
| thread flung to a house the road touches | one out-edge `u→v` |
| the thief, and what he is owed | `ew[e]` — the weight, attached to the arc |
| "directional: a road paid one way is not paid the other" | directed CSR built from `src[]` only; `dst→src` is never implied |
| roads the garden *actually* touches | CSR row `off[u]..off[u+1]` — adjacency, not an `n×n` probe |
| chalking a smaller sum, "stolen from its old amount, replaced, though the stone stays exactly where it stood" | **in-place relaxation**: `dist_out[v] = nd` and `fd[pos[v]] = nd`; the candidate never changes slot, nothing is re-inserted |
| a knot of three threads, detached and called back on with a lower number | the three parallel arrays `fd` / `fu` / `pos` re-tied for the same stone: **decrease-key at O(1), no push** |

**Silent assumption broken:** *"a priority structure must be consulted before every relaxation."* A relaxation here is a compare and two stores. Nothing is pushed anywhere; the bird "changes its mind" only because the number under it changed.

### SEED 3 — "A locked garden letter is never rebuilt again, and threads that reach nowhere are simply dropped and thrown away."

| World object | Computational object |
|---|---|
| locked letter, never rebuilt | `pos[u] = -2`; guaranteed safe because IEEE addition is monotone: `fl(dist[x]+w) >= dist[x] >= dist[u]` for `w >= 0`, so a locked node can never pass `nd < dist_out[u]` |
| thread into bramble / over the tide / round a wall to nowhere new | an edge to an already-locked or non-improving node — `nd < dist_out[v]` fails; nothing is enqueued |
| stones that stay chalked with unreadable debt forever | unreachable nodes are **never enrolled in the candidate array at all**, so they cost zero sweep time, forever |
| "the traveler was never going to need them" | no early-exit test needed; the candidate array simply empties and the loop ends |

**Silent assumption broken:** *"the whole graph must be explored to know any single distance."* The sweep's length is the reached frontier; the unreached part of the graph is never touched by even one comparison.

---

# CHOSEN SEED

**Seed 1** (with Seeds 2 and 3 as its two supporting organs — they are parts of the same machine, not alternatives).

First, plainly: **none of the three seeds breaks "each place's distance must be finalized before its neighbors are explored."** The native is emphatic about the opposite — *"Wherever a nightingale lands, that stone's letter is set... From that newly locked house I cast threads again."* That is settle-then-expand, exactly Dijkstra's discipline. I will not pretend otherwise. So I fall back to the most literal and most-different-from-known-way seed, which is Seed 1: the extract-min is a **flock sweeping a compacted frontier**, with *no heap in existence*.

# ASSUMPTION BROKEN

*"The next place to finalize is found by comparing against every remaining place."*

The known way dodges this with a heap (log n per operation, m pushes, pointer-chasing, cache misses). The textbook alternative dodges it the other way with a full `O(n²)` array scan. The native does neither: he sweeps **only the stones that are both reached and unlocked**, contiguously, with a flock (SIMD). This collapses to the validated dense-Dijkstra `O(n²)` scan in the worst case (which is the *known-good* technique for dense/small graphs — I am letting the metaphor arrive at it rather than inventing something), and is strictly better than it whenever the frontier is thin.

# ARTIFACT

Regime recognition, as required, is in the metaphor itself: **"I do not walk everywhere at once. I wait."** The native *waits* — he counts how much circling the birds have already done, and if the garden of unlocked stones has grown so wide that the flock is spending longer aloft than the walking would have cost, he stops sending birds and stacks the unlocked stones into a cairn where the smallest rises to the top (a binary heap — the known way, reached as a *fallback*, mid-flight, sharing the same letters). Budget: `2·(m+n)·log₂(n+2)` stone-comparisons, so the worst case is ~10–25 % of heap time wasted before switching, never a multiple of it. This is the guard for the exact condition my own verdict names (large-and-sparse). No thread parallelism: at benchmark sizes one sweep is a few microseconds, below OpenMP fork cost — vectorization only, as instructed.

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

/* ---------------------------------------------------------------
   THE NIGHTINGALE SWEEP
   The flock circles the contiguous array of owed sums (unlocked,
   readable stones only) and drops onto the smallest, singing back
   its slot.  4 lanes (AVX2) or 8 lanes (AVX-512) circle at once.
   --------------------------------------------------------------- */
static inline int nightingale_argmin(const double *restrict a, int n)
{
#if defined(__AVX512F__)
    if (n >= 16) {
        __m512d vmin = _mm512_loadu_pd(a);
        __m512i vidx = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512i vcur = vidx;
        const __m512i v8 = _mm512_set1_epi64(8);
        int i = 8;
        for (; i + 8 <= n; i += 8) {
            vcur = _mm512_add_epi64(vcur, v8);
            __m512d v = _mm512_loadu_pd(a + i);
            __mmask8 lt = _mm512_cmp_pd_mask(v, vmin, _CMP_LT_OQ);
            vmin = _mm512_mask_blend_pd(lt, vmin, v);
            vidx = _mm512_mask_blend_epi64(lt, vidx, vcur);
        }
        double mv[8]; long long mi[8];
        _mm512_storeu_pd(mv, vmin);
        _mm512_storeu_si512((void *)mi, vidx);
        double best = mv[0]; int bi = (int)mi[0];
        for (int k = 1; k < 8; ++k) if (mv[k] < best) { best = mv[k]; bi = (int)mi[k]; }
        for (; i < n; ++i)          if (a[i] < best)  { best = a[i];  bi = i; }
        return bi;
    }
#endif
#if defined(__AVX__)
    if (n >= 8) {
        __m256d vmin = _mm256_loadu_pd(a);
        __m256i vidx = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i vcur = vidx;
        const __m256i v4 = _mm256_set1_epi64x(4);
        int i = 4;
        for (; i + 4 <= n; i += 4) {
            vcur = _mm256_add_epi64(vcur, v4);
            __m256d v  = _mm256_loadu_pd(a + i);
            __m256d lt = _mm256_cmp_pd(v, vmin, _CMP_LT_OQ);
            vmin = _mm256_blendv_pd(vmin, v, lt);
            vidx = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(vidx),
                                                       _mm256_castsi256_pd(vcur), lt));
        }
        double mv[4]; long long mi[4];
        _mm256_storeu_pd(mv, vmin);
        _mm256_storeu_si256((__m256i *)mi, vidx);
        double best = mv[0]; int bi = (int)mi[0];
        for (int k = 1; k < 4; ++k) if (mv[k] < best) { best = mv[k]; bi = (int)mi[k]; }
        for (; i < n; ++i)          if (a[i] < best)  { best = a[i];  bi = i; }
        return bi;
    }
#endif
    {
        double best = a[0]; int bi = 0;
        for (int i = 1; i < n; ++i) if (a[i] < best) { best = a[i]; bi = i; }
        return bi;
    }
}

/* ---------------------------------------------------------------
   THE CAIRN (fallback for the wide-garden / thin-roads regime):
   a plain lazy binary heap -- the known way, kept in reserve.
   --------------------------------------------------------------- */
typedef struct { double d; int u; } Cairn;

static inline void cairn_push(Cairn *restrict h, int *restrict hs, double d, int u)
{
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= h[i].d) break;
        Cairn t = h[p]; h[p] = h[i]; h[i] = t; i = p;
    }
}

static inline Cairn cairn_pop(Cairn *restrict h, int *restrict hs)
{
    Cairn top = h[0];
    (*hs)--; h[0] = h[*hs];
    int i = 0, s;
    for (;;) {
        int l = 2 * i + 1, r = l + 1;
        s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        Cairn t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}

#define LOCKED (-2)
#define UNSEEN (-1)

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;      /* unreadable debt */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;                                  /* "nothing owed" */
    if (m <= 0) return;

    /* ---- the roads: directed CSR, each thief's toll, one direction only ---- */
    int    *off  = (int *)   calloc((size_t)n + 1, sizeof(int));
    int    *cur  = (int *)   malloc((size_t)n * sizeof(int));
    int    *edst = (int *)   malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    /* ---- the knot of three threads, one per candidate stone ---- */
    double *fd   = (double *)malloc((size_t)n * sizeof(double)); /* owed sum   */
    int    *fu   = (int *)   malloc((size_t)n * sizeof(int));    /* place-name */
    int    *pos  = (int *)   malloc((size_t)n * sizeof(int));    /* back-thread*/

    if (!off || !cur || !edst || !ew || !fd || !fu || !pos) {
        free(off); free(cur); free(edst); free(ew); free(fd); free(fu); free(pos);
        return;
    }

    for (int i = 0; i < m; ++i) ++off[src[i] + 1];
    for (int i = 0; i < n; ++i) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; ++i) {
        int p = cur[src[i]]++;
        edst[p] = dst[i];
        ew[p]   = weight[i];
    }
    free(cur);

    const int    *restrict EO = off;
    const int    *restrict ED = edst;
    const double *restrict EW = ew;
    double *restrict D  = dist_out;
    double *restrict FD = fd;
    int    *restrict FU = fu;
    int    *restrict PS = pos;

    memset(PS, 0xFF, (size_t)n * sizeof(int));               /* all UNSEEN */

    int fs = 0;
    FD[0] = 0.0; FU[0] = source; PS[source] = 0; fs = 1;     /* climb the tower */

    /* "I wait": how much circling the flock may do before the native decides
       the garden is too wide for birds and starts stacking a cairn instead. */
    const double budget = 2.0 * ((double)m + (double)n) * log2((double)n + 2.0) + 4096.0;
    double circled = 0.0;
    int tired = 0;

    while (fs > 0) {
        if (circled > budget) { tired = 1; break; }
        circled += (double)fs;

        int k = nightingale_argmin(FD, fs);                  /* the bird lands  */
        int u = FU[k];
        double du = FD[k];

        PS[u] = LOCKED;                                      /* letter is set   */
        --fs;
        if (k != fs) { FD[k] = FD[fs]; FU[k] = FU[fs]; PS[FU[fs]] = k; }

        /* cast threads from the newly locked house */
        int e = EO[u], ee = EO[u + 1];
        for (; e < ee; ++e) {
            int v = ED[e];
            double nd = du + EW[e];
            /* A locked stone can never pass this test: IEEE addition is
               monotone and every toll is >= 0, so nd >= du >= D[v]. */
            if (nd < D[v]) {
                D[v] = nd;
                int p = PS[v];
                if (p >= 0) FD[p] = nd;                      /* knot re-tied    */
                else { PS[v] = fs; FD[fs] = nd; FU[fs] = v; ++fs; }  /* enrolled */
            }
        }
    }

    if (tired) {
        /* wide garden, thin roads: stack the unlocked stones into a cairn and
           finish the walk the known way, on the very same letters. */
        Cairn *heap = (Cairn *)malloc(((size_t)n + (size_t)m + 2) * sizeof(Cairn));
        if (heap) {
            int hs = 0;
            for (int i = 0; i < fs; ++i) cairn_push(heap, &hs, FD[i], FU[i]);
            while (hs > 0) {
                Cairn t = cairn_pop(heap, &hs);
                int u = t.u;
                if (PS[u] == LOCKED) continue;
                PS[u] = LOCKED;
                double du = D[u];
                int e = EO[u], ee = EO[u + 1];
                for (; e < ee; ++e) {
                    int v = ED[e];
                    double nd = du + EW[e];
                    if (nd < D[v]) { D[v] = nd; cairn_push(heap, &hs, nd, v); }
                }
            }
            free(heap);
        }
    }

    free(off); free(edst); free(ew); free(fd); free(fu); free(pos);
}
```

# PREDICTION

**PREDICTION: speedup_vs_heap = 2.0**

(Stated before any measurement. Reasoning: the CSR build is identical to the reference, so it is common overhead and caps the attainable speedup. Above it, I remove *all* `m` heap pushes — the dominant cost of the reference, each a random-access sift of depth up to log₂ m — and replace the extract-min with a contiguous 4–8-lane sweep of the frontier. On a dense or moderate graph I expect 2.5–6×; on a thin-frontier geometric graph 1.5–3×; on a large sparse random graph the budget trips and I expect **0.85–1.0×**, i.e. roughly break-even minus the wasted circling. A mixed benchmark should land near 2.)

# MEASUREMENT

**Not measured — and I will not dress this up.** This session was invoked with no tools available: `dijkstra_bench` and `dijkstra_contract` could not be called, so there is no timing number, no reference comparison, and consequently **zero of the four permitted improvement rounds were taken**. Everything above is design-time reasoning folded into a single artifact, which is strictly weaker evidence than one honest measurement.

What I verified by hand instead, since that is all that was available:

- **Contract**: signature, argument order and types are copied verbatim; `dist_out` is filled with `INFINITY` first, `0.0` at `source`.
- **Correctness of the lock**: a locked node cannot be re-enrolled, because `fl(dist[x]+w) >= fl(dist[x]+0) = dist[x] >= dist[u]` for `w >= 0` (IEEE addition is monotone), so `nd < D[u]` is always false. This is why no `done[]` array is needed, and why omitting the `pos` load on the non-improving path is safe.
- **Bounds**: each node is enrolled at most once (`UNSEEN → slot → LOCKED`, never back), so `fs <= n` and `fd/fu` never overflow; the cairn takes at most `n + m` pushes.
- **Numerics**: relaxations are the same `dist[u] + w` in the same order-of-path as the reference; the SIMD sweep may break exact ties at a different index than a scalar scan, which changes lock *order* but not any distance value.
- **Degenerate inputs**: `n<=0`, `m==0`, bad `source`, allocation failure all return with `dist_out` well-defined.

**What would falsify my prediction**, stated up front: if the benchmark is dominated by a single large sparse random graph (`n >= 10^5`, average degree ~4–10), the budget will trip and the measured number should come out **below 1.0** — probably 0.85–0.95. That would not be a bug; it would mean the metaphor's mechanism does not fit the benchmark's regime and the guard is paying its insurance premium. If the measured number is below ~0.8, that indicates the budget factor `2.0` is too generous and should be cut to ~0.5.

# VERDICT

The native's machine is real and it is not the textbook one. Its named risk and the guard for it:

- **Where it wins:** dense graphs (`m ~ n²`) — it beats the heap by the full `log n` factor with no heap ever allocated; small graphs — no heap, no pointer-chasing, everything in cache; and thin-frontier graphs (grids, road networks, layered DAGs) where the flock circles `O(√n)` stones instead of `n`, which is *better* than the classic dense scan the known-way section names.
- **Where it could lose, named explicitly:** a large, sparse, high-expansion graph, where the reached-and-unlocked frontier is `Θ(n)` and the sweep degenerates to `O(n²)`.
- **How that risk is discharged, not hand-waved:** the native's own "I wait" — a runtime circling budget of `2·(m+n)·log₂(n+2)` comparisons, after which he abandons the birds mid-walk and finishes with the validated binary heap over the identical `dist` array. The two regimes the known-way section names (dense/small vs. sparse/large) therefore both have a path, and the switch is *measured at runtime*, not guessed from `n` and `m` a priori — because frontier width is not a function of `n` and `m`. Worst case is bounded at roughly `1.1–1.25×` the heap, not a multiple of it.
- **What I refused to do:** no thread parallelism. One sweep at benchmark sizes is single-digit microseconds, under the OpenMP fork cost — the metaphor's unit of work (one flight of the flock) is genuinely too small, and saying so is more useful than shipping a `#pragma` that loses.

Honest bottom line: the mechanism is sound, literal, and self-guarding, but **it is unmeasured**. Treat `speedup_vs_heap = 2.0` as a prediction awaiting refutation, not a result — and if the harness reports below 1.0 on a sparse-dominated suite, the fix is the one-constant change named above, not a redesign.