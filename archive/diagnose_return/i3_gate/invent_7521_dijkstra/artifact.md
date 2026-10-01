## MAPPING

**SEED 1 — "The nightingales always land first on the stone carrying the smallest owed letter."**

| World object | Computational object |
|---|---|
| garden stone / house | node `v ∈ [0,n)` |
| chalked letter on a stone | `dist[v]`, a `double` |
| "nothing owed" stone under the tower | `dist[source] = 0.0` |
| "unreadable, far number" | `INFINITY` |
| unlocked stones | not-yet-settled frontier, held as a plain array `scan[]` |
| nightingales circling *all* unlocked stones | one full linear pass over `scan[lo..n)` |
| the bird dropping on the smallest | `argmin` over that array |
| singing its place-name down | the returned index `u` |
| locking the letter | `settled[u]=1; scan[u]=INFINITY` |

Breaks: **"the next place to finalize is found by comparing against every remaining place"** — no, it *affirms* that. What it actually breaks is **"a priority structure must be consulted before every relaxation"**: there is no heap at all; the ordering structure is consulted once per *settle*, never per *edge*.

**SEED 2 — "Threads cast from the tower price each road by what its thief is owed, one direction at a time."**

| World object | Computational object |
|---|---|
| tower / cast site | the currently settled node `u` |
| thread flung to a house the roads touch | one CSR out-edge `off[u]..off[u+1]` |
| the thief's price, directional | `weight[e]`, asymmetric — a directed edge |
| "scratch out and chalk the smaller sum" | `if (du+w < dist[v]) dist[v] = du+w;` |
| "the stone stays exactly where it stood" | in-place write, no reinsertion, no position move |

Breaks: **"a priority structure must be consulted before every relaxation."** Relaxation is a bare compare-and-store into a flat array — a thread costs a load, an add, a compare, a store, and nothing else.

**SEED 3 — "A locked garden letter is never rebuilt again, and threads that reach nowhere are dropped and thrown away."**

| World object | Computational object |
|---|---|
| locked letter, never rechalked | settled node, never re-relaxed (safe ⇔ weights ≥ 0) |
| thread into bramble / over the tide | stale heap entry, or edge into an unreached region |
| letting the thread drop | lazy deletion: `if (done[u]) continue;` — no decrease-key bookkeeping |
| stones left with unreadable debt forever | unreachable nodes keep `INFINITY`, never visited |
| "the traveler was never going to need them" | **early exit** when the minimum owed sum is `INFINITY` |

Breaks: **"the whole graph must be explored to know any single distance."**

## CHOSEN SEED

**SEED 1**, folded together with SEED 2 (they are the same gesture seen from the bird and from the tower) and using SEED 3 as the stopping rule.

Honest statement first, as required: **none of the three seeds breaks "each place's distance must be finalized before its neighbors are explored."** The native is emphatic about the opposite — *"Wherever a nightingale lands, that stone's letter is set — permanently rebuilt one last time, locked"* and only *then* *"From that newly locked house I cast threads again."* Settle-then-expand is the load-bearing beam of this world, not a target. So I fall back to the most literal seed, which is SEED 1.

## ASSUMPTION BROKEN

**"A priority structure must be consulted before every relaxation."**

The native never pushes anything anywhere. Improving a stone's letter is *chalk on a stone that stays exactly where it stood* — an O(1) array write with no sift, no allocation, no reinsertion. The ordering cost is paid once per lock, by a flock of birds that look at every unlocked stone simultaneously. That is exactly the well-validated **O(n²) array-scan Dijkstra**, and per step 4 I let the mechanism arrive there rather than inventing something new — with one addition the textbook version does not have: *a flock of birds looking at all stones at once is a SIMD horizontal argmin*, not a scalar loop. 4 stones per glance (AVX2, two independent accumulators → 8 per iteration).

Regime awareness (step 5): the known_way explicitly names two regimes, so the native must too. Before climbing, **he counts the threads hanging in the garden**. If the garden is thick with roads — `32·m ≥ n²`, i.e. average out-degree ≥ n/32 — the birds' circuit is cheap against the thread-casting and he sends the flock. If the garden is vast and its roads few, the flock wastes its flight, and he falls back to **the knot**: *"A knot of three threads, once cast toward a stone, can be detached … and called back on"* — a >2-ary heap, which I realize as a **4-ary lazy-deletion heap** (4 children per knot: index arithmetic by shift, one cache line per sibling group), strictly better-constant than the reference binary heap. SEED 3 licenses the laziness: a superseded thread is simply let drop when the bird reaches it.

Thread-level parallelism is deliberately **dropped, not guarded**: the metaphor's unit of work is one bird circuit, and there are `n` of them; an OpenMP fork per circuit (~2–5 µs) times n circuits swamps the ~n·8 bytes of work each circuit does at every size where the dense path is selected. SIMD only.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ------------------------------------------------------------------ *
 * The garden, built once: every thread that leaves a stone, grouped
 * by the stone it leaves from (CSR), priced one direction at a time.
 * ------------------------------------------------------------------ */
static void build_csr(int n, int m,
                      const int *restrict src, const int *restrict dst,
                      const double *restrict w,
                      int *restrict off, int *restrict edst, double *restrict ew)
{
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    for (int i = 0; i < m; i++) {           /* off[] used as a moving cursor */
        int u = src[i];
        int p = off[u]++;
        edst[p] = dst[i];
        ew[p]   = w[i];
    }
    for (int i = n; i > 0; i--) off[i] = off[i - 1];   /* slide the cursors back */
    off[0] = 0;
}

/* ------------------------------------------------------------------ *
 * THE FLOCK.  One circuit = one horizontal argmin over every unlocked
 * stone at once.  A locked stone carries INFINITY, so no bird can land
 * on it and no branch is needed to exclude it.  bi < 0 means every
 * remaining letter is still unreadable -> the traveler never needed
 * them -> stop (SEED 3).
 * ------------------------------------------------------------------ */
static int argmin_range(const double *restrict a, int lo, int n, double *bestout)
{
    double best = INFINITY;
    int bi = -1;
    int i = lo;
#if defined(__AVX2__)
    if (n - lo >= 16) {
        __m256d b0 = _mm256_set1_pd(INFINITY), b1 = b0;
        __m256i k0 = _mm256_set1_epi64x(-1),  k1 = k0;
        __m256i c0 = _mm256_set_epi64x(lo + 3, lo + 2, lo + 1, lo + 0);
        __m256i c1 = _mm256_set_epi64x(lo + 7, lo + 6, lo + 5, lo + 4);
        const __m256i step = _mm256_set1_epi64x(8);
        for (; i + 8 <= n; i += 8) {
            __m256d v0 = _mm256_loadu_pd(a + i);
            __m256d v1 = _mm256_loadu_pd(a + i + 4);
            __m256d m0 = _mm256_cmp_pd(v0, b0, _CMP_LT_OQ);
            __m256d m1 = _mm256_cmp_pd(v1, b1, _CMP_LT_OQ);
            b0 = _mm256_blendv_pd(b0, v0, m0);
            b1 = _mm256_blendv_pd(b1, v1, m1);
            k0 = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(k0),
                                                      _mm256_castsi256_pd(c0), m0));
            k1 = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(k1),
                                                      _mm256_castsi256_pd(c1), m1));
            c0 = _mm256_add_epi64(c0, step);
            c1 = _mm256_add_epi64(c1, step);
        }
        double    bv[8];
        long long bx[8];
        _mm256_storeu_pd(bv, b0);
        _mm256_storeu_pd(bv + 4, b1);
        _mm256_storeu_si256((__m256i *)bx, k0);
        _mm256_storeu_si256((__m256i *)(bx + 4), k1);
        for (int t = 0; t < 8; t++)
            if (bx[t] >= 0 && bv[t] < best) { best = bv[t]; bi = (int)bx[t]; }
    }
#endif
    for (; i < n; i++) { double v = a[i]; if (v < best) { best = v; bi = i; } }
    *bestout = best;
    return bi;
}

/* thick garden: send the birds. */
static void solve_flock(int n, const int *restrict off, const int *restrict edst,
                        const double *restrict ew, int source, double *restrict dist)
{
    double        *scan    = (double *)malloc((size_t)n * sizeof(double));
    unsigned char *settled = (unsigned char *)calloc((size_t)n, 1);
    if (!scan || !settled) { free(scan); free(settled); return; }

    for (int i = 0; i < n; i++) scan[i] = INFINITY;
    scan[source] = 0.0;

    int lo = 0;
    for (int it = 0; it < n; it++) {
        while (lo < n && settled[lo]) lo++;
        if (lo >= n) break;

        double best;
        int u = argmin_range(scan, lo, n, &best);
        if (u < 0) break;                    /* threads into bramble: drop them */

        settled[u] = 1;
        scan[u] = INFINITY;                  /* locked; never chalked again */

        double du = dist[u];
        int e0 = off[u], e1 = off[u + 1];
        for (int e = e0; e < e1; e++) {      /* cast, price, rechalk in place */
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; scan[v] = nd; }
        }
    }
    free(scan);
    free(settled);
}

/* ------------------------------------------------------------------ *
 * THE KNOT.  Sparse fallback: 4-ary lazy-deletion heap.  A superseded
 * thread is not untied -- it is let drop when a bird reaches it.
 * ------------------------------------------------------------------ */
typedef struct { double d; int u; int pad; } HItem;   /* 16 bytes */

static inline void h4_push(HItem *restrict h, int *hs, double d, int u)
{
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}

static inline int h4_pop(HItem *restrict h, int *hs, double *dout)
{
    int top_u = h[0].u;
    *dout = h[0].d;
    int sz = --(*hs);
    if (sz > 0) {
        HItem last = h[sz];
        int i = 0;
        for (;;) {
            int c = 4 * i + 1;
            if (c >= sz) break;
            int e = c + 4; if (e > sz) e = sz;
            int bc = c; double bd = h[c].d;
            for (int k = c + 1; k < e; k++) if (h[k].d < bd) { bd = h[k].d; bc = k; }
            if (bd >= last.d) break;
            h[i] = h[bc];
            i = bc;
        }
        h[i] = last;
    }
    return top_u;
}

static void solve_knot(int n, int m, const int *restrict off, const int *restrict edst,
                       const double *restrict ew, int source, double *restrict dist)
{
    unsigned char *done = (unsigned char *)calloc((size_t)n, 1);
    HItem *h = (HItem *)malloc((size_t)(m + 2) * sizeof(HItem));
    if (!done || !h) { free(done); free(h); return; }
    int hs = 0;
    h4_push(h, &hs, 0.0, source);
    while (hs > 0) {
        double d;
        int u = h4_pop(h, &hs, &d);
        if (done[u]) continue;               /* dropped thread, thrown away */
        done[u] = 1;
        double du = dist[u];
        int e0 = off[u], e1 = off[u + 1];
        for (int e = e0; e < e1; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; h4_push(h, &hs, nd, v); }
        }
    }
    free(done);
    free(h);
}

/* ------------------------------------------------------------------ */
void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    int    *off  = (int *)   malloc((size_t)(n + 1) * sizeof(int));
    int    *edst = (int *)   malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !edst || !ew) { free(off); free(edst); free(ew); return; }

    build_csr(n, m, src, dst, weight, off, edst, ew);

    /* count the threads in the garden: thick -> flock, thin -> knot */
    int thick = ((double)m * 32.0 >= (double)n * (double)n);
    if (thick) solve_flock(n, off, edst, ew, source, dist_out);
    else       solve_knot(n, m, off, edst, ew, source, dist_out);

    free(off); free(edst); free(ew);
}
```

## PREDICTION

PREDICTION: speedup_vs_heap = 1.8

Breakdown of where that number comes from, stated before any measurement:
- **Dense/thick cases selected by the guard** (`m ≥ n²/32`): the flock does ~0.4 cycles/stone/circuit with AVX2 (load, cmp, 2 blends, add over 8 doubles), so ~0.4·n² cycles total ordering cost, versus the reference's ~(P+n)·~50 cycles of sift work plus its cache-hostile 16-byte-item heap of size up to m. Expected **3–6×**.
- **Sparse cases** (the fallback): 4-ary heap shallower by log₂4 = 2× in depth, sibling groups on one cache line, plus a cheaper CSR build (no `deg`/`fill` arrays, no `calloc` of n ints twice). Expected **1.15–1.45×**.
- If the benchmark is a mix, the geometric mean lands near 1.8. If it is **all sparse**, I expect ~1.25 and the dense machinery never runs.

## MEASUREMENT

**Not measured. `dijkstra_bench` and `dijkstra_contract` were not available in this session** — no tools were exposed, so the compile-and-time loop and the four improvement rounds could not be run. I am reporting that plainly rather than inventing numbers; there is no measured speedup in this answer, and the prediction above stands untested.

What I would look for if run, and what falsifies the mechanism:
1. **Correctness vs. reference** on all shapes, including unreachable nodes (`INFINITY` preserved), self-loops, duplicate edges, zero-weight edges, `m=0`, `source` isolated.
2. **Regime probe**: force `thick` on and off for the same graph and read off the true crossover. If the measured crossover is at `m·k ≥ n²` with k far from 32, that constant is simply wrong and should be refit — it is the one number in this kernel I am guessing rather than deriving.
3. **Falsifier for the flock**: if a dense case shows the flock *slower* than the reference, the SIMD argmin is not the bottleneck and the n² term dominates earlier than modeled — the guard constant must drop (toward `m·8 ≥ n²`), or the flock is dead weight at all tested sizes.
4. **Falsifier for the knot**: if 4-ary loses to binary, the 4-child linear min scan is costing more than the saved depth, and the sparse path should revert verbatim to the reference heap.

## VERDICT

The native's gesture is, taken literally, **the O(n²) array-scan Dijkstra** — a real, known, validated technique, exactly as step 4 prefers. The thing I refuse to quietly swap in is the heap: the native genuinely never consults a priority structure to relax an edge, and I kept that. The one thing the metaphor gave me that the textbook version does not have is *a flock*, plural, looking at every unlocked stone in the same instant — which is a SIMD horizontal argmin with 64-bit lane indices, not a scalar `for` loop. That is the only place this kernel is doing something the textbook dense Dijkstra isn't.

**Stated risk and how it is addressed.** The flock's own weakness is explicit: *the birds waste their flight over a vast, thin garden*. O(n²) is catastrophic on large sparse graphs — at n = 10⁵, m = 10⁶ it would be ~100× slower than the reference. Per step 4 I have not shipped that risk unguarded: the density check `32·m ≥ n²` is a hard runtime gate with a complete, independent fallback path (the 4-ary knot), so on any sparse input the flock code is never entered and the worst case degrades to "a 4-ary heap instead of a binary one," which is a wash at worst. The guard constant 32 is the honest weak point — it is a modeled crossover (0.4·n² SIMD cycles vs. ~50 cycles per sift), not a fitted one, and without `dijkstra_bench` I could not fit it. It is deliberately set *conservative*: it prefers the heap in the ambiguous band, so an error in it costs at most ~2× on a borderline-dense graph instead of ~100× on a sparse one.

Thread parallelism was considered and **dropped rather than guarded**: the metaphor's unit of work is one bird circuit over ≤ n doubles, and there are n circuits; an OpenMP fork per circuit costs more than the circuit itself at every size where the flock is selected. SIMD, `restrict`, and CSR cache layout only, as instructed.