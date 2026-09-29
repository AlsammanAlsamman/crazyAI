## APPROACH

The disguised solution is Dijkstra with the "walk the whole chalkboard each round" linear scan replaced by a priority structure whose *front* is always the minimum and whose *insert* only disturbs a few neighbouring entries. Mapping, piece by piece:

| Pip's world | Real implementation |
|---|---|
| chalkboard line per tree | `dist_out[v]`, initialised to `INFINITY`, `0` at `source` |
| "walk the entire chalkboard to find the smallest" (the thing being removed) | the O(n) scan — eliminated |
| Squeaky's narrow filing box, soonest guess at the front | implicit **4-ary min-heap** of `{double d; int u;}` items; `h[0]` is the front card |
| "slides it past just a few neighbouring cards" | `sift-up`: an insert walks only the *one* root-ward path, `log4(size)` swaps — never the whole box |
| "asks for whatever card is at the very front" | `h[0]`, O(1); then sift-down of the last item |
| a card per *new or better guess* (old cards may linger) | **lazy insertion**: push on every successful relaxation, no decrease-key, no position index; a lingering card is discarded at pop by `du > dist_out[u]` |
| "locks that tree in, then walks its bridges" | a node is scanned exactly once, from CSR adjacency |

Because keys pushed for a given node are *strictly* decreasing (`nd < dist_out[v]` is strict) and weights are non-negative, the popped key equals `dist_out[u]` for exactly one card per node — so the stale test needs no `done[]` array at all, and total pushes are bounded by `m+1`.

Implementation work beyond the reference, all of it *inside* this mechanism rather than replacing it:

1. **4-ary instead of binary heap** — `log4` instead of `log2` depth (half the sift-up path length), and the base pointer is offset by 3 elements from a 64-byte boundary so each sibling group of 4 × 16 B lands in **exactly one cache line**. One miss per sift-down level instead of one or two.
2. **Hole-shifting sift-up/sift-down** — the last item is held in a register and the hole is moved; no three-way swaps.
3. **CSR built with one int array, no scratch** — count into `off[u+1]`, prefix-sum (which leaves `off[u] = start(u)`), scatter destructively with `off[u]++`, then shift right by one to restore starts. Saves an `n`-int allocation and a pass.
4. **Interleaved edge array** `{double w; int v;}` — the relaxation loop streams one array instead of two, so a short adjacency list costs one cache line rather than two or three.
5. **Software prefetch of `dist_out[v]`** for higher-degree nodes: the random read of `dist_out` is the dominant miss in the relax loop, so a cheap pre-pass over the (already-cached) adjacency issues all its prefetches before any of them is needed.
6. Parallel `INFINITY` fill only for very large `n` (runtime `if` clause so small graphs never pay OpenMP overhead). The search itself stays sequential — Dijkstra's lock-in order is inherently serial.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct { double w; int v; int pad; } CEdge;   /* 16 B: one stream per adjacency */
typedef struct { double d; int u; int pad; } HItem;   /* 16 B: 4 siblings == 1 cache line */

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    #pragma omp parallel for schedule(static) if(n >= 262144)
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;

    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- CSR build: count -> prefix -> destructive scatter -> shift ---- */
    int *off = (int *)malloc((size_t)(n + 2) * sizeof(int));
    CEdge *E = (CEdge *)malloc((size_t)m * sizeof(CEdge));
    if (!off || !E) { free(off); free(E); return; }

    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int u = 0; u < n; u++) off[u + 1] += off[u];   /* off[u] == start(u), off[n] == m */
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int p = off[u]++;
        E[p].v = dst[i];
        E[p].w = weight[i];
    }
    for (int u = n; u > 0; u--) off[u] = off[u - 1];    /* undo the destructive ++ */
    off[0] = 0;

    /* ---- Squeaky's box: 4-ary min-heap, sibling groups cache-line aligned ---- */
    size_t cap = (size_t)m + 8;
    void *hraw = malloc(cap * sizeof(HItem) + 64);
    if (!hraw) { free(off); free(E); return; }
    HItem *hbase = (HItem *)(((uintptr_t)hraw + 63u) & ~(uintptr_t)63u);
    HItem *h = hbase + 3;   /* children of i are 4i+1..4i+4 -> exactly one 64 B line */

    int hs = 1;
    h[0].d = 0.0; h[0].u = source;

    while (hs) {
        const double du = h[0].d;
        const int u = h[0].u;

        /* pop front, sift the last card down through the hole */
        if (--hs) {
            const HItem last = h[hs];
            const double ld = last.d;
            int i = 0;
            for (;;) {
                int c = (i << 2) + 1;
                int best; double bd;
                if (c + 3 < hs) {
                    bd = h[c].d; best = c;
                    double t1 = h[c + 1].d, t2 = h[c + 2].d, t3 = h[c + 3].d;
                    if (t1 < bd) { bd = t1; best = c + 1; }
                    if (t2 < bd) { bd = t2; best = c + 2; }
                    if (t3 < bd) { bd = t3; best = c + 3; }
                } else if (c < hs) {
                    bd = h[c].d; best = c;
                    for (int k = c + 1; k < hs; k++) {
                        double tk = h[k].d;
                        if (tk < bd) { bd = tk; best = k; }
                    }
                } else break;
                if (bd >= ld) break;
                h[i] = h[best];
                i = best;
            }
            h[i] = last;
        }

        if (du > dist_out[u]) continue;   /* lingering card: a better one already won */

        const int s = off[u], t = off[u + 1];
        if (t - s >= 6)
            for (int e = s; e < t; e++) __builtin_prefetch(&dist_out[E[e].v], 0, 1);

        for (int e = s; e < t; e++) {
            const int v = E[e].v;
            const double nd = du + E[e].w;
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                int i = hs++;                      /* hand Squeaky a new card */
                while (i) {                        /* slide past a few neighbours only */
                    int p = (i - 1) >> 2;
                    if (h[p].d <= nd) break;
                    h[i] = h[p];
                    i = p;
                }
                h[i].d = nd; h[i].u = v;
            }
        }
    }

    free(hraw); free(E); free(off);
}
```

## PREDICTION

PREDICTION: 1.8

(Read as: ~1.8× the throughput of the reference binary-heap Dijkstra above. Reasoning: the search loop should gain ~1.4–2.0× from the 4-ary cache-line-aligned heap plus hole-shifting plus dropping `done[]`, the CSR build ~1.3× from one fewer array and one fewer pass, and the prefetch pass is worth little on degree ≤ 5 graphs and maybe 1.15× on denser ones. If the benchmark graph is dense enough that the reference is dominated by heap traffic, 2.5× is possible; if it is a long thin graph with degree 1–2, the heap barely matters and I would expect only ~1.2×.)

## MEASUREMENT

Not performed. This session was invoked with **no tools available** — `dijkstra_bench` and `dijkstra_contract` are listed in the task text but are not callable here, and there is no shell, so I could not compile, verify against the reference, or take a single timing. I therefore also did not get to use my budget of four improvement rounds; what is above is round zero, unmeasured.

Stating that plainly: the prediction of 1.8 is an estimate from first principles about cache-line counts and heap depth, not an observation, and the two places it could be wrong are (a) if the benchmark graph's average out-degree is very low, the heap optimisations have little to bite on, and (b) if the harness's timed region excludes the CSR build, optimisation 3 contributes nothing.

Correctness I can argue but not demonstrate: every node is popped as non-stale at most once (keys pushed for a node strictly decrease, so exactly one card carries the final key), total pushes ≤ m+1 so the `m+8`-item heap cannot overflow, the `hbase+3` offset only reads/writes indices `3 .. m+3` of the aligned buffer, and the CSR shift-back restores `off[u] = start(u)` exactly. Unreachable nodes are never pushed and keep `INFINITY`.

## VERDICT

A faithful translation of the mechanism — Squeaky's box is a lazy 4-ary implicit min-heap with O(1) front access and single-path insertion, and the "walk the entire chalkboard" scan is gone entirely. I believe it is meaningfully faster than the reference binary-heap version and I expect roughly 1.8×, but **I did not measure it and cannot claim it**. Submit it to the pipeline; if the measured speedup lands below ~1.2× the first thing to check is the average out-degree, and the first thing to try is dropping the prefetch pre-pass (pure overhead on sparse graphs) and testing d=8 versus d=4 for the heap arity.