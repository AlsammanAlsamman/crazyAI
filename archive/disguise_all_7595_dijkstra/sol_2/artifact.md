## APPROACH

**Mapping the disguised solution onto the real problem.**

| Sunny Hollow | Real implementation |
|---|---|
| "scratch-guess of fastest time so far" for every house | `DJNode.d` — tentative distance, `INFINITY` = "no idea yet" |
| "the little **sorted shelf** of not-yet-finalized houses, nearest in front" | An **addressable (indexed) 4-ary min-heap** holding *only* unfinalized nodes — at most `n` entries, never a duplicate |
| "grab whatever's at the front of the shelf" | `extract-min` at heap slot 0 |
| "**slide that house forward** on the shelf when its guess shrinks" | **decrease-key**: sift-up starting from the node's *known current slot* (`DJNode.pos`), not a fresh push |
| "re-shelve it" / knowing where a house currently sits | `pos[]` back-pointer maintained on every heap move |
| "carved in stone as final" | `pos = -2`; never re-enters the shelf |
| "walk its outgoing bridges" | one sequential scan of the node's CSR adjacency slice |
| "one tree-house at a time, nearest to farthest" | the outer `while (heapsize > 0)` loop, unchanged Dijkstra order |

This is deliberately **not** the reference's lazy/duplicate-push heap (which is a shelf you keep putting *extra copies* of the same house on and then throw away). The shelf here holds each house exactly once — which is what "slide it forward" means literally. That bounds the heap at `n` instead of `m` and is where most of the win comes from.

**Engineering on top of that same mechanism (not a different mechanism):**

1. **4-ary heap instead of binary.** Half the depth ⇒ ~half the sift-up steps (decrease-key dominates), and the 4 children are 32 contiguous bytes — half a cache line, one miss per level instead of two. The 4-child minimum is written branchlessly (cmov tournament).
2. **`{double d; int pos;}` fused into one 16-byte record.** Every relaxation touches both the guess and the shelf position of the same node at a random address; fused, that's **one** cache miss instead of two. Copied out to `dist_out` sequentially at the end.
3. **`{double w; int v;}` fused edge record (16 B).** CSR scatter becomes one random 16-byte store instead of two random stores into two arrays; adjacency scan is a single stream.
4. **Zero-scratch CSR build.** Counts are accumulated into `off[u+1]`, prefix-summed, used *as* the write cursors, then shifted back down — no cursor array, no `calloc`+`memcpy`.
5. **Optional parallel CSR build** (large `m` only): per-thread histograms give conflict-free exact write positions, so counting *and* scatter both run in parallel. The cross-thread prefix pass is **blocked (4096 nodes)** so the `T×n` strided walk stays L2-resident. Thread `t` owns the lowest edge indices, so adjacency order is bit-identical to the serial build — and therefore to the reference.
6. **Clamped software prefetch** of `ni[dst[e+4]]` inside the relaxation loop, hiding the random-access latency that is the real bottleneck.
7. Heap allocation is `n` items, not `m+2` — on a 10M-edge graph that's 160 MB of memory traffic the reference pays and this does not.

Determinism: every reachable node's final value is the min over the same multiset of relaxations as the reference (each node's edges relaxed exactly once, in input order within a node), so results match bit-for-bit, not merely within tolerance. Tie-order differences cannot change a value because `nd < dist[v]` is already false for any finalized `v` under non-negative weights.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Fused edge record: one random 16B store on build, one stream on scan. */
typedef struct { double w; int v; int pad_; } DJEdge;
/* Fused node record: distance + shelf position share one cache line. */
typedef struct { double d; int pos; int pad_; } DJNode;

#define DJ_NEW  (-1)
#define DJ_DONE (-2)

/* "slide the house forward on the shelf" : decrease-key / insert, 4-ary sift-up */
static inline void dj_up(double *restrict hk, int *restrict hv,
                         DJNode *restrict ni, int i, double k, int u)
{
    while (i > 0) {
        int p = (i - 1) >> 2;
        double kp = hk[p];
        if (kp <= k) break;
        int vp = hv[p];
        hk[i] = kp; hv[i] = vp; ni[vp].pos = i;
        i = p;
    }
    hk[i] = k; hv[i] = u; ni[u].pos = i;
}

/* restore the shelf after grabbing the front item : 4-ary sift-down */
static inline void dj_down(double *restrict hk, int *restrict hv,
                           DJNode *restrict ni, int hs, double k, int u)
{
    int i = 0;
    for (;;) {
        int c = (i << 2) + 1;
        if (c >= hs) break;
        int best; double bk;
        if (c + 3 < hs) {                     /* full 4 children: branchless tournament */
            double k0 = hk[c], k1 = hk[c + 1], k2 = hk[c + 2], k3 = hk[c + 3];
            int s1 = (k1 < k0); double a = s1 ? k1 : k0; int ia = c + s1;
            int s2 = (k3 < k2); double b = s2 ? k3 : k2; int ib = c + 2 + s2;
            int s3 = (b < a);   bk = s3 ? b : a;         best = s3 ? ib : ia;
        } else {
            best = c; bk = hk[c];
            for (int j = c + 1; j < hs; j++) {
                double t = hk[j];
                if (t < bk) { bk = t; best = j; }
            }
        }
        if (!(bk < k)) break;
        int vb = hv[best];
        hk[i] = bk; hv[i] = vb; ni[vb].pos = i;
        i = best;
    }
    hk[i] = k; hv[i] = u; ni[u].pos = i;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (m < 0) m = 0;

    size_t nn = (size_t)n;
    size_t mm = (size_t)(m > 0 ? m : 1);

    int    *off = (int *)   malloc((nn + 1) * sizeof(int));
    DJNode *ni  = (DJNode *)malloc(nn * sizeof(DJNode));
    double *hk  = (double *)malloc(nn * sizeof(double));
    int    *hv  = (int *)   malloc(nn * sizeof(int));
    DJEdge *adj = (DJEdge *)malloc(mm * sizeof(DJEdge));

    if (!off || !ni || !hk || !hv || !adj) {          /* degenerate safety net */
        for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
        if (source >= 0 && source < n) dist_out[source] = 0.0;
        free(off); free(ni); free(hk); free(hv); free(adj);
        return;
    }

    int nthr = 1;
#ifdef _OPENMP
    nthr = omp_get_max_threads();
    if (nthr > 8) nthr = 8;
    if (nthr < 1) nthr = 1;
#endif

    int par_build = 0;
#ifdef _OPENMP
    if (nthr > 1 && m >= (1 << 21) &&
        (double)nthr * (double)nn * 4.0 <= 2.5e8) par_build = 1;
#endif

    /* ---------------- CSR construction ---------------- */
#ifdef _OPENMP
    if (par_build) {
        int *hist = (int *)malloc((size_t)nthr * nn * sizeof(int));
        if (!hist) {
            par_build = 0;
        } else {
            #pragma omp parallel num_threads(nthr)
            {
                int t = omp_get_thread_num();
                int *h = hist + (size_t)t * nn;
                memset(h, 0, nn * sizeof(int));
                long long lo = (long long)m * t / nthr;
                long long hi = (long long)m * (t + 1) / nthr;
                for (long long i = lo; i < hi; i++) h[src[i]]++;
            }
            /* blocked cross-thread exclusive prefix: keeps T*4096 ints L2-resident.
               thread 0 owns the lowest edge indices => adjacency order == input order */
            {
                int run = 0;
                const int B = 4096;
                for (int b = 0; b < n; b += B) {
                    int be = b + B; if (be > n) be = n;
                    for (int u = b; u < be; u++) {
                        off[u] = run;
                        for (int t = 0; t < nthr; t++) {
                            int *p = hist + (size_t)t * nn + (size_t)u;
                            int c = *p; *p = run; run += c;
                        }
                    }
                }
                off[n] = run;
            }
            #pragma omp parallel num_threads(nthr)
            {
                int t = omp_get_thread_num();
                int *h = hist + (size_t)t * nn;
                long long lo = (long long)m * t / nthr;
                long long hi = (long long)m * (t + 1) / nthr;
                for (long long i = lo; i < hi; i++) {
                    int u = src[i];
                    int p = h[u]++;
                    adj[p].w = weight[i];
                    adj[p].v = dst[i];
                }
            }
            free(hist);
        }
    }
#endif
    if (!par_build) {
        memset(off, 0, (nn + 1) * sizeof(int));
        for (int i = 0; i < m; i++) off[src[i] + 1]++;
        for (int u = 0; u < n; u++) off[u + 1] += off[u];   /* off[u] = start(u) */
        for (int i = 0; i < m; i++) {                        /* off[] doubles as cursor */
            int p = off[src[i]]++;
            adj[p].w = weight[i];
            adj[p].v = dst[i];
        }
        for (int u = n; u >= 1; u--) off[u] = off[u - 1];    /* shift cursors back */
        off[0] = 0;
    }

    /* ---------------- scratch-guesses: everyone "no idea yet" ---------------- */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nthr) if(n >= 65536)
#endif
    for (int i = 0; i < n; i++) { ni[i].d = INFINITY; ni[i].pos = DJ_NEW; }

    if (source >= 0 && source < n) {
        ni[source].d = 0.0;
        ni[source].pos = 0;
        hk[0] = 0.0; hv[0] = source;
        int hs = 1;

        /* ---------------- nearest-to-farthest, one house at a time ---------------- */
        while (hs > 0) {
            int u = hv[0];
            double du = hk[0];
            ni[u].pos = DJ_DONE;                 /* carved in stone */
            hs--;
            if (hs > 0) dj_down(hk, hv, ni, hs, hk[hs], hv[hs]);

            int e  = off[u];
            int ee = off[u + 1];
            for (; e < ee; e++) {
                int pfi = e + 4; if (pfi >= ee) pfi = ee - 1;
                __builtin_prefetch(&ni[adj[pfi].v], 1, 1);

                int v = adj[e].v;
                double nd = du + adj[e].w;
                DJNode *V = ni + v;
                if (nd < V->d) {                 /* false for DONE nodes: weights >= 0 */
                    V->d = nd;
                    int p = V->pos;
                    if (p >= 0) {
                        dj_up(hk, hv, ni, p, nd, v);      /* slide forward on the shelf */
                    } else {
                        int idx = hs++;
                        dj_up(hk, hv, ni, idx, nd, v);    /* first time on the shelf */
                    }
                }
            }
        }
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nthr) if(n >= 65536)
#endif
    for (int i = 0; i < n; i++) dist_out[i] = ni[i].d;

    free(off); free(ni); free(hk); free(hv); free(adj);
}
```

## PREDICTION

PREDICTION: 2.3

(Interpretation: ~2.3× faster end-to-end than the given binary-heap lazy-deletion reference. Breakdown behind the number: ~1.35× from the 4-ary indexed heap with true decrease-key — heap bounded at `n` instead of `m`, ~½ the sift depth, and no popping-and-discarding of stale entries; ~1.3× from the fused `{d,pos}` and `{w,v}` records plus prefetching, which roughly halve the random-miss count on the relaxation path; ~1.15× from the cheaper CSR build (no cursor array, optional parallel scatter on large `m`). On a small/dense graph where the whole working set is L3-resident I expect closer to 1.4–1.6×, since the cache-locality half of the win evaporates and only the heap-work reduction remains.)

## MEASUREMENT

**Not performed — I could not measure this.** The task listed `dijkstra_bench` and `dijkstra_contract`, but this session was invoked with no tool access at all ("No tools are available in this session"), so there was no way to compile, run the contract check, or time anything. The "improve it at most four times" loop therefore never ran: the artifact above is a single un-iterated shot, and the prediction is an estimate from first principles, not a measured result.

What I verified by hand instead, since that is all that was available:

- **Contract shape** — `void kernel(int, int, const int*, const int*, const double*, int, double*)`, argument order and semantics unchanged; `dist_out` length `n`, `INFINITY` for unreachable, `0.0` at `source`.
- **CSR offset algebra** — traced the count → prefix → cursor-scatter → shift-back sequence symbolically: after scatter `off[u] == start(u+1)`, so the descending shift `off[u] = off[u-1]`, `off[0] = 0` restores exactly `start(u)`, with `off[n] == m`.
- **Parallel-build edge ordering** — thread `t` owns edge indices `[m·t/T, m·(t+1)/T)` and the prefix assigns thread blocks in ascending `t`, so each node's adjacency is in input order, identical to the serial path and to the reference.
- **Heap invariants** — `pos` is written on every element move in both sift directions and on the final placement; the popped node is marked `DJ_DONE` *before* `dj_down` runs, and `dj_down` is skipped when the heap empties, so the popped node is never re-addressed.
- **Skipping the `done` check is safe** — under non-negative weights, for finalized `v` any candidate `nd >= dist[v]`, so `nd < V->d` is already false; zero-weight edges give equality, not strict less-than.

Unmeasured risks I would have gone looking for first, in order: (1) whether `par_build` ever fires on the benchmark's actual sizes — if `m < 2^21` the parallel path is dead code and the speedup drops toward ~2.0×; (2) whether the unconditional clamped prefetch is a net loss on very-low-degree graphs (average degree ≤ 2), where it prefetches a line that is used one iteration later anyway; (3) whether 16 B/edge loses to 12 B/edge on a high-degree graph that is bandwidth-bound on the adjacency scan.

## VERDICT

The mechanism is implemented faithfully — a single-copy sorted shelf with real decrease-key ("slide it forward"), not the reference's push-a-duplicate-and-discard-later heap — and every optimization above is an improvement to *how* that shelf is implemented, not a substitute for it. I have good structural reasons to expect roughly 2× (fewer heap entries, half the sift depth, half the random misses per relaxation), and the code is hand-checked for correctness against the contract.

But the honest bottom line is that **this is an unverified claim**. I predicted 2.3 and then could not test it, so I have a prediction and no measurement — which by the standard I was asked to hold myself to counts as an incomplete result, not a success. If the benchmark harness runs it and the number comes back at 1.1×, or it fails the contract check on some edge case I mis-traced, that outcome is entirely live and I would not be surprised by it. Anyone using this should run `dijkstra_contract` before `dijkstra_bench`.