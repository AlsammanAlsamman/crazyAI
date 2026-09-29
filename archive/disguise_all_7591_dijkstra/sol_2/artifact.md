## APPROACH

The disguised solution is an **indexed priority queue with decrease-key**, and every image in it maps onto a concrete implementation choice:

| Mice story | Real implementation |
|---|---|
| "one ticket per burrow" (not one per improvement) | the heap holds **at most one entry per node** — an *indexed* heap, not the reference's lazy heap that can hold up to `m` stale entries. This is the single biggest algorithmic difference: `pop` count is `n`, not `m`, and there is no `done[]`/stale-skip check |
| "its ticket gets pulled out and slotted back into its new, better position" | **decrease-key**: `pos[v]` records where node `v`'s entry lives, and a relaxation sifts that entry *up* in place instead of pushing a duplicate |
| "the line is always kept in order" | heap order, but as an **aligned 4-ary heap**: children `4i+1..4i+4` are 16-byte items, and the array base is shifted by 3 slots so each sibling group of four lands on exactly one 64-byte cache line. Depth is `log₄n` instead of `log₂n`, and each level's 4 comparisons are one cache line and become branchless `minsd`/`cmov` |
| "take the ticket at the very front — nobody walks the meadow" | `H[0]`, no O(n) scan (the assumption being killed) |
| "locked in, then its tunnels get checked" | settle-then-relax, kept exactly; CSR adjacency |

Implementation work beyond the mechanism itself (mechanism unchanged):
- **`dist` and `pos` fused** into one 16-byte record per node, so a relaxation of edge `u→v` costs **one** cache miss instead of two (`dist[v]` + `pos[v]`). `dist_out` is filled by one sequential pass at the end.
- **Hole-based sifting** (shift items into the hole, place the payload once) instead of swap-based sifting — halves the stores per level.
- **CSR built in one counting pass + one scatter**, no `fill[]` array (the cursor is `off` itself, un-shifted afterwards), and sortedness of `src[]` is detected for free during the counting pass: **if the edge list is already grouped by source (very common for generated graphs), the adjacency arrays alias the caller's `dst`/`weight` and the whole O(m) random-write scatter and its 16 MB allocation are skipped.** Two specializations of the main loop (aliased split arrays vs. packed 16-byte edges) are generated from one macro so they can't drift.
- **Software prefetch**: `&off[H[0].u]` is issued at the end of each `pop` (the sift-down hides the latency of the next iteration's `off` load), and `nodes[v]` is prefetched 4 edges ahead — but only for nodes with degree ≥ 8, so short adjacency lists don't pay for the extra instructions.
- **No `done[]` array at all.** With non-negative weights and monotone FP addition, `fl(dist[u]+w) ≥ dist[u] ≥ dist[v]` for any already-settled `v`, so the `nd < dist[v]` test alone can never re-open a settled node. (`pos[u] = -1` is still written on pop — one store per node — so a stale index can never be followed even if an input violated the contract.)

No dense O(n²) fallback: that would be switching mechanisms, which was ruled out.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#if defined(__GNUC__) || defined(__clang__)
#  define PF(p)       __builtin_prefetch((const void *)(p))
#  define LIKELY(x)   __builtin_expect(!!(x), 1)
#else
#  define PF(p)       ((void)0)
#  define LIKELY(x)   (x)
#endif

/* 16-byte records: one node record = one cache-line-safe unit (d and pos fused,
   so a relaxation touches a single line); heap items carry their key inline so
   sifting never chases an indirection. */
typedef struct { double d; int pos; int pad; } NRec;
typedef struct { double d; int u;   int pad; } HItem;
typedef struct { double w; int v;   int pad; } PEdge;

/* sift-up: used both for insert (i == hs) and for decrease-key (i == pos[u]).
   Hole-based: parents are shifted down into the hole, payload placed once. */
static inline void h_up(HItem *restrict H, NRec *restrict nd,
                        int i, double d, int u)
{
    while (i > 0) {
        int p = (i - 1) >> 2;          /* 4-ary parent */
        double pd = H[p].d;
        if (pd <= d) break;
        int pu = H[p].u;
        H[i].d = pd; H[i].u = pu;
        nd[pu].pos = i;
        i = p;
    }
    H[i].d = d; H[i].u = u; nd[u].pos = i;
}

/* pop root: move last item to the root and sift it down; aligned 4-ary groups
   let the 4 child keys be compared branchlessly out of one cache line. */
static inline int h_pop(HItem *restrict H, NRec *restrict nd,
                        int hs, const int *restrict off)
{
    hs--;
    if (hs <= 0) return 0;
    double d = H[hs].d;
    int    u = H[hs].u;
    int i = 0;
    for (;;) {
        int c = 4 * i + 1;
        if (c >= hs) break;
        double bd; int b;
        if (LIKELY(c + 3 < hs)) {
            double d0 = H[c].d, d1 = H[c + 1].d, d2 = H[c + 2].d, d3 = H[c + 3].d;
            int k1 = (d1 < d0); double m01 = k1 ? d1 : d0;
            int k2 = (d3 < d2); double m23 = k2 ? d3 : d2;
            if (m23 < m01) { bd = m23; b = c + 2 + k2; }
            else           { bd = m01; b = c + k1;     }
        } else {
            bd = H[c].d; b = c;
            for (int k = c + 1; k < hs; k++)
                if (H[k].d < bd) { bd = H[k].d; b = k; }
        }
        if (bd >= d) break;
        int bu = H[b].u;
        H[i].d = bd; H[i].u = bu;
        nd[bu].pos = i;
        i = b;
    }
    H[i].d = d; H[i].u = u; nd[u].pos = i;
    PF(&off[H[0].u]);                  /* next iteration's CSR offset */
    return hs;
}

#define RELAX_ONE(VV, WW)                                                   \
    do {                                                                    \
        int v = (VV); double ndv = du + (WW);                               \
        if (ndv < nodes[v].d) {                                             \
            nodes[v].d = ndv;                                               \
            int p = nodes[v].pos;                                           \
            if (p < 0) { h_up(H, nodes, hs, ndv, v); hs++; }  /* insert  */ \
            else         h_up(H, nodes, p,  ndv, v);          /* dec-key */ \
        }                                                                   \
    } while (0)

#define DIJ_LOOP(EV, EW)                                                    \
    while (hs > 0) {                                                        \
        int u = H[0].u; double du = H[0].d;                                 \
        int e = off[u], end = off[u + 1];                                    \
        hs = h_pop(H, nodes, hs, off);                                       \
        nodes[u].pos = -1;                                                   \
        if (end - e >= 8) {                                                  \
            int lim = end - 4;                                               \
            for (; e < lim; e++) {                                           \
                PF(&nodes[EV(e + 4)]);                                       \
                RELAX_ONE(EV(e), EW(e));                                     \
            }                                                                \
        }                                                                    \
        for (; e < end; e++) RELAX_ONE(EV(e), EW(e));                        \
    }

#define SV(i) (av[i])
#define SW(i) (aw[i])
#define PV(i) (Ep[i].v)
#define PW(i) (Ep[i].w)

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (m < 0) m = 0;
    if (source < 0 || source >= n) {
        for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
        return;
    }

    /* ---------- CSR offsets: one counting pass, sortedness detected free ---- */
    int *restrict off = (int *)malloc((size_t)(n + 2) * sizeof(int));
    if (!off) { for (int i = 0; i < n; i++) dist_out[i] = INFINITY; return; }
    memset(off, 0, (size_t)(n + 1) * sizeof(int));

    int sorted = 1;
    {
        int prev = 0;
        for (int i = 0; i < m; i++) {
            int s = src[i];
            off[s + 1]++;
            sorted &= (s >= prev);
            prev = s;
        }
    }
    for (int i = 0; i < n; i++) off[i + 1] += off[i];

    const int    *restrict av = dst;      /* zero-copy path                   */
    const double *restrict aw = weight;
    PEdge *restrict Ep = NULL;            /* packed path (unsorted input)     */

    if (!sorted && m > 0) {
        Ep = (PEdge *)malloc((size_t)m * sizeof(PEdge));
        if (!Ep) { free(off); for (int i = 0; i < n; i++) dist_out[i] = INFINITY; return; }
        for (int i = 0; i < m; i++) {
            int p = off[src[i]]++;        /* off doubles as the fill cursor   */
            Ep[p].w = weight[i];
            Ep[p].v = dst[i];
        }
        for (int u = n; u > 0; u--) off[u] = off[u - 1];   /* un-shift */
        off[0] = 0;
    }

    /* ---------- node records + 64B-aligned 4-ary heap in one block ---------- */
    size_t nb = (size_t)n * sizeof(NRec);
    size_t hb = (size_t)(n + 3) * sizeof(HItem);
    void *raw = malloc(nb + hb + 192);
    if (!raw) { free(off); free(Ep); for (int i = 0; i < n; i++) dist_out[i] = INFINITY; return; }
    uintptr_t p0 = ((uintptr_t)raw + 63) & ~(uintptr_t)63;
    NRec *restrict nodes = (NRec *)p0;
    uintptr_t p1 = (p0 + nb + 63) & ~(uintptr_t)63;
    HItem *Hbase = (HItem *)p1;
    HItem *restrict H = Hbase + 3;   /* logical i at Hbase[i+3] => each child
                                        group 4i+1..4i+4 is one 64B line     */

    for (int i = 0; i < n; i++) { nodes[i].d = INFINITY; nodes[i].pos = -1; }
    nodes[source].d = 0.0;

    int hs = 0;
    h_up(H, nodes, 0, 0.0, source);
    hs = 1;

    if (Ep) { DIJ_LOOP(PV, PW) }
    else    { DIJ_LOOP(SV, SW) }

    for (int i = 0; i < n; i++) dist_out[i] = nodes[i].d;

    free(raw);
    free(off);
    free(Ep);
}
```

## PREDICTION

PREDICTION: 2.2

(Meaning: ~2.2× faster than the supplied binary-heap/lazy-deletion reference on the benchmark mix — roughly 1.5–1.8× from the indexed decrease-key 4-ary heap plus fused `dist`/`pos`, and the rest from the zero-copy CSR path if the generated edge list is grouped by source; ~1.5× if it is not.)

## MEASUREMENT

**Not performed — I could not measure this.** The task listed `dijkstra_bench` and `dijkstra_contract`, but this session was started with no tools available (no bench, no compiler, no shell execution), so there was no way to compile, verify against the reference, or time anything. I am not going to report numbers I did not observe, and I did not get to use any of the four allowed improvement rounds: the artifact above is a first cut, unmeasured.

What I can state without measuring:
- **Contract**: signature, argument order, and semantics match exactly (`dist_out[i]` = shortest distance, `INFINITY` for unreachable, `dist_out[source] = 0`); no input array is modified; all scratch is freed; `m == 0`, `n == 1`, self-loops, parallel edges, zero weights, and unreachable components are all handled by the same code path.
- **Invariants I reasoned through rather than tested**: heap capacity ≤ `n` (guaranteed by `pos`), no out-of-bounds read in the branchless min-of-4 (guarded by `c + 3 < hs`), the `off` cursor un-shift restores exact CSR boundaries, and correctness without a `done[]` array follows from `fl(a+w) ≥ a` for `w ≥ 0` plus non-decreasing pop keys.
- **Untested risks I'd look at first with a bench in hand**: (1) whether the per-edge prefetch pays or costs on small in-cache graphs — the degree-≥8 gate is a guess, not a measurement; (2) whether 4-ary actually beats binary here (it usually does, but the `n`/`m` mix decides); (3) whether the `sorted` fast path ever fires on this generator — if it never does, the split/packed duplication is dead weight and the packed scatter is on the critical path.

## VERDICT

The mechanism is translated faithfully and, I believe, in about the strongest form it has: one ticket per node with real decrease-key (`≤ n` heap entries instead of `≤ m`), a cache-line-aligned 4-ary line instead of a binary one, hole-based sifting, `dist`+`pos` fused to one miss per relaxation, no `done[]`, and a CSR build that skips its most expensive pass whenever the input permits. On paper it should comfortably beat the reference; **on this machine it is unverified, and the honest status is "plausible, unmeasured," not "faster."** The prediction of 2.2 stands as a claim to be checked, and if the bench comes back below ~1.3× my first suspicion is the prefetch gate and the 4-ary arity, in that order.