## APPROACH

**Mapping the fable onto the real problem.** Pip the magpie with nesting boxes is a *priority queue* — the "smallest number always sitting right on top, instantly grabbable" is `O(1)` find-min, replacing the council's "walk the entire list of unsettled trees" (`O(n)` linear scan). The crucial detail is *"Pip slides that tree's marker to its correct spot in a few quick moves, without disturbing everything else"*: **one marker per tree**, repositioned in place. That is an **indexed heap with decrease-key**, not the lazy multi-push heap of the reference — each node occupies at most one heap slot, so the heap never exceeds `n` entries and there are no stale pops to discard. "Stars that tree, settled, and checks its bridges exactly as before" keeps Dijkstra's settle-then-relax invariant untouched, so the answer is bit-identical to the reference ordering semantics.

So the mechanism is fixed (heap Dijkstra); all my effort goes into *how* the heap and the graph are represented:

1. **4-ary heap instead of binary.** Depth `log₄ n` = half the levels, halving the pointer-chasing on both sift-up (the dominant op — one per successful relaxation) and sift-down (one per pop).
2. **Cache-line-aligned sibling groups.** Heap entries are 16-byte `{double key; int node; int pad}` records, and the array is offset by 3 elements from a 64-byte boundary so that the four children `4i+1…4i+4` always land in **exactly one 64-byte cache line**. A sift-down level costs one cache miss, not four, and the key sits next to its node id (no random probe back into `dist[]` to compare).
3. **Indexed decrease-key** via `pos[]`: `pos[v] ≥ 0` → in heap (sift up), `-1` → absent. A settled node can never re-enter, because with non-negative weights `du + w ≥ du ≥ dist[v_settled]` holds exactly in IEEE arithmetic, so `nd < dist[v]` is always false for settled `v`. No `done[]` array needed at all.
4. **CSR built with the cursor-shift trick**: count into `off[u+1]`, prefix-sum, scatter using `off[u]++` as the cursor, then shift `off` right by one. This eliminates the reference's separate `deg[]` and `fill[]` arrays (two fewer `n`-sized allocations and two fewer random-write passes).
5. **Split `edst`/`ew` arrays** (12 B/edge streamed) rather than a padded 16-byte struct, plus a modest prefetch of `dist_out[edst[e+4]]` in the relaxation loop, since that scatter-read is the only truly random access in the inner loop.

Gus's bureaucracy (finalize-before-explore) is kept — it's what makes the algorithm correct — but the `O(n)` scan and the redundant duplicate markers are gone.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct { double d; int u; int pad; } HItem;

/* move item (key,u) up from slot i in a 4-ary heap */
static inline void hp_up(HItem *h, int *pos, int i, double key, int u)
{
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= key) break;
        h[i] = h[p];
        pos[h[i].u] = i;
        i = p;
    }
    h[i].d = key; h[i].u = u; pos[u] = i;
}

/* move item (key,u) down from slot i in a 4-ary heap of given size */
static inline void hp_down(HItem *h, int *pos, int size, int i, double key, int u)
{
    for (;;) {
        int c = (i << 2) + 1;
        if (c >= size) break;
        int lim = c + 4; if (lim > size) lim = size;
        int best = c; double bd = h[c].d;          /* the 4 siblings share one cache line */
        for (int j = c + 1; j < lim; j++) {
            double t = h[j].d;
            if (t < bd) { bd = t; best = j; }
        }
        if (bd >= key) break;
        h[i] = h[best];
        pos[h[i].u] = i;
        i = best;
    }
    h[i].d = key; h[i].u = u; pos[u] = i;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- CSR: count -> prefix -> scatter (cursor) -> shift back ---- */
    int *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !edst || !ew) { free(off); free(edst); free(ew); return; }

    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    for (int i = 0; i < m; i++) {
        int p = off[src[i]]++;
        edst[p] = dst[i];
        ew[p]  = weight[i];
    }
    for (int i = n; i > 0; i--) off[i] = off[i - 1];
    off[0] = 0;

    /* ---- indexed 4-ary heap, sibling groups on one 64B line ---- */
    int *pos = (int *)malloc((size_t)n * sizeof(int));
    void *hraw = malloc((size_t)(n + 4) * sizeof(HItem) + 64);
    if (!pos || !hraw) { free(off); free(edst); free(ew); free(pos); free(hraw); return; }
    HItem *harr = (HItem *)(((uintptr_t)hraw + 63u) & ~(uintptr_t)63u);
    HItem *h = harr + 3;   /* children 4i+1..4i+4 -> harr[4(i+1)..4(i+1)+3] : one cache line */

    for (int i = 0; i < n; i++) pos[i] = -1;

    int size = 1;
    h[0].d = 0.0; h[0].u = source; pos[source] = 0;

    while (size > 0) {
        double du = h[0].d;
        int u = h[0].u;
        pos[u] = -1;                       /* settled: can never re-enter (w >= 0) */
        size--;
        if (size > 0) hp_down(h, pos, size, 0, h[size].d, h[size].u);

        int e = off[u], e1 = off[u + 1];
        for (; e < e1; e++) {
            if (e + 4 < e1) __builtin_prefetch(&dist_out[edst[e + 4]], 1, 1);
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                int p = pos[v];
                if (p >= 0) hp_up(h, pos, p, nd, v);      /* decrease-key in place */
                else        hp_up(h, pos, size++, nd, v); /* first sighting: insert */
            }
        }
    }

    free(off); free(edst); free(ew); free(pos); free(hraw);
}
```

PREDICTION: 1.8

## MEASUREMENT

**Not measured.** No tools were available in this session — `dijkstra_bench` and `dijkstra_contract` could not be invoked, so the 1.8× figure above stands as an unverified prediction, not a result. Reporting that plainly rather than inventing numbers.

What I can assert without the harness, by inspection:

- **Contract compliance**: signature, argument order, and semantics match exactly; `dist_out[i]` is `INFINITY` for unreachable `i`, `0.0` at `source`; no input buffer is written; all scratch is freed on every exit path (including allocation failure).
- **Exactness vs. reference**: the relaxation arithmetic is the identical `du + w` in the identical settle order, so results are bit-identical to the reference, not merely within tolerance.
- Where the predicted gain would come from, if it comes at all: ~2× fewer heap levels (4-ary), one cache miss per sift-down level instead of up to four (aligned sibling groups), heap bounded by `n` instead of `m` (decrease-key), and two fewer `n`-sized passes in CSR construction.

**Risk I would test first if I could measure**: on very sparse graphs the `pos[]` write on every heap move can cost more than it saves, and lazy insertion (the reference's scheme) sometimes wins there. If the benchmark showed that, the fix inside this mechanism would be to keep the aligned 4-ary heap but drop `pos[]` for lazy push + skip-on-stale — same structure, different bookkeeping. The prefetch distance (4) is the other obvious knob.

## VERDICT

Faithful translation of the disguised mechanism: Pip's nesting boxes → an indexed 4-ary heap with `O(1)` find-min and in-place decrease-key; "settle, then check its bridges" → unchanged Dijkstra relaxation; the council's full-list scan → eliminated. The artifact is complete, correct by construction, and predicted at **1.8×** over the reference binary-heap version — but that prediction is **unverified**, because this session had no benchmark tool. Treat the speedup as a claim awaiting measurement, and the correctness as the part I'm actually confident in.