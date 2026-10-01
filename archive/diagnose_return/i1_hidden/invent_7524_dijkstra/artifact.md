## MAPPING

**SEED 1** — *"A glint released from the traveler's jewel crawls down every cord at a pace set by the cord's own length."*

| World object | Problem object |
|---|---|
| jewel hung for a place | node `0..n-1` |
| infinite net threading the jewels | the directed graph |
| blood-dark cord, jewel→jewel | directed edge `src[i] -> dst[i]` |
| length of the cord (taut/slack) | `weight[i] >= 0` |
| the one jewel I touch | `source` |
| glint | a tentative arrival (a distance label in flight) |
| "moving no faster than the cord is long" | arrival time `= departure time + w` |
| glint *waiting on the way* down a long cord | the label parked in a **time-indexed slab (bucket)** until the clock reaches it |
| "must travel every cord it can reach" | *all* out-edges of a lit jewel fire at once, immediately, with no permission asked |
| my crouching / the clock I wait on | a monotone simulated-time cursor `kcur`, advancing, never rewinding |

**SEED 2** — *"The first flare is chalked once; later flares along longer cords are thrown away unmarked."*

| World object | Problem object |
|---|---|
| chalk grid, **one square per jewel** | the flat array `dist_out[n]` — *not* a ranked structure |
| white-dust number = instant of first fire | `dist_out[v]` = shortest distance |
| a later, slacker flare | a relaxation with `nd >= dist_out[v]` |
| "thrown away without marking twice" | the `if (nd < dist_out[v])` test; no re-enqueue, no duplicate work |

**SEED 3** — *"A square that never catches any flare is left blank and crossed out."*

| World object | Problem object |
|---|---|
| square that stays dark | node never relaxed |
| the crossing scratch | `INFINITY` written at init and never overwritten |
| "however long I crouch" | loop terminates when the queue empties, not when all squares are filled |

## CHOSEN SEED

**SEED 1.** It is the only one that names a *mechanism* rather than a bookkeeping rule, and every one of its nouns has a hard computational counterpart. Decisively: the native owns **no ranking device**. There is no jewel-comparing, no "which jewel is smallest", no heap. There is a *clock* and a *grid of squares*. The ordering work is done by the cords themselves — physical delay — not by a comparison. The literal translation of "the glint waits on the way, a length of time set by the cord" is a **calendar / bucket queue indexed by arrival time**, and the literal translation of "one square per jewel, marked in dust" is a flat array. Replacing the binary heap with a clock is the whole artifact.

## ASSUMPTION BROKEN

> *"A road can only be considered once its starting place is fully settled."*

The native never settles anything. A jewel flares and **instantly** fires every cord it has; nothing checks first whether that flare is the final one. I take this completely literally: the clock advances in slabs of width `delta`, and every jewel in the current slab fires its edges even though a jewel later in the same slab may still improve it. When that happens — when a smaller flare reaches an already-fired jewel *within the same slab* — the jewel simply **flares again** (it is re-inserted into the same slab and re-drained). No jewel is ever declared settled, at any point, which is exactly why the slab width is allowed to exceed the shortest cord. Correctness therefore does not rest on extraction order at all: it rests on the fact that the run only stops when no cord is left unrelaxed (a least-fixed-point argument), so even a mis-binned jewel cannot produce a wrong number — only extra flares.

Consequences kept literal:
- "first light is the only light" → strict `<` test, one write.
- "crossed out like a wind-erased track" → `INFINITY`, untouched.
- the only global state is a clock (`kcur`), a slab table (`bhead`), and the chalk grid (`dist_out`).

## ARTIFACT

Slab width choice (the one piece of engineering the native doesn't dictate): `delta = wminp` (shortest cord) whenever the circular window can then still span `wmax` — that is Dial's algorithm and reproduces exact Dijkstra order with *zero* re-flares. Only when the dynamic range is too wide for that does `delta` coarsen toward `wmax/(NBMAX-128)`, trading a few re-flares for a bounded window. A 64-bit occupancy bitmap makes skipping dark slabs cost 1/64 of a check each. Per step 4, the risk my own mechanism carries — a re-flare storm when the weight range is so wide that slabs must be coarser than typical cords — is guarded by a hard flare budget (`4n + 65536`) that aborts to a plain lazy binary-heap Dijkstra, so the worst case is bounded by the baseline rather than by Bellman–Ford.

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define DJ_NBMAX 16384   /* slab-table size, multiple of 64 */

/* circular search for the next lit slab, 64 slabs per word test */
static int dj_scan(const uint64_t *bm, int nw, int p)
{
    int w = p >> 6, b = p & 63;
    uint64_t x = bm[w] >> b;
    if (x) return (int)__builtin_ctzll(x);
    int sk = 64 - b;
    for (int i = 1; i <= nw; ++i) {
        int ww = w + i; if (ww >= nw) ww -= nw;
        if (bm[ww]) return sk + (int)__builtin_ctzll(bm[ww]);
        sk += 64;
    }
    return -1;
}

/* guarded fallback: textbook lazy binary-heap Dijkstra on the same CSR */
static void dj_heap(int n, const int *off, const int *eto, const double *ew,
                    int source, double *dist)
{
    int cap = n + 64, hs = 0;
    double *hk = (double*)malloc((size_t)cap * sizeof(double));
    int    *hv = (int*)   malloc((size_t)cap * sizeof(int));
    if (!hk || !hv) { free(hk); free(hv); return; }
    for (int i = 0; i < n; ++i) dist[i] = INFINITY;
    dist[source] = 0.0;
    hk[hs] = 0.0; hv[hs] = source; ++hs;
    while (hs > 0) {
        double bk = hk[0]; int bu = hv[0];
        --hs;
        if (hs > 0) {
            double k = hk[hs]; int v = hv[hs]; int i = 0;
            for (;;) {
                int c = 2*i + 1; if (c >= hs) break;
                if (c + 1 < hs && hk[c+1] < hk[c]) ++c;
                if (hk[c] >= k) break;
                hk[i] = hk[c]; hv[i] = hv[c]; i = c;
            }
            hk[i] = k; hv[i] = v;
        }
        if (bk > dist[bu]) continue;
        int e1 = off[bu+1];
        for (int e = off[bu]; e < e1; ++e) {
            int v = eto[e];
            double nd = bk + ew[e];
            if (nd < dist[v]) {
                dist[v] = nd;
                if (hs == cap) {
                    int nc = cap + (cap >> 1) + 64;
                    double *t1 = (double*)realloc(hk, (size_t)nc * sizeof(double));
                    if (!t1) { free(hk); free(hv); return; }
                    hk = t1;
                    int *t2 = (int*)realloc(hv, (size_t)nc * sizeof(int));
                    if (!t2) { free(hk); free(hv); return; }
                    hv = t2; cap = nc;
                }
                int i = hs++;
                while (i > 0) {
                    int par = (i - 1) >> 1;
                    if (hk[par] <= nd) break;
                    hk[i] = hk[par]; hv[i] = hv[par]; i = par;
                }
                hk[i] = nd; hv[i] = v;
            }
        }
    }
    free(hk); free(hv);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;   /* every square crossed out */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;                                /* the one touched jewel */
    if (m <= 0) return;

    /* ---- the net: CSR, plus the cord-length statistics, in one pass ---- */
    int    *off = (int*)   calloc((size_t)n + 1, sizeof(int));
    int    *cur = (int*)   malloc((size_t)n * sizeof(int));
    int    *eto = (int*)   malloc((size_t)m * sizeof(int));
    double *ew  = (double*)malloc((size_t)m * sizeof(double));
    if (!off || !cur || !eto || !ew) { free(off); free(cur); free(eto); free(ew); return; }

    double wmax = 0.0, wsum = 0.0, wminp = INFINITY;
    for (int i = 0; i < m; ++i) {
        unsigned s = (unsigned)src[i], t = (unsigned)dst[i];
        if (s < (unsigned)n && t < (unsigned)n) off[s + 1]++;
        double w = weight[i];
        if (w > wmax) wmax = w;
        wsum += w;
        if (w > 0.0 && w < wminp) wminp = w;
    }
    for (int i = 0; i < n; ++i) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; ++i) {
        unsigned s = (unsigned)src[i], t = (unsigned)dst[i];
        if (s < (unsigned)n && t < (unsigned)n) {
            int p = cur[s]++; eto[p] = (int)t; ew[p] = weight[i];
        }
    }

    /* every cord of length zero: one flash lights all reachable jewels at 0 */
    if (!(wmax > 0.0)) {
        int *stk = cur, top = 0;
        stk[top++] = source;
        while (top) {
            int u = stk[--top], e1 = off[u+1];
            for (int e = off[u]; e < e1; ++e) {
                int v = eto[e];
                if (dist_out[v] != 0.0) { dist_out[v] = 0.0; stk[top++] = v; }
            }
        }
        free(off); free(cur); free(eto); free(ew);
        return;
    }
    free(cur);

    /* ---- slab width: how finely the clock is read ---- */
    double dneed = wmax / (double)(DJ_NBMAX - 128);       /* coarsest window that spans wmax */
    double dheur = (wsum / (double)m) * ((double)n / (double)m); /* mean cord / mean degree */
    double delta = wminp;                                  /* ideal: exact order, no re-flare */
    if (dneed > delta) {
        delta = (dneed < dheur) ? dneed : dheur;
        if (delta < wminp) delta = wminp;
    }
    if (!(delta > 0.0) || !isfinite(delta)) delta = 1.0;

    long long need = (long long)(wmax / delta) + 2;
    if (need > DJ_NBMAX - 128) need = DJ_NBMAX - 128;
    int nb = (int)(((need + 63) / 64 + 2) * 64);
    if (nb < 192) nb = 192;
    if (nb > DJ_NBMAX) nb = DJ_NBMAX;
    int nw = nb >> 6;
    long long lim = (long long)nb - 65;
    double invd = 1.0 / delta;

    int      *bhead = (int*)     malloc((size_t)nb * sizeof(int));
    uint64_t *bm    = (uint64_t*)calloc((size_t)nw, sizeof(uint64_t));
    int      *nxt   = (int*)     malloc((size_t)n * sizeof(int));
    int      *prv   = (int*)     malloc((size_t)n * sizeof(int));
    int      *inb   = (int*)     malloc((size_t)n * sizeof(int));
    if (!bhead || !bm || !nxt || !prv || !inb) {
        if (bhead && bm && nxt && prv && inb) { /*unreachable*/ }
        dj_heap(n, off, eto, ew, source, dist_out);
        free(bhead); free(bm); free(nxt); free(prv); free(inb);
        free(off); free(eto); free(ew);
        return;
    }
    for (int i = 0; i < nb; ++i) bhead[i] = -1;
    memset(inb, 0xFF, (size_t)n * sizeof(int));            /* -1 : in no slab */

#define DJ_INS(V,B) do { int _v=(V), _b=(B); int _h=bhead[_b];                 \
        nxt[_v]=_h; prv[_v]=-1; if(_h>=0) prv[_h]=_v; bhead[_b]=_v;            \
        inb[_v]=_b; bm[_b>>6] |= (uint64_t)1<<(_b&63); } while(0)
#define DJ_REM(V) do { int _v=(V), _b=inb[_v], _p=prv[_v], _q=nxt[_v];         \
        if(_p>=0) nxt[_p]=_q;                                                  \
        else { bhead[_b]=_q; if(_q<0) bm[_b>>6] &= ~((uint64_t)1<<(_b&63)); }  \
        if(_q>=0) prv[_q]=_p; inb[_v]=-1; } while(0)

    long long kcur = 0, pops = 0, budget = 4*(long long)n + 65536;
    int nwin = 1, bail = 0;
    DJ_INS(source, 0);

    while (nwin > 0) {
        int p = (int)(kcur % nb);
        if (bhead[p] < 0) {                                /* crouch: skip dark slabs */
            int sk = dj_scan(bm, nw, p);
            if (sk < 0) break;
            kcur += sk;
            p = (int)(kcur % nb);
        }
        while (bhead[p] >= 0) {                            /* drain this instant */
            int u = bhead[p], q = nxt[u];
            bhead[p] = q;
            if (q >= 0) prv[q] = -1;
            else bm[p>>6] &= ~((uint64_t)1<<(p&63));
            inb[u] = -1; --nwin;
            if (++pops > budget) { bail = 1; break; }      /* guard: re-flare storm */
            double du = dist_out[u];
            int e1 = off[u+1];
            for (int e = off[u]; e < e1; ++e) {            /* fire every cord at once */
                int v = eto[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) {                    /* first/earlier light only */
                    dist_out[v] = nd;
                    long long bi = (long long)(nd * invd);
                    if (bi < kcur) bi = kcur;
                    else if (bi > kcur + lim) bi = kcur + lim;
                    int nbk = (int)(bi % nb), ob = inb[v];
                    if (ob != nbk) {
                        if (ob >= 0) { DJ_REM(v); } else ++nwin;
                        DJ_INS(v, nbk);
                    }
                }
            }
        }
        if (bail) break;
        ++kcur;
    }
#undef DJ_INS
#undef DJ_REM

    if (bail) dj_heap(n, off, eto, ew, source, dist_out);

    free(bhead); free(bm); free(nxt); free(prv); free(inb);
    free(off); free(eto); free(ew);
}
```

PREDICTION: speedup_vs_heap = 2.0

Reasoning behind the number, stated before any measurement: CSR construction is shared with the baseline and is memory-bound, so it caps the achievable ratio; within the search itself, every push/pop becomes O(1) pointer surgery instead of O(log n) sift with `log n ≈ 20` comparisons and cache-hostile heap traversal, and dark-slab skipping costs one `ctz` per 64 empty instants. I expect the search phase to run 3–4× faster and the whole kernel, build included, to land near 2×.

## MEASUREMENT

**Not performed — honest statement.** No tools were available in this session: `dijkstra_bench` and `dijkstra_contract` could not be called, so there are zero measurements behind this artifact and zero of the four permitted improvement rounds were taken. The prediction above is unvalidated. What I can assert is the argument, not the number:

- **Contract**: signature and argument order copied verbatim; `dist_out` is filled with `INFINITY` first and `dist_out[source] = 0.0`; unreachable nodes are never written again.
- **Correctness argument** (independent of slab arithmetic): `dist_out[v]` is only ever assigned `dist_out[u] + w`, so every value is the length of a real path; the run halts only when no queued node remains, and every strict improvement re-queues its node, so at halt `dist[v] <= dist[u] + w` holds for all edges. A least fixed point of that system with non-negative weights is the shortest-distance vector. This means a mis-binned slab index (from `nd * invd` rounding, from the `kcur` clamp, or from a coarse `delta`) can cost extra flares but cannot produce a wrong distance.
- Zero-weight cords, all-zero-weight graphs, a source with no out-edges, and out-of-range endpoints are each handled explicitly.

## VERDICT

The native's picture is not a metaphor for Dijkstra — it is a different machine that computes the same answer. The heap is a *ranking* device; the native has no ranking device, only a clock and a grid of chalk squares. Translating literally therefore forces exactly one substitution — comparison-based priority queue → simulated-time slab calendar — and that substitution is the entire performance thesis. The refusal to settle ("I touch, and the touch *must* travel every cord it can reach", no permission sought) is what licenses slabs wider than the shortest cord, i.e. what makes the calendar cheap.

Named risk, and how it is guarded rather than waved at: if the weight dynamic range is wide enough that the window cannot span `wmax` at `delta = wminp`, slabs coarsen, many jewels share one instant, and re-flaring can approach Bellman–Ford. Three guards, all in the shipped code: (1) `delta` prefers `wminp` and only coarsens when forced, which keeps re-flares at exactly zero for equal, integer, or moderate-range weights; (2) the slab window is bounded at 16384 with out-of-window insertions clamped to its far end, so no aliasing and no unbounded table; (3) a hard flare budget of `4n + 65536` aborts to a plain lazy binary-heap Dijkstra, so the pathological case degrades to roughly the baseline instead of to quadratic. Residual honest exposure: a false trip of that budget wastes up to `4n` pops before the fallback, costing perhaps 0.6× instead of 2×; and the untested code paths — far-end clamp, zero-weight traversal, budget abort — have never been compiled or run here. If the measured speedup comes back below 1.0, the first thing I would inspect is whether the budget is tripping on the benchmark's weight distribution, and the first fix would be to raise `DJ_NBMAX` so `delta` can stay at `wminp`.