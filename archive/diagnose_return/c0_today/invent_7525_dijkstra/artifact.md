## MAPPING

**SEED 1 — "nightingales sent down every road sing one note per stride and die at the first board, carving that board's lattice slot"**

| world object | computational object |
|---|---|
| traveler's place | `source` |
| a place / a cross-marked board | a node `0..n-1` |
| a road out of a place | one directed edge `src[i]->dst[i]` |
| stride of road beneath the bird | one unit of `weight[i]` |
| note growing hoarser with distance | accumulated tentative distance `dist[u] + w` |
| "loose them all at once" for every road out of the place | the inner loop over one CSR row (all out-edges of the settled node), issued as one batch |
| bird dies at the **first** board on its path | the edge's head `dst[e]` — one hop only, never a multi-hop flight |
| carving the dying note into the slot | `dist_out[v] = nd`, a direct store |
| **rigid lattice of old memory, one fixed slot per place** | a flat `double[n]` — *not* a dynamic priority structure; the array is the sole source of truth |
| keep the fainter note, **let the louder rot** | min-update only; a superseded entry is never removed, it is ignored when it later surfaces (lazy deletion, no `done[]` array) |
| dead roads marked once, never a bird again | the `nd < dist[v]` guard rejecting an edge once; every edge is scanned exactly once |

*Silent assumption broken:* **"a priority structure must be consulted before every relaxation."** A relaxation here is a bird dying into a fixed lattice slot; nothing is asked of any queue. (It also kills the `done[]` array: "let the louder rot" is exactly a stale-key skip.)

**SEED 2 — "at each board a cow's-head woman milks a smaller cow to re-measure the note by way of the last settled board"**

| world object | computational object |
|---|---|
| opening a cow's head at the board | the relaxation test performed at the node, not at the queue |
| a woman inside milking a *smaller* cow | the nested, self-similar inner loop (edges of `u`) inside the outer settle loop |
| the milk she draws | candidate `du + ew[e]` |
| "by way of the last board I settled, plus the stretch between" | `du` = distance of the just-settled node; `ew[e]` = the edge |
| milk thinner than what's carved | `nd < dist_out[v]` |
| scrape the slot clean, cut hers | overwrite in place |

*Silent assumption broken:* same one — the truth about `v` is re-derived locally from the last settled board, so the priority structure is never queried about `v`'s presence or key.

**SEED 3 — "the quietest unsettled note is always settled next; unreached places stand blank as the desert"**

| world object | computational object |
|---|---|
| quietest unsettled note in the lattice | minimum over the tentative array, restricted to unsettled nodes |
| chosen next to settle | extract-min |
| never touching a settled board twice | removal from the active set / stale skip |
| no bird sings and no cow yields thinner milk | frontier empty → terminate |
| blank as the pink-brown desert | `INFINITY` for unreachable nodes |
| "I read the traveler the whole lattice in order" | `dist_out[0..n-1]`, all `n` values |

*Silent assumption broken:* it does **not** break "the next place to finalize is found by comparing against every remaining place" — it **embraces** it, i.e. it licenses the heap-free O(n²) scan as a first-class path rather than an embarrassment.

**Plainly: none of the three seeds breaks "the whole graph must be explored to know any single distance."** The native's closing act is reading the traveler *the whole lattice in order* — which is precisely what the contract demands (all `n` distances, `INFINITY` for unreachable). There is no in-world hook for early termination, and I will not invent one.

## CHOSEN SEED

**SEED 1**, by the fallback rule (most literal, most unlike a binary heap). Its central object — *"the rigid lattice of old memory that keeps one fixed slot for every place"* — is the exact negation of the reference's heap, which holds up to `m` 16-byte items and a parallel `done[]` array. Seeds 2 and 3 are not competitors: in the literal reading they are Seed 1's inner loop (the cow's head = relaxation) and outer loop (quietest-first = extract-min), so the chosen mapping composes all three.

**Regime reading, from the native's own practice (required by step 5 — `known_way` names two regimes).** Before loosing a single bird the native paces the land and counts boards against roads: *if reading the whole lattice at every board would cost more strides than the birds will ever sing, he does not read the lattice at all — he hangs the unsettled notes in a roost, faintest on top, and takes the top bird each time; if the land is small, the flat lattice read straight through (four notes heard at once) is cheaper than any roost.* That is a runtime `n` vs `m` cost comparison choosing between an O(n²) SIMD lattice scan and a heap, with all other machinery shared.

Per step 4, I let this arrive at **validated known techniques** rather than novelties: the flat-lattice path *is* array-scan Dijkstra (explicitly named as a real win for small/dense graphs), and the roost is a **lazy 4-ary heap with cache-line-aligned child groups and stale-key skipping** — a standard, measured improvement over a lazy binary heap with a `done[]` array. Arity 4 and the 64-byte alignment are my engineering addition, consistent with "four notes at once" but not derived from the native's words; I say so rather than pretending the metaphor supplied them. **No thread parallelism:** the settle order is inherently sequential, the only parallelisable piece (CSR build) is a small share, and the metaphor's unit of work (one bird, one board) is far too small at these sizes.

## ASSUMPTION BROKEN

*"A priority structure must be consulted before every relaxation"* — and with it, *"each place's distance must be finalized before its neighbors are explored"* in its bookkeeping form: there is no `done[]` array at all. The lattice slot alone decides everything (`nd < dist[v]` to carve; `du > dist[v]` to let a louder note rot). Not broken, and stated as such: *the whole graph must be explored to know any single distance.*

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* one dying note: 16 bytes exactly, so four of them fill one cache line */
typedef struct { double d; int u; int pad; } kd_item;

/* ---------------- the roost: lazy 4-ary heap, no done[] array ------------- */
static void kd_push(kd_item *restrict H, int *restrict hsp, double d, int u)
{
    int i = (*hsp)++;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (H[p].d <= d) break;
        H[i] = H[p];
        i = p;
    }
    H[i].d = d; H[i].u = u;
}

static void kd_pop(kd_item *restrict H, int *restrict hsp)
{
    int hs = --(*hsp);
    if (hs <= 0) return;
    double d = H[hs].d; int u = H[hs].u;
    int i = 0;
    for (;;) {
        int c = (i << 2) + 1;
        if (c >= hs) break;
        int last = c + 4; if (last > hs) last = hs;
        int b = c; double bk = H[c].d;
        for (int k = c + 1; k < last; k++) { double kk = H[k].d; if (kk < bk) { bk = kk; b = k; } }
        if (bk >= d) break;
        H[i] = H[b];
        i = b;
    }
    H[i].d = d; H[i].u = u;
}

static void kd_roost(int n, const int *restrict off, const int *restrict edst,
                     const double *restrict ew, int source,
                     double *restrict dist, kd_item *restrict H)
{
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    int hs = 0;
    kd_push(H, &hs, 0.0, source);
    while (hs > 0) {
        double du = H[0].d;
        int u = H[0].u;
        kd_pop(H, &hs);
        if (du > dist[u]) continue;              /* a louder note, left to rot */
        int e = off[u], ee = off[u + 1];
        for (; e < ee; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; kd_push(H, &hs, nd, v); }
        }
    }
}

/* -------- the flat lattice: one fixed slot per place, read four at once ---- */
static void kd_lattice(int n, const int *restrict off, const int *restrict edst,
                       const double *restrict ew, int source,
                       double *restrict dist,
                       double *restrict actv, int *restrict actid, int *restrict posn)
{
    for (int i = 0; i < n; i++) { dist[i] = INFINITY; actv[i] = INFINITY; actid[i] = i; posn[i] = i; }
    for (int i = n; i < n + 16; i++) actv[i] = INFINITY;
    int cnt = n;
    dist[source] = 0.0;
    actv[posn[source]] = 0.0;

    while (cnt > 0) {
        /* pass 1: the quietest unsettled note */
        double best = INFINITY;
        int i = 0;
#if defined(__AVX2__)
        {
            __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0, m2 = m0, m3 = m0;
            for (; i + 16 <= cnt; i += 16) {
                m0 = _mm256_min_pd(m0, _mm256_loadu_pd(actv + i));
                m1 = _mm256_min_pd(m1, _mm256_loadu_pd(actv + i + 4));
                m2 = _mm256_min_pd(m2, _mm256_loadu_pd(actv + i + 8));
                m3 = _mm256_min_pd(m3, _mm256_loadu_pd(actv + i + 12));
            }
            m0 = _mm256_min_pd(_mm256_min_pd(m0, m1), _mm256_min_pd(m2, m3));
            double t[4];
            _mm256_storeu_pd(t, m0);
            best = t[0];
            if (t[1] < best) best = t[1];
            if (t[2] < best) best = t[2];
            if (t[3] < best) best = t[3];
        }
#endif
        for (; i < cnt; i++) if (actv[i] < best) best = actv[i];
        if (!(best < INFINITY)) break;           /* only desert left */

        /* pass 2: which slot holds it */
        int j = -1;
        i = 0;
#if defined(__AVX2__)
        {
            __m256d bv = _mm256_set1_pd(best);
            for (; i + 4 <= cnt; i += 4) {
                int msk = _mm256_movemask_pd(
                              _mm256_cmp_pd(_mm256_loadu_pd(actv + i), bv, _CMP_EQ_OQ));
                if (msk) { j = i + (int)__builtin_ctz((unsigned)msk); break; }
            }
        }
#endif
        if (j < 0) { for (; i < cnt; i++) if (actv[i] == best) { j = i; break; } }

        int u = actid[j];
        cnt--;                                   /* settled: never read again */
        actv[j] = actv[cnt]; actid[j] = actid[cnt]; posn[actid[j]] = j;
        posn[u] = -1;
        dist[u] = best;

        int e = off[u], ee = off[u + 1];
        for (; e < ee; e++) {
            int v = edst[e];
            double nd = best + ew[e];
            /* a settled v has dist[v] <= best <= nd, so this guard also
               guarantees a settled board is never touched twice */
            if (nd < dist[v]) { dist[v] = nd; actv[posn[v]] = nd; }
        }
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (m < 0) m = 0;
    size_t sn = (size_t)n, sm = (size_t)m;

    /* --- pacing the land: boards against roads, before a single bird flies --- */
    double dn = (double)n, dm = (double)m;
    double lg = log2(dn > 2.0 ? dn : 2.0);
    double avgdeg = (dn > 0.0) ? dm / dn : 0.0;
    double notes = dn * (1.0 + log(1.0 + avgdeg));      /* expected carvings */
    if (notes > dm) notes = dm;
    double roost_cost   = notes * (6.0 + 2.0 * lg) + 2.0 * dn;
    double lattice_cost = 0.18 * dn * dn + dm;
    int use_lattice = (lattice_cost <= roost_cost);

    /* --- one block for the roads (CSR) --- */
    size_t csr_bytes = sm * sizeof(double) + (sm + 2 * (sn + 1) + 2) * sizeof(int);
    char *csr = (char *)malloc(csr_bytes + 64);
    if (!csr) {
        for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
        dist_out[source] = 0.0;
        return;
    }
    double *ew = (double *)csr;
    int *ip = (int *)(ew + sm);
    int *off = ip; ip += sn + 1;
    int *cur = ip; ip += sn + 1;
    int *edst = ip;

    memset(off, 0, (sn + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(cur, off, (sn + 1) * sizeof(int));
    for (int i = 0; i < m; i++) {
        int p = cur[src[i]]++;
        edst[p] = dst[i]; ew[p] = weight[i];
    }

    if (use_lattice) {
        char *blk = (char *)malloc((sn + 16) * sizeof(double) + 2 * sn * sizeof(int) + 64);
        if (blk) {
            double *actv = (double *)blk;
            int *q = (int *)(actv + sn + 16);
            kd_lattice(n, off, edst, ew, source, dist_out, actv, q, q + sn);
            free(blk);
            free(csr);
            return;
        }
        use_lattice = 0;                          /* fall through to the roost */
    }

    {
        char *raw = (char *)malloc((sm + 12) * sizeof(kd_item) + 64);
        if (!raw) {
            for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
            dist_out[source] = 0.0;
            free(csr);
            return;
        }
        kd_item *base = (kd_item *)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
        kd_item *H = base + 3;   /* children of i are 4i+1..4i+4 -> one aligned line */
        kd_roost(n, off, edst, ew, source, dist_out, H);
        free(raw);
    }
    free(csr);
}
```

PREDICTION: speedup_vs_heap = 1.8

## PREDICTION

Written before any measurement. Reasoning: on large sparse graphs (the likely bench regime) the roost path pays ~half the heap cache misses per sift (4 line-aligned children per level instead of 2 levels of pairs), drops the `done[]` array and its random byte loads entirely (replaced by the free `du > dist[u]` rot-check), and cuts allocations from seven to two with one fewer pass over the edge list. Those are each 10–40 % effects on the heap-dominated portion; compounded I expect **1.5–2.2×**, centred at 1.8. On small graphs (`n` ≲ a few hundred) the lattice path should show 2–4× because the log factors and heap bookkeeping vanish. I do **not** expect a large win on dense graphs: there both paths are dominated by the `m` edge scans and my own cost model says so, which is why the regime test picks the roost there rather than the lattice.

## MEASUREMENT

**Not measured — reported as failure of measurement, not as success.** No tools were available in this session (`dijkstra_bench` / `dijkstra_contract` could not be called), so every number above is a model estimate, unvalidated. The pipeline will compile and measure; the honest statement is that the prediction of 1.8 stands untested.

What to check, in order, and what would falsify me:
1. **Exactness** vs the reference (all `n` values, `INFINITY` preserved for unreachable). Both paths are standard Dijkstra with non-negative weights, so distances should agree bit-for-bit up to tie-broken equal-cost paths.
2. **Roost-path speedup** on large sparse cases. Below ~1.3× my cache-miss accounting is wrong and the win is mostly the allocation/CSR trimming.
3. **Regime routing.** If the bench is all large-sparse, `use_lattice` is never taken and the metaphor's flat lattice contributes nothing measurable — I would say that plainly rather than claim the SIMD scan earned the number.
4. **Small-`n` cases.** If speedup there is ≈1.0, the constants `0.18` / `6.0 + 2·lg` in the land-pacing formula are miscalibrated, not the mechanism.

## VERDICT

The native's account maps onto array-scan Dijkstra plus lazy-deletion relaxation, with no invention required, and the two most valuable pieces are the ones that look least like the textbook: **no `done[]` array** ("let the louder rot" = stale-key skip, which is strictly cheaper than the reference's extra byte array and its random loads) and **the lattice slot as sole source of truth** (`nd < dist[v]` doing double duty as the relaxation test *and* the proof that a settled board is never touched twice — provable from non-negative weights, and the reason the lattice path needs no settled flag either).

Named risk and how it is addressed, not deferred: **the flat-lattice scan is catastrophically worse than a heap on large sparse graphs** (O(n²) vs O(m log n)). It is therefore guarded by the runtime land-pacing test computed from `n` and `m` *before any allocation*, with the roost as the fallback — and the lattice's scratch allocation failing also falls through to the roost. AVX2 is behind `#if defined(__AVX2__)` with a scalar path. No OpenMP, because the metaphor's unit of work (one bird, one board) is too small at these sizes and the settle order is sequential — I would rather ship no threads than a guarded thread path whose own guard I can't measure.

Where I expect this to be unimpressive and say so: on dense graphs both paths are dominated by the `m` relaxations, so the metaphor's headline device — hearing the whole lattice at once — buys almost nothing there; and the one assumption I would most like to have broken, *"the whole graph must be explored to know any single distance,"* is untouched, because the native himself ends by reading the traveler the whole lattice in order.