## MAPPING

**SEED 1 — "The nightingales always land first on the stone carrying the smallest owed letter."**

| World object | Problem object |
|---|---|
| garden | the node set |
| stone | one node slot |
| chalked letter | tentative distance `key[i]` |
| "nothing owed" stone under the tower | `key[source] = 0` |
| unreadable far number | `INFINITY` |
| unlocked stones | the *compacted* candidate array (shrinks every round) |
| nightingales circling | one linear argmin pass over that array |
| bird drops on the smallest | extract-min **with no priority structure at all** |
| landing = letter set, locked | `dist_out[u] = key`, node removed from the circuit |
| "no thief charges a negative toll" | `w ≥ 0` ⇒ finality of the popped minimum |

*Breaks:* **"a priority structure must be consulted before every relaxation."** In this reading a relaxation is a bare compare-and-store into a flat array — nothing is pushed, nothing sifts, no structure is touched. The ordering cost is paid once per *settle*, never per *edge*.

**SEED 2 — "Threads cast from the tower price each road by what its thief is owed, one direction at a time."**

| World object | Problem object |
|---|---|
| thread cast from tower | one relaxation attempt |
| thief guarding the road | the edge weight |
| "directional / paid one way is not the other" | directed CSR; `src[i]->dst[i]` only |
| "each house the roads actually touch" | out-adjacency row `off[u]..off[u+1]` |
| rechalk, stone stays where it stood | in-place `key[v] = nd`, no reinsertion |

*Breaks:* **"a road can only be considered once its starting place is fully settled"** — weakly. It really only fixes the CSR/directionality; the tower still casts only from locked houses. Mostly a data-layout seed.

**SEED 3 — "A locked garden letter is never rebuilt again, and threads that reach nowhere are dropped and thrown away."**

| World object | Problem object |
|---|---|
| locked letter | `pos[u] = -1`, excluded from all future scans |
| never rebuilt | no stale queue entries ever exist (unlike the lazy reference heap) |
| bramble / tide / loop to nowhere | edges into settled nodes, self-loops, dead ends |
| threads let drop | relaxation that fails the `pos ≥ 0` test — zero further work |
| stones chalked forever with unreadable debt | unreachable nodes keep `INFINITY` |
| traveler never needed them | **early exit** the instant the circuit's minimum is `INFINITY` |

*Breaks:* **"the whole graph must be explored to know any single distance."**

## CHOSEN SEED

**SEED 1.** It is the most literal (a bird sweeping a garden *is* a linear argmin) and the most structurally distant from binary-heap Dijkstra — it deletes the heap entirely.

**Plainly: none of the three seeds breaks "each place's distance must be finalized before its neighbors are explored."** The native is explicit — *"From that newly locked house I cast threads again."* He locks first, then casts. I am not going to pretend otherwise; per the instruction I fall back to the most literal seed. SEED 2 and SEED 3 are folded in as they stand (CSR/directional pricing; locked-letter exclusion and dropped threads giving the early exit).

## ASSUMPTION BROKEN

**"A priority structure must be consulted before every relaxation."** Broken outright on the sweep path: `m` relaxations cost `m` compare-and-stores and touch no ordering structure whatsoever. Secondarily, **"the whole graph must be explored"** is broken by the bramble rule (exit when the best unlocked letter is `INFINITY`).

Two regimes are named in the problem statement (dense/small vs. sparse), so the native must recognise which garden he is standing in. He does it the way he'd actually do it: **he counts roads per house before climbing.** A *thick* garden (or one small enough for the birds to see whole) gets the nightingale circuit. A *thin, wide* garden gets the other device he already described — *"a knot of three threads … detached … and called back on"* — which is literally a **3-ary heap with in-place decrease-key** (a knot holds three threads; a thread is detached and recast at a lower number rather than a duplicate being flung). Note this is still not the textbook path: the reference is a *lazy binary* heap with up to `m` entries and stale pops; the knot holds at most `n` and, per SEED 3, never leaves a dead letter behind.

No thread parallelism. One bird circuit is `≤ n/2` doubles — a few thousand cycles at benchmark sizes, far below an OpenMP barrier, and the circuits are strictly sequential. The metaphor's unit of work is too small; SIMD only.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ==========================================================================
   THE GARDEN OF NIGHTINGALE
     stone            -> node slot            letter  -> tentative distance
     thief            -> directed edge weight thread  -> one relaxation
     nightingale circuit -> vectorised argmin over the UNLOCKED stones only
     locking          -> final distance; the stone leaves the birds' circuit
     bramble / tide   -> unreachable: the circuit finds only INFINITY -> stop
     knot of three threads -> 3-ary heap, decrease-key in place (thin garden)
   ========================================================================== */

#define GARDEN_SMALL 1024      /* a garden the birds can see whole          */
#define GARDEN_THICK 125.0     /* roads thick enough to be worth circling   */

/* ---- one circuit of the birds: index of the smallest unlocked letter ---- */
static inline int nightingale_circuit(const double *restrict key, int cnt,
                                      double *best_out)
{
#if defined(__AVX2__)
    double bv = INFINITY; int bi = -1; int i = 0;
    if (cnt >= 16) {
        __m256d b0 = _mm256_set1_pd(INFINITY), b1 = b0, b2 = b0, b3 = b0;
        __m256d n0 = _mm256_set1_pd(-1.0), n1 = n0, n2 = n0, n3 = n0;
        __m256d c0 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        __m256d c1 = _mm256_set_pd(7.0, 6.0, 5.0, 4.0);
        __m256d c2 = _mm256_set_pd(11.0, 10.0, 9.0, 8.0);
        __m256d c3 = _mm256_set_pd(15.0, 14.0, 13.0, 12.0);
        const __m256d st = _mm256_set1_pd(16.0);
        const int lim = cnt - 16;
        for (; i <= lim; i += 16) {
            __m256d v0 = _mm256_loadu_pd(key + i);
            __m256d v1 = _mm256_loadu_pd(key + i + 4);
            __m256d v2 = _mm256_loadu_pd(key + i + 8);
            __m256d v3 = _mm256_loadu_pd(key + i + 12);
            __m256d m0 = _mm256_cmp_pd(v0, b0, _CMP_LT_OQ);
            __m256d m1 = _mm256_cmp_pd(v1, b1, _CMP_LT_OQ);
            __m256d m2 = _mm256_cmp_pd(v2, b2, _CMP_LT_OQ);
            __m256d m3 = _mm256_cmp_pd(v3, b3, _CMP_LT_OQ);
            b0 = _mm256_min_pd(b0, v0); n0 = _mm256_blendv_pd(n0, c0, m0);
            b1 = _mm256_min_pd(b1, v1); n1 = _mm256_blendv_pd(n1, c1, m1);
            b2 = _mm256_min_pd(b2, v2); n2 = _mm256_blendv_pd(n2, c2, m2);
            b3 = _mm256_min_pd(b3, v3); n3 = _mm256_blendv_pd(n3, c3, m3);
            c0 = _mm256_add_pd(c0, st); c1 = _mm256_add_pd(c1, st);
            c2 = _mm256_add_pd(c2, st); c3 = _mm256_add_pd(c3, st);
        }
        double vb[16], ib[16];
        _mm256_storeu_pd(vb,      b0); _mm256_storeu_pd(vb + 4,  b1);
        _mm256_storeu_pd(vb + 8,  b2); _mm256_storeu_pd(vb + 12, b3);
        _mm256_storeu_pd(ib,      n0); _mm256_storeu_pd(ib + 4,  n1);
        _mm256_storeu_pd(ib + 8,  n2); _mm256_storeu_pd(ib + 12, n3);
        for (int j = 0; j < 16; ++j)
            if (vb[j] < bv) { bv = vb[j]; bi = (int)ib[j]; }
    }
    for (; i < cnt; ++i) { double v = key[i]; if (v < bv) { bv = v; bi = i; } }
    *best_out = bv;
    return bi;
#else
    double bv = INFINITY;
    for (int j = 0; j < cnt; ++j) { double v = key[j]; if (v < bv) bv = v; }
    *best_out = bv;
    if (!(bv < INFINITY)) return -1;
    for (int j = 0; j < cnt; ++j) if (key[j] == bv) return j;
    return -1;
#endif
}

/* ---------------- thick / small garden: send the nightingales ------------ */
static void garden_of_nightingale(int n,
                                  const int    *restrict off,
                                  const int    *restrict edst,
                                  const double *restrict ew,
                                  int source,
                                  double *restrict dist_out,
                                  int    *restrict cand,   /* slot -> stone  */
                                  int    *restrict cpos,   /* stone -> slot  */
                                  double *restrict ckey)   /* slot -> letter */
{
    for (int i = 0; i < n; ++i) { cand[i] = i; cpos[i] = i; ckey[i] = INFINITY; }
    ckey[source] = 0.0;

    int cnt = n;
    while (cnt > 0) {
        double best;
        int p = nightingale_circuit(ckey, cnt, &best);
        if (p < 0) break;                    /* only bramble left: drop it   */

        int u = cand[p];
        dist_out[u] = best;                  /* the letter is locked         */

        --cnt;                               /* stone leaves the circuit     */
        int moved = cand[cnt];
        cand[p] = moved; ckey[p] = ckey[cnt]; cpos[moved] = p;
        cpos[u] = -1;

        const int e1 = off[u + 1];
        for (int e = off[u]; e < e1; ++e) {  /* cast threads, price by thief */
            int v = edst[e];
            int q = cpos[v];
            if (q >= 0) {                    /* locked/nowhere -> thread dropped */
                double nd = best + ew[e];
                if (nd < ckey[q]) ckey[q] = nd;   /* rechalk; stone unmoved  */
            }
        }
    }
}

/* ------------- thin, wide garden: knots of three threads ----------------- */
static void knots_of_three(int n,
                           const int    *restrict off,
                           const int    *restrict edst,
                           const double *restrict ew,
                           int source,
                           double *restrict dist_out,
                           int    *restrict heap,
                           int    *restrict pos,
                           double *restrict key)
{
    for (int i = 0; i < n; ++i) { pos[i] = -1; key[i] = INFINITY; }
    key[source] = 0.0; heap[0] = source; pos[source] = 0;
    int hs = 1;

    while (hs > 0) {
        int u = heap[0];
        double du = key[u];
        dist_out[u] = du;
        pos[u] = -2;                          /* locked, never rebuilt again */

        --hs;
        if (hs > 0) {                         /* settle the knot again       */
            int w = heap[hs]; double kw = key[w]; int i = 0;
            for (;;) {
                int c = 3 * i + 1;
                if (c >= hs) break;
                int bidx = c; double bk = key[heap[c]];
                int cend = c + 3; if (cend > hs) cend = hs;
                for (int j = c + 1; j < cend; ++j) {
                    double t = key[heap[j]];
                    if (t < bk) { bk = t; bidx = j; }
                }
                if (bk >= kw) break;
                heap[i] = heap[bidx]; pos[heap[i]] = i; i = bidx;
            }
            heap[i] = w; pos[w] = i;
        }

        const int e1 = off[u + 1];
        for (int e = off[u]; e < e1; ++e) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < key[v]) {
                key[v] = nd;
                int pv = pos[v];
                if (pv >= 0) {                /* detach and call back on     */
                    int i = pv;
                    while (i > 0) {
                        int par = (i - 1) / 3;
                        if (key[heap[par]] <= nd) break;
                        heap[i] = heap[par]; pos[heap[i]] = i; i = par;
                    }
                    heap[i] = v; pos[v] = i;
                } else if (pv == -1) {        /* first thread to reach it    */
                    int i = hs++;
                    while (i > 0) {
                        int par = (i - 1) / 3;
                        if (key[heap[par]] <= nd) break;
                        heap[i] = heap[par]; pos[heap[i]] = i; i = par;
                    }
                    heap[i] = v; pos[v] = i;
                }
            }
        }
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m <= 0) { dist_out[source] = 0.0; return; }

    /* ---- the roads, priced one direction at a time (CSR, single pass) ---- */
    int *off = (int *)calloc((size_t)n + 2, sizeof(int));
    int *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !edst || !ew) { free(off); free(edst); free(ew); return; }
    for (int i = 0; i < m; ++i) off[src[i] + 2]++;
    for (int i = 2; i <= n + 1; ++i) off[i] += off[i - 1];
    for (int i = 0; i < m; ++i) {
        int p = off[src[i] + 1]++;
        edst[p] = dst[i]; ew[p] = weight[i];
    }

    int    *ia = (int *)malloc((size_t)n * sizeof(int));
    int    *ib = (int *)malloc((size_t)n * sizeof(int));
    double *da = (double *)malloc((size_t)n * sizeof(double));
    if (!ia || !ib || !da) { free(off); free(edst); free(ew);
                             free(ia); free(ib); free(da); return; }

    /* ---- count the roads per house before climbing: which garden is this? */
    double nn = (double)n * (double)n;
    int thick_or_small = (n <= GARDEN_SMALL) || ((double)m * GARDEN_THICK >= nn);

    if (thick_or_small)
        garden_of_nightingale(n, off, edst, ew, source, dist_out, ia, ib, da);
    else
        knots_of_three(n, off, edst, ew, source, dist_out, ia, ib, da);

    free(off); free(edst); free(ew); free(ia); free(ib); free(da);
}
```

## PREDICTION

**PREDICTION: speedup_vs_heap = 2.5**

Reasoning behind the number, stated before any measurement: on the thick/small path the sweep costs `≈0.15·n²` cycles of pure streaming L2 work plus `m` branch-and-store relaxations, versus the reference's `≈m` heap pushes at ~25–60 cycles each with random `dist` traffic — a 3–8× win where it applies. On the thin path the knot-of-three only beats the reference lazy binary heap by removing stale entries (`≤n` resident instead of `≤m`) and shortening the tree (`log₃` depth, ~1.9·log₂n comparisons instead of 2·log₂n) — realistically 1.3–1.8×, and both paths pay the same unavoidable CSR build, which floors the achievable ratio. 2.5 is the blend I expect across a mixed benchmark.

The four refinements applied, all driven by the native's own words rather than by a profiler:
1. Circuit over **unlocked stones only** (compacted, swap-removed) — `n²/2` instead of `n²`; "they circle the unlocked stones."
2. **AVX2 argmin, four independent accumulators** — the bird sees sixteen stones per pass, breaking the `min_pd` latency chain.
3. **`cpos[v] = -1` as the locked marker** — fuses the settled test into the slot lookup, so no separate `done[]` byte is read per edge; "a locked letter is never rebuilt."
4. **Bramble exit** — stop the moment a circuit returns only `INFINITY`.

## MEASUREMENT

**Not performed. No tools were available in this session** — `dijkstra_bench` and `dijkstra_contract` could not be called, so I have no measured speedup to report, only the prediction above. I will not invent a number. What I can assert from the code itself:

- **Exactness:** every relaxation computes `dist_out[u] + w` with `dist_out[u]` the identical settled value the reference produces, so results match bit-for-bit, not merely within tolerance. Unreachable nodes are never locked and retain the `INFINITY` written at entry.
- **Tie handling** differs from the reference's heap order but cannot change any final value: a predecessor settled after `v` has `d(u) ≥ d(v)` and `w ≥ 0`.
- `hs ≤ n` always (a node is pushed at most once; `pos = -2` after locking and `nd < key[v]` is then impossible), so the heap allocation is safe.

## VERDICT

The literal translation is a sweep, and a sweep has exactly one stated risk: **it is `O(n²)` and loses badly on a large sparse graph.** Per step 4 I have not shipped that risk unguarded — the native counts roads per house before climbing the tower (`n ≤ 1024` or `m·125 ≥ n²`), and the thin-garden case falls through to his other device, the knot of three threads. Both paths come out of the same description; neither is the textbook binary heap.

Honest exposure that remains: the crossover constant `125` (density ≥ 0.8% of complete) is derived from a cost model, not measured on this machine. Near the crossover a wrong call costs roughly 2×; far from it the guard puts us on the right side. If `dijkstra_bench` shows a regression at some middling density, that constant is the single knob to move, and the failure would be mine — a mis-tuned threshold — not the metaphor's.

Thread parallelism was considered and rejected on the metaphor's own terms: one bird circuit is too small a unit of work to pay an OpenMP barrier, and circuits are strictly sequential. Vectorisation only, as instructed.