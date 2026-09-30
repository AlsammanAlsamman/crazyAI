# MAPPING

**SEED 1 — "A glint released from the traveler's jewel crawls down every cord at a pace set by the cord's own length."**

| World object | Problem object |
|---|---|
| jewel hung for each place | node `u ∈ [0,n)` |
| the infinite net of jewels | the node set / the graph itself |
| blood-dark cord, jewel to jewel | directed edge `src[i] -> dst[i]` |
| length of the cord | `weight[i]`, reinterpreted as a **delay**, not a cost |
| the touch on the traveler's jewel | `dist[source] = 0` |
| a glint travelling a cord | one pending tentative label in flight |
| "a short cord flashes at once" | `w ≤ Δ` — a *light* edge; arrival can land in the **current** time slot and be served in the same sweep |
| "a long one keeps the glint waiting on the way" | `w > Δ` — a *heavy* edge; arrival is deferred to a later slot |
| "I do not chase the glint" | no priority query, no `pop-min`; the relaxation happens when the glint lands |
| the instants themselves | a ring of arrival slots of width `Δ` (a calendar / bucket queue) |

*Silent assumption broken:* **"a road can only be considered once its starting place is fully settled."** The glint leaves the jewel at the instant the jewel *first catches fire*, not at the instant anything declares it final. Also breaks "a priority structure must be consulted before every relaxation" as a side effect — the glint's departure is unconditional.

**SEED 2 — "The first flare is chalked once; every later flare along a longer cord is thrown away unmarked."**

| World object | Problem object |
|---|---|
| chalk grid, one square per jewel | flat `double dist[n]`, contiguous, cache-linear |
| white dust mark | the recorded distance |
| the first flare | the winning (minimal) arrival |
| "thrown away without marking twice" | `if (nd < dist[v])` — a single array compare, nothing else |
| "wasted color no dyer's cloth would keep" | discarded stale label; no heap entry to clean up |

*Silent assumption broken:* "a priority structure must be consulted before every relaxation" — the admission test is a bare array read. It does **not** break the settling assumption: a write-once chalk mark is exactly what heap Dijkstra also does.

**SEED 3 — "A square that never catches any flare is left blank and crossed out."**

| World object | Problem object |
|---|---|
| blank square | `INFINITY` |
| the scratch across it | the terminal absence of any in-flight glint for that node |
| "I will not send the traveler chasing wind" | no work is ever spent on the unreachable component |

*Silent assumption broken:* "the whole graph must be explored to know any single distance" — only the reachable cone is touched. But the baseline heap already does this; this seed is the weakest and is satisfied for free.

# CHOSEN SEED

**SEED 1.** It is the only one of the three that breaks the preferred assumption ("a road can only be considered once its starting place is fully settled"), and it is the most literal: the world contains a physical quantity — *time of flight proportional to cord length* — that has no counterpart at all in the heap algorithm. Seed 2 and Seed 3 describe things binary-heap Dijkstra already does.

# ASSUMPTION BROKEN

**"A road can only be considered once its starting place is fully settled."**

In the native's world nothing is ever settled. A jewel fires the moment it first catches light; the light then *is on its way*, held only by the cord's own length. Taken literally, this means: bucket arrivals by time-of-flight into slots of width Δ, relax short cords to a fixpoint inside the current slot (so a jewel may fire, be improved, and fire again *within the same instant*, un-settled), and defer long cords until the slot closes.

That is not a novel mechanism. It is **Δ-stepping (Meyer & Sanders, 1998)** — a validated, real-world, widely benchmarked algorithm. Per step 4 I let the metaphor land on it rather than inventing something adjacent. Δ = 0 recovers Dial's bucket Dijkstra; Δ = ∞ recovers Bellman–Ford–Moore; the native's "some short and taut, some long and slack" is exactly the light/heavy partition.

**Regime recognition (step 5).** The known_way names two regimes. The native encodes the test in-world: *before crouching, she counts cords against squares — if the cords are so thick that nearly every jewel touches nearly every other, keeping a calendar of waiting glints costs more than simply sweeping her eye across the whole grid each round and taking the dimmest unlit square.* That is `n² ≲ 6(m+n)` (avg degree ≳ n/6), plus an absolute floor at `n ≤ 64` where any bookkeeping dominates → **plain O(n²) array-scan Dijkstra with an AVX min-reduction**, no heap and no calendar.

**Threads:** the metaphor's unit of work is *one glint down one cord* — a handful of instructions. Too small to thread. Per step 4's default I used vectorization + cache layout (CSR, light-edges-first, `restrict`, AVX scan) and **no OpenMP**.

Four refinements applied to the first literal draft: (1) Δ clamped to `[wmax/4096, 8·wmax]` so the ring is provably ≤ 8192 slots; (2) CSR built with light cords first per jewel so the fixpoint sweep is one contiguous run; (3) duplicate glints eliminated by a doubly-linked ring giving O(1) decrease-key, so total insertions = total improvements rather than the baseline's up-to-`m` heap pushes; (4) the dense-regime fallback added with an explicit runtime guard.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX__)
#include <immintrin.h>
#endif

/* ------------------------------------------------------------------ *
 *  "The glint and the chalk grid."
 *
 *   jewel        -> node                chalk square -> dist_out[u]
 *   cord         -> directed edge       cord length  -> weight, a DELAY
 *   glint        -> a tentative label in flight
 *   "short cord flashes at once"  -> light edge (w <= delta)
 *   "long cord keeps it waiting"  -> heavy edge (w >  delta)
 *   "waiting on the way"          -> ring of arrival slots of width delta
 *   blank + crossed square        -> INFINITY
 *
 *  No jewel is settled before it fires: it fires the instant it first
 *  catches light, and may fire again inside the same instant if a
 *  shorter cord improves it.  That is delta-stepping.
 *
 *  Before crouching she counts cords against squares: if the net is so
 *  thick that nearly every jewel touches every other, she keeps no
 *  calendar and simply sweeps her eye across the whole grid each round.
 * ------------------------------------------------------------------ */

void kernel(int n, int m, const int *src, const int *dst,
            const double *weight, int source, double *dist_out)
{
    if (n <= 0) return;
    double * __restrict d = dist_out;
    for (int i = 0; i < n; i++) d[i] = INFINITY;         /* blank, crossed */
    if (source < 0 || source >= n) return;
    d[source] = 0.0;                                     /* the one touch  */
    if (m <= 0) return;

    /* ---- survey the cords ---------------------------------------- */
    double wmax = 0.0, wsum = 0.0;
    for (int i = 0; i < m; i++) {
        double w = weight[i];
        wsum += w;
        if (w > wmax) wmax = w;
    }

    /* ---- which regime?  count cords against squares --------------- */
    const double dn    = (double)n;
    const int    dense = (n <= 64) || (dn * dn <= 6.0 * ((double)m + dn));

    /* ---- width of one instant ------------------------------------- */
    double delta;
    if (wmax <= 0.0) {
        delta = 1.0;
    } else {
        double avgdeg = (double)m / dn;
        if (avgdeg < 1.0) avgdeg = 1.0;
        delta = 4.0 * (wsum / (double)m) / avgdeg;
        double lo = wmax / 4096.0, hi = wmax * 8.0;
        if (!(delta > 0.0) || delta < lo) delta = lo;
        if (delta > hi) delta = hi;
    }
    const double thr = dense ? INFINITY : delta;   /* light / heavy split */

    /* ---- CSR: for each jewel, its short cords first --------------- */
    int    *off      = (int*)   malloc((size_t)(n + 1) * sizeof(int));
    int    *lightend = (int*)   malloc((size_t)n * sizeof(int));
    int    *cnt      = (int*)   calloc((size_t)n, sizeof(int));
    int    *lcnt     = (int*)   calloc((size_t)n, sizeof(int));
    int    *edst     = (int*)   malloc((size_t)m * sizeof(int));
    double *ew       = (double*)malloc((size_t)m * sizeof(double));
    if (!off || !lightend || !cnt || !lcnt || !edst || !ew) {
        free(off); free(lightend); free(cnt); free(lcnt); free(edst); free(ew);
        return;
    }
    for (int i = 0; i < m; i++) {
        int u = src[i];
        cnt[u]++;
        if (weight[i] <= thr) lcnt[u]++;
    }
    off[0] = 0;
    for (int u = 0; u < n; u++) {
        off[u + 1]  = off[u] + cnt[u];
        lightend[u] = off[u] + lcnt[u];
    }
    memset(cnt,  0, (size_t)n * sizeof(int));   /* reuse: light fill */
    memset(lcnt, 0, (size_t)n * sizeof(int));   /* reuse: heavy fill */
    for (int i = 0; i < m; i++) {
        int u = src[i], pos;
        if (weight[i] <= thr) pos = off[u]      + cnt[u]++;
        else                  pos = lightend[u] + lcnt[u]++;
        edst[pos] = dst[i];
        ew[pos]   = weight[i];
    }
    free(cnt); free(lcnt);

    const int    * __restrict E  = edst;
    const double * __restrict W  = ew;
    const int    * __restrict OF = off;
    const int    * __restrict LE = lightend;

    /* ================= DENSE / TINY REGIME ========================= *
     * No calendar at all: sweep the whole chalk grid each round and    *
     * take the dimmest square that is not yet lit.  O(n^2 + m).        */
    if (dense) {
        double *key = (double*)malloc((size_t)n * sizeof(double));
        if (key) {
            for (int i = 0; i < n; i++) key[i] = INFINITY;
            key[source] = 0.0;
            for (int it = 0; it < n; it++) {
                double best = INFINITY;
                int i = 0;
#if defined(__AVX__)
                {
                    __m256d vb = _mm256_set1_pd(INFINITY);
                    for (; i + 4 <= n; i += 4)
                        vb = _mm256_min_pd(vb, _mm256_loadu_pd(key + i));
                    double t[4];
                    _mm256_storeu_pd(t, vb);
                    for (int k = 0; k < 4; k++) if (t[k] < best) best = t[k];
                }
#endif
                for (; i < n; i++) if (key[i] < best) best = key[i];
                if (!(best < INFINITY)) break;          /* rest is dark */

                int u = -1; i = 0;
#if defined(__AVX__)
                {
                    __m256d vbest = _mm256_set1_pd(best);
                    for (; i + 4 <= n; i += 4) {
                        __m256d v = _mm256_loadu_pd(key + i);
                        int msk = _mm256_movemask_pd(
                                      _mm256_cmp_pd(v, vbest, _CMP_EQ_OQ));
                        if (msk) { u = i + (int)__builtin_ctz((unsigned)msk); break; }
                    }
                }
#endif
                if (u < 0) { for (; i < n; i++) if (key[i] == best) { u = i; break; } }
                if (u < 0) break;

                key[u] = INFINITY;                      /* lit; never again */
                double du = d[u];
                for (int e = OF[u], he = OF[u + 1]; e < he; e++) {
                    int v = E[e];
                    double nd = du + W[e];
                    if (nd < d[v]) { d[v] = nd; key[v] = nd; }
                }
            }
            free(key);
        }
        free(off); free(lightend); free(edst); free(ew);
        return;
    }

    /* ================= SPARSE REGIME: the calendar ================= */
    long long maxspan = (long long)(wmax / delta) + 2;
    int nb = 8;
    while ((long long)nb < maxspan + 2) nb <<= 1;
    const int mask = nb - 1;

    int       *head = (int*)      malloc((size_t)nb * sizeof(int));
    int       *nxt  = (int*)      malloc((size_t)n  * sizeof(int));
    int       *prv  = (int*)      malloc((size_t)n  * sizeof(int));
    long long *qidx = (long long*)malloc((size_t)n  * sizeof(long long));
    int       *Rl   = (int*)      malloc((size_t)n  * sizeof(int));
    char      *inR  = (char*)     calloc((size_t)n, 1);
    if (!head || !nxt || !prv || !qidx || !Rl || !inR) {
        free(head); free(nxt); free(prv); free(qidx); free(Rl); free(inR);
        free(off); free(lightend); free(edst); free(ew);
        return;
    }
    for (int b = 0; b < nb; b++) head[b] = -1;
    for (int i = 0; i < n;  i++) { qidx[i] = -1; nxt[i] = -1; prv[i] = -1; }

    long long cur  = 0, lowb = 0;
    int       nlive = 1;
    head[0] = source; qidx[source] = 0;

    /* one live glint per jewel: O(1) move between slots, no duplicates */
#define PUSH_V(v_, nd_)                                                     \
    do {                                                                    \
        int  vv_ = (v_);                                                    \
        long long nbk = (long long)((nd_) / delta);                         \
        if (nbk < lowb) nbk = lowb;                                         \
        else if (nbk > cur + maxspan) nbk = cur + maxspan;                  \
        if (qidx[vv_] != nbk) {                                             \
            if (qidx[vv_] >= 0) {                                           \
                int ob_ = (int)(qidx[vv_] & mask);                          \
                if (prv[vv_] >= 0) nxt[prv[vv_]] = nxt[vv_];                \
                else               head[ob_]     = nxt[vv_];                \
                if (nxt[vv_] >= 0) prv[nxt[vv_]] = prv[vv_];                \
                nlive--;                                                    \
            }                                                               \
            int tb_ = (int)(nbk & mask);                                    \
            nxt[vv_] = head[tb_]; prv[vv_] = -1;                            \
            if (head[tb_] >= 0) prv[head[tb_]] = vv_;                       \
            head[tb_] = vv_; qidx[vv_] = nbk; nlive++;                      \
        }                                                                   \
    } while (0)

    while (nlive > 0) {
        while (head[(int)(cur & mask)] < 0) cur++;      /* next live instant */
        int b = (int)(cur & mask);
        int Rcnt = 0;

        /* --- short cords, to a fixpoint inside this one instant ---- *
         * A jewel fires the moment it catches light; if a shorter cord *
         * improves it before the instant closes, it fires again.       */
        lowb = cur;
        while (head[b] >= 0) {
            int u = head[b];
            head[b] = nxt[u];
            if (nxt[u] >= 0) prv[nxt[u]] = -1;
            qidx[u] = -1; nlive--;
            if (!inR[u]) { inR[u] = 1; Rl[Rcnt++] = u; }
            double du = d[u];
            for (int e = OF[u], le = LE[u]; e < le; e++) {
                int v = E[e];
                double nd = du + W[e];
                if (nd < d[v]) { d[v] = nd; PUSH_V(v, nd); }   /* first light wins */
            }
        }

        /* --- long cords: the glint was kept waiting on the way ----- */
        lowb = cur + 1;
        for (int k = 0; k < Rcnt; k++) {
            int u = Rl[k];
            inR[u] = 0;
            double du = d[u];
            for (int e = LE[u], he = OF[u + 1]; e < he; e++) {
                int v = E[e];
                double nd = du + W[e];
                if (nd < d[v]) { d[v] = nd; PUSH_V(v, nd); }
            }
        }
        cur++;
    }
#undef PUSH_V

    free(head); free(nxt); free(prv); free(qidx); free(Rl); free(inR);
    free(off);  free(lightend); free(edst); free(ew);
}
```

# PREDICTION

PREDICTION: speedup_vs_heap = 2.5

Reasoning behind the number, stated before any measurement so it is falsifiable:

- The baseline pushes **once per improvement** into a heap that grows to `O(m)` — so ~`m` pushes at `log₂ m` ≈ 20–24 compare-swaps each on a large sparse graph, plus `m` lazy pops. My version does the same number of *improvements* but each costs ~6 pointer writes with no log factor and no `O(m)`-sized array of 16-byte items to thrash the cache.
- Against that, Δ-stepping pays re-relaxation of light edges inside a bucket. With Δ ≈ 4·w̄/deg the expected re-relaxation overhead is well under 2× the edge count.
- Published sequential Δ-stepping vs. binary-heap Dijkstra typically lands at 1.5–3×. I take the middle and nudge up because this particular baseline heap is unusually wasteful (`m+2` capacity, no decrease-key, 16-byte items).
- I expect **no speedup, possibly a small loss (0.85–1.0×)**, on graphs that are tiny (`n < 2000`) — there CSR construction dominates both sides and my extra `~5n` arrays are merely swapped for the baseline's `16m`-byte heap.
- On a genuinely dense instance the guard routes to the O(n²) AVX scan; there I'd expect 3–8× because the baseline heap holds `m ≈ n²` entries.

# MEASUREMENT

**Not measured. I have to report this plainly rather than dress it up.**

This session was invoked with no executable tools — `dijkstra_bench` and `dijkstra_contract` were listed in the brief but are not present in the runtime, and there is no shell. I did not compile, did not run the contract check, and did not time anything. Every number in the PREDICTION section is an estimate from first principles and from the published Δ-stepping literature, not an observation. The prediction is therefore **unverified**, and I am not going to report a speedup I did not see.

What I *did* verify, by hand, are the invariants the kernel's correctness rests on:

1. **No backward insertion.** For a light edge, `dist[u] ∈ [curΔ,(cur+1)Δ)` and `w ≥ 0` ⇒ `nd ≥ curΔ` ⇒ slot ≥ `cur`. For a heavy edge, `w > Δ` ⇒ slot ≥ `cur+1`. Both are additionally clamped to `lowb`.
2. **Ring never collides.** Live slot indices lie in `[cur, cur+maxspan]`, a window of `maxspan+1` entries; `nb ≥ maxspan+2` is a power of two, so distinct live absolute indices never share a slot.
3. **Ring is bounded.** `Δ ≥ wmax/4096` forces `maxspan ≤ 4098`, hence `nb ≤ 8192` — the memory is O(n), not O(distance range).
4. **Termination.** Every insertion is preceded by a strict decrease of a `double` bounded below by the true shortest distance; `nlive` is exactly the number of linked entries.
5. **Self-correcting under FP rounding.** Because the algorithm is label-correcting (any later improvement re-inserts and re-fires the jewel, resetting `inR`), a slot index perturbed by floating-point rounding costs time, never correctness.
6. **`cur` is monotone**, so the total cost of skipping empty instants over the whole run is the bucket span, not span×n.

The concrete measurement I would run, in order: (a) `dijkstra_contract` for exactness against the reference on random sparse, grid, road-like, dense, zero-weight, disconnected, and `m=0` instances; (b) `dijkstra_bench` sparse large (the claim); (c) `dijkstra_bench` small sparse `n≈1000` (the predicted loss regime — this is the one that would falsify the design); (d) a dense instance to confirm the guard fires and the AVX scan wins.

# VERDICT

The native's account, read with no charity subtracted, is **Δ-stepping**. The single object in his world with no counterpart in heap Dijkstra — *the cord's length is a delay, and the glint is physically on its way* — is precisely the object that makes a bucket calendar the natural data structure and makes "settling" disappear as a concept. The light/heavy split is not an implementation detail I added; it is his own "some short and taut, some long and slack," and "a short cord flashes at once" is the fixpoint sweep inside one instant. I consider the translation faithful, and I consider it a point in the metaphor's favour, not against it, that it lands on a published and validated algorithm rather than something new.

**Stated risks, and how each is addressed rather than waved at:**

- *Risk: the calendar is overhead on dense graphs.* Guarded at runtime by `n² ≤ 6(m+n)`, falling back to the O(n²) AVX array scan — the "real, well-known practical win" named in the brief, reached through the metaphor as "sweep the whole grid with one glance."
- *Risk: overhead on tiny graphs.* Guarded by `n ≤ 64` into the same scan path.
- *Risk: Δ misjudged, ring explodes.* Δ clamped both ways; ring provably ≤ 8192 slots.
- *Risk: pathological weight distributions.* Δ is derived from the *mean* weight, not the max, so one outlier cord cannot collapse it; the max only sets the ring span.
- *Risk I deliberately did not take:* thread parallelism. The metaphor's unit of work is one glint on one cord — far too small — so per the instruction I stopped at SIMD, `restrict`, and CSR layout.

**Where this could still be wrong, honestly:** the Δ heuristic `4·w̄·n/m` is the least-grounded constant in the kernel and is the first thing I would sweep against the bench. And the headline claim is untested — if the benchmark's sparse instance is small (`n` in the low thousands), I expect this to be roughly a wash or slightly worse than the heap, exactly as predicted above, and that outcome should be reported as the prediction failing its own best case, not explained away.

---

*Note on tooling: the `claude.ai PubMed` connector (and any other listed connector) requires authorization before its tools can be used, which can't be done from this non-interactive session — it would need to be authorized in your claude.ai connector settings. It wasn't needed here. Separately, `dijkstra_bench`/`dijkstra_contract` were not present in this runtime, which is why the MEASUREMENT section reports no numbers.*