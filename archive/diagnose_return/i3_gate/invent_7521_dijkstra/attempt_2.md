## MAPPING

**SEED 1 — "The nightingales always land first on the stone carrying the smallest owed letter."**

| World object | Problem object |
|---|---|
| garden stone at a house | slot `dist_out[v]` at a fixed address, never permuted |
| chalked letter | tentative distance, mutated in place |
| "unreadable, far number" | `INFINITY` — a number the birds cannot read, so they skip it |
| the flock circling the *unlocked* stones | a linear sweep over the readable-and-unlocked stones only |
| bird drops **first** onto the smallest | argmin over that sweep |
| bird drops *again* in the same circle ("no thief charges a negative toll and nothing cheaper can reach it now") | a second, threshold test licensing **several** finalizations per circle |
| landing = locking | finalize |
| no queue anywhere in the garden | **no priority structure at all** |

Breaks: *"a priority structure must be consulted before every relaxation"* — and, via multiple landings per circle, *"the next place to finalize is found by comparing against every remaining place"* (one comparison pass yields many finalizations, and it compares only against *readable* stones, not every remaining one).

**SEED 2 — "Threads cast from the tower price each road by what its thief is owed, one direction at a time."**

| World object | Problem object |
|---|---|
| tower at the current house | the node being expanded |
| thread that lands | a directed out-edge in CSR |
| thief's toll, directional | `weight[i]`, `src→dst` only |
| "rebuild the letter, stone stays where it stood" | `if (nd < dist_out[v]) dist_out[v] = nd;` — in-place, no sift, no relink |
| "knot of three threads… detached and called back on, recast at the lower number" | a 3-field record (place-name, owed sum, owed sum + cheapest onward toll) mutated in place — never a duplicate entry |

Breaks: *"a priority structure must be consulted before every relaxation"* (relaxation touches only the stone).

**SEED 3 — "A locked letter is never rebuilt again, and threads that reach nowhere are dropped."**

| World object | Problem object |
|---|---|
| locked letter | settled node, excluded from all later sweeps |
| thread into bramble/tide | edge to a node never improved; dropped |
| stone left unreadable forever | unreachable node stays `INFINITY` |

Breaks: *"the whole graph must be explored to know any single distance"*.

## CHOSEN SEED

**SEED 1.** Plainly: **none of the three seeds breaks "each place's distance must be finalized before its neighbors are explored"** — the native locks a stone *before* casting threads from it, in all three. So I fall back to the most literal seed, which is also the one furthest from the known way: SEED 1, the flock. SEED 2 is absorbed into it (the "knot of three threads" is literally the 3-field roster record, and relaxation is the bare in-place rebuild), and SEED 3 supplies the roster's entry/exit rule.

The core of the kernel *is* the flock: there is no heap, no bucket, no distance-keyed structure anywhere on the main path. Ordering is produced only by birds circling readable stones and comparing letters.

## ASSUMPTION BROKEN

**"A priority structure must be consulted before every relaxation."** Here relaxation is one compare and one store at a fixed address. All ordering work is deferred to the circle, and one circle pays for many locks — the native's own justification ("nothing cheaper can reach it now") is exactly the condition `key[v] ≤ min_u(key[u] + cheapest toll leaving u)`, which is **Crauser–Mehlhorn–Meyer–Sanders' OUT criterion** (1998), a validated technique. Per step 4 I let the metaphor arrive at it rather than invent something.

**Regime recognition, in-world:** if the garden is small, or thread-thick, one bird circles *every* stone in a single breath and no roster is worth keeping (→ the classic dense/small array-scan Dijkstra, preloaded roster). If the garden is vast, the birds circle only the readable stones (→ frontier sweep). **Safety valve:** if the flock keeps circling long circuits and dropping almost nobody (bramble: zero-ish tolls pin the threshold), the native abandons the garden and finishes the old slow way — a guarded heap fallback, so the worst case is ~heap speed, not catastrophe.

No thread parallelism: at benchmark frontier sizes a circle is shorter than the cost of summoning the flock. SIMD only.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the old slow way, kept only as the bramble fallback ---- */
typedef struct { double d; int u; } HItem;
static void hpush(HItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) >> 1; if (h[p].d <= h[i].d) break;
                    HItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HItem hpop(HItem *h, int *hs) {
    HItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0, sz = *hs;
    for (;;) { int l = 2*i+1, r = l+1, s = i;
        if (l < sz && h[l].d < h[s].d) s = l;
        if (r < sz && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HItem t = h[s]; h[s] = h[i]; h[i] = t; i = s; }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m < 0) m = 0;

    /* ---- the roads: one direction at a time; and each house's cheapest onward thief ---- */
    int    *off  = (int*)   malloc((size_t)(n + 1) * sizeof(int));
    int    *cnt  = (int*)   malloc((size_t)n * sizeof(int));
    double *mout = (double*)malloc((size_t)n * sizeof(double));
    size_t  ms   = (size_t)(m > 0 ? m : 1);
    int    *edst = (int*)   malloc(ms * sizeof(int));
    double *ew   = (double*)malloc(ms * sizeof(double));

    /* ---- the garden the birds circle: name, owed sum, owed sum + cheapest onward toll ---- */
    int    *rn   = (int*)   malloc((size_t)n * sizeof(int));
    double *rk   = (double*)malloc((size_t)n * sizeof(double));
    double *rl   = (double*)malloc((size_t)n * sizeof(double));
    int    *rpos = (int*)   malloc((size_t)n * sizeof(int));   /* >=0 in circuit, -1 never chalked, -2 locked */
    int    *stl  = (int*)   malloc((size_t)n * sizeof(int));

    if (!off || !cnt || !mout || !edst || !ew || !rn || !rk || !rl || !rpos || !stl) {
        free(off); free(cnt); free(mout); free(edst); free(ew);
        free(rn); free(rk); free(rl); free(rpos); free(stl);
        return;
    }

    for (int i = 0; i < n; i++) { cnt[i] = 0; mout[i] = INFINITY; }
    for (int e = 0; e < m; e++) { int u = src[e]; cnt[u]++; if (weight[e] < mout[u]) mout[u] = weight[e]; }
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + cnt[i];
    for (int i = 0; i < n; i++) cnt[i] = off[i];
    for (int e = 0; e < m; e++) { int u = src[e]; int p = cnt[u]++; edst[p] = dst[e]; ew[p] = weight[e]; }

    /* ---- which garden is this? small or thread-thick: one bird circles it all in a breath ---- */
    double dn = (double)n, dm = (double)m;
    int preload = (n <= 2048) || (dm >= 0.05 * dn * dn);

    int F;
    if (preload) {
        for (int i = 0; i < n; i++) { rn[i] = i; rk[i] = INFINITY; rl[i] = INFINITY; rpos[i] = i; }
        F = n;
        rk[source] = 0.0; rl[source] = mout[source];
    } else {
        for (int i = 0; i < n; i++) rpos[i] = -1;
        rn[0] = source; rk[0] = 0.0; rl[0] = mout[source]; rpos[source] = 0; F = 1;
    }
    dist_out[source] = 0.0;

    /* bramble budget: if the circles stop paying for themselves, leave the garden */
    double used = 0.0;
    double budget = 16.0 * (dn + dm + 16.0) * (log2(dn + 2.0));
    int bailed = 0;

    while (F > 0) {
        if (used > budget) { bailed = 1; break; }
        used += 3.0 * (double)F;

        /* ---- one circle: smallest owed letter, and the cheapest way anything could still be reached ---- */
        double mv = INFINITY, LB = INFINITY;
        int i = 0;
#if defined(__AVX2__)
        {
            __m256d a0 = _mm256_set1_pd(INFINITY), a1 = a0, b0 = a0, b1 = a0;
            for (; i + 8 <= F; i += 8) {
                a0 = _mm256_min_pd(a0, _mm256_loadu_pd(rk + i));
                a1 = _mm256_min_pd(a1, _mm256_loadu_pd(rk + i + 4));
                b0 = _mm256_min_pd(b0, _mm256_loadu_pd(rl + i));
                b1 = _mm256_min_pd(b1, _mm256_loadu_pd(rl + i + 4));
            }
            a0 = _mm256_min_pd(a0, a1); b0 = _mm256_min_pd(b0, b1);
            double ta[4], tb[4];
            _mm256_storeu_pd(ta, a0); _mm256_storeu_pd(tb, b0);
            for (int j = 0; j < 4; j++) { if (ta[j] < mv) mv = ta[j]; if (tb[j] < LB) LB = tb[j]; }
        }
#endif
        for (; i < F; i++) { if (rk[i] < mv) mv = rk[i]; if (rl[i] < LB) LB = rl[i]; }

        if (!(mv < INFINITY)) break;            /* only unreadable debts left */
        double thr = (LB > mv) ? LB : mv;       /* always >= mv: at least one bird lands */
        if (!(thr < INFINITY)) thr = DBL_MAX;   /* never lock an unreadable stone */

        /* ---- the landings: lock every stone nothing cheaper can reach; compact the circuit ---- */
        int ns = 0, w = 0;
        i = 0;
#if defined(__AVX2__)
        {
            __m256d vt = _mm256_set1_pd(thr);
            for (; i + 4 <= F; i += 4) {
                __m256d k = _mm256_loadu_pd(rk + i);
                int msk = _mm256_movemask_pd(_mm256_cmp_pd(k, vt, _CMP_LE_OQ));
                if (msk == 0) {
                    if (w != i) {
                        _mm256_storeu_pd(rk + w, k);
                        _mm256_storeu_pd(rl + w, _mm256_loadu_pd(rl + i));
                        _mm_storeu_si128((__m128i*)(rn + w), _mm_loadu_si128((const __m128i*)(rn + i)));
                        rpos[rn[w]] = w; rpos[rn[w+1]] = w+1;
                        rpos[rn[w+2]] = w+2; rpos[rn[w+3]] = w+3;
                    }
                    w += 4;
                } else {
                    for (int j = 0; j < 4; j++) {
                        int idx = i + j;
                        if (msk & (1 << j)) {
                            int u = rn[idx];
                            dist_out[u] = rk[idx]; rpos[u] = -2; stl[ns++] = u;
                        } else {
                            rn[w] = rn[idx]; rk[w] = rk[idx]; rl[w] = rl[idx];
                            rpos[rn[w]] = w; w++;
                        }
                    }
                }
            }
        }
#endif
        for (; i < F; i++) {
            if (rk[i] <= thr) {
                int u = rn[i];
                dist_out[u] = rk[i]; rpos[u] = -2; stl[ns++] = u;
            } else {
                rn[w] = rn[i]; rk[w] = rk[i]; rl[w] = rl[i];
                rpos[rn[w]] = w; w++;
            }
        }
        F = w;

        /* ---- cast threads from each newly locked house; rebuild letters in place ---- */
        for (int s = 0; s < ns; s++) {
            int u = stl[s];
            double base = dist_out[u];
            int e1 = off[u + 1];
            for (int e = off[u]; e < e1; e++) {
                int v = edst[e];
                double nd = base + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    int p = rpos[v];
                    if (p >= 0) {                       /* knot detached and recast, same stone */
                        rk[p] = nd; rl[p] = nd + mout[v];
                    } else if (p == -1) {               /* first readable chalk: joins the circuit */
                        p = F++; rpos[v] = p;
                        rn[p] = v; rk[p] = nd; rl[p] = nd + mout[v];
                    }
                }
            }
        }
    }

    /* ---- bramble fallback: finish the old slow way from the exact same state ---- */
    if (bailed) {
        HItem *h = (HItem*)malloc((size_t)((size_t)F + ms + 2) * sizeof(HItem));
        if (h) {
            int hs = 0;
            for (int i = 0; i < F; i++) if (rk[i] < INFINITY) hpush(h, &hs, rk[i], rn[i]);
            while (hs > 0) {
                HItem t = hpop(h, &hs);
                int u = t.u;
                if (rpos[u] == -2) continue;
                if (t.d > dist_out[u]) continue;
                rpos[u] = -2;
                double base = dist_out[u];
                int e1 = off[u + 1];
                for (int e = off[u]; e < e1; e++) {
                    int v = edst[e];
                    double nd = base + ew[e];
                    if (nd < dist_out[v]) {
                        dist_out[v] = nd;
                        if (rpos[v] != -2) hpush(h, &hs, nd, v);
                    }
                }
            }
            free(h);
        }
    }

    free(off); free(cnt); free(mout); free(edst); free(ew);
    free(rn); free(rk); free(rl); free(rpos); free(stl);
}
```

## PREDICTION

PREDICTION: speedup_vs_heap = 2.5

Reasoning stated before any measurement: the main path does zero heap operations; per settled node it pays ~3 contiguous double-compares per *readable unlocked* stone amortized over a whole batch of landings, at ~0.2 cycles/element under AVX2, versus ~17 cache-missing sift levels per pop. I expect 3–6× on small/dense graphs (n ≤ 2048 or avg degree ≳ n/20), 2–3× on large sparse graphs (batch size ≈ √(key-density) per circle ⇒ rounds ≪ n), and ≈0.7–1× on bramble graphs with many near-zero tolls, where the budget trips and the heap finishes the job.

## MEASUREMENT

**Not measured.** `dijkstra_bench` and `dijkstra_contract` were unavailable in this session — no tools were exposed — so I state this plainly rather than reporting a number I did not obtain. The prediction above is unverified. What the harness should check, in order:

1. Exactness against the reference on every case (the criterion is exact, not heuristic; the only expected deviations are ≤1-ulp on ties, which tolerance absorbs).
2. Whether `bailed` ever triggers (add a counter if instrumenting) — if it fires on ordinary sparse cases, the budget constant `16.0` is too tight, not the mechanism.
3. Speedup split by regime: `n ≤ 2048` / dense / large sparse. If large-sparse speedup < 1, the round count is the culprit; measure rounds, not cycles, first.

## VERDICT

The core is the native's flock, translated literally: the garden array is never permuted, relaxation is one compare and one in-place rebuild with no structure consulted, ordering exists only as birds comparing letters on a circle, locked letters leave the circuit forever, and stones never chalked keep their unreadable debt. The two details I had previously discarded as decoration turned out to be load-bearing: "drop **first**" (so a circle yields many landings, which is exactly Crauser's OUT criterion — a validated technique the metaphor *derives* rather than a novelty I invented) and "unreadable" (so the birds circle the frontier, not the garden, which is what makes a scan-based method viable at all on sparse graphs).

Honest risks, each addressed rather than named and shipped:

- *Frontier scans degenerate to O(n²) when tolls are near-zero and the threshold collapses to the minimum.* Guarded by the sweep budget with a heap fallback that resumes from the identical state; worst case ≈ heap + one wasted pass, not a blow-up.
- *Roster bookkeeping is pure overhead on dense/small graphs.* Guarded by the `preload` regime test, which makes the circuit the whole garden and removes every append and branch — landing on the known O(n²) array-scan win for that regime.
- *Thread parallelism.* Deliberately omitted: the flock's circle at these sizes (tens of thousands of doubles, ~25 µs) is not large enough to amortize summoning threads once per round, and a per-round OpenMP barrier over hundreds of rounds would lose outright. SIMD only, as instructed.

Where it could still lose: a graph with a large frontier *and* a threshold pinned near the minimum every round, big enough that the wasted pre-bail passes cost real time — there it should land near 0.7×. If measurement shows that case dominating the benchmark, the right next move is a two-level "districts of the garden" flock (cached per-district smallest songs) rather than abandoning the mechanism.