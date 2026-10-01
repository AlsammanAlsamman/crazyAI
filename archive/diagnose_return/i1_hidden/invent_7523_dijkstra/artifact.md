## MAPPING

**SEED 1** — *"A runner walks each road carrying its notch and lays a new stick on the far heap equal to the home ember's length plus that notch."*

| World object | Problem object |
|---|---|
| place / heap | vertex `v ∈ [0,n)` |
| road | directed edge `src[i] -> dst[i]` |
| notch on the runner's stick | `weight[i]` |
| ember at a heap | `dist_out[v]` (current tentative distance) |
| lit ember | `dist_out[v] < INFINITY` |
| runner sent down a road | one edge relaxation from the tail |
| stick laid on the far heap | candidate value `dist_out[u] + w` offered to `dst` |
| "all roads at once" | `#pragma omp for` over the out-edges of the active set |
| stick dropped by the wayside | candidate discarded (tail has no ember / head sealed) |
| feeding the dropped stick to the hearth | no-op; discarded work keeps the round going |

**SEED 2** — *"Every heap throws its sticks together in one land-wide throw, and the row read straight across picks each heap's shortest sliver as its new ember while the rest are burned."*

| World object | Problem object |
|---|---|
| one land-wide throw | one synchronous round (a single OpenMP parallel phase + barrier) |
| a heap's pile of sticks | the multiset of candidates offered to `v` this round |
| reading straight across a single row, one sliver per heap, same breath | the *simultaneous* per-vertex `min` — every vertex reduced in the same phase, no ordering between vertices |
| shortest sliver becomes the new ember | `dist_out[v] = min(dist_out[v], candidates)` — implemented as a lock-free atomic min |
| burning the longer slivers for fuel | losing candidates are simply dropped; only the winner is stored |
| coal banked forward to the next throw | `dist_out` persists across rounds (monotone non-increasing) |

**SEED 3** — *"A garrison seals a heap once its ember stops changing between throws, fixing its distance for good and halting its runners."*

| World object | Problem object |
|---|---|
| garrison on a heap | `inq[v] == 0` — vertex not in the active frontier |
| ember unchanged across a throw | no relaxation lowered `dist_out[v]` this round |
| sealed heap's runners stop going out | vertex is **not** scanned next round: its out-edges are skipped |
| garrison "refuses entry" to incoming sticks | incoming candidate rejected because it isn't shorter |
| cold ash that never catches a spark | `dist_out[v]` stays `INFINITY` to quiescence → unreachable |
| every heap wears its garrison | frontier empty → global termination |
| hooded figure recites the embers in one pass | `dist_out` is already the answer buffer; no extraction step |
| "elephants never walked" dead field | unreachable component |

## CHOSEN SEED

**SEED 3 (the garrison).** It is the most literal *and* it is the one that breaks the forbidden assumption. Seeds 1 and 2 describe how work is done; Seed 3 describes how a vertex is *finalized*, and it finalizes by a purely **local** test — "did *my own* ember change?" — with no reference to any other heap. There is no priority queue, no global minimum, no ordering of vertices against one another anywhere in this construction.

## ASSUMPTION BROKEN

> *"The next place to finalize is found by comparing against every remaining place."*

Dijkstra finalizes by proving a vertex is the global argmin over the unsettled set — a comparison against every remaining place, which is exactly what the heap exists to amortize. The native never compares two places. A heap is done when *it personally* stopped moving. Settling is therefore not a sequential, globally-ordered event; it is a per-vertex fixed-point test that every vertex can evaluate independently and simultaneously. That is what makes the whole land throw at once.

**Where I had to repair the native, stated plainly.** Taken at the letter, "sealed for good" is unsound. Counterexample: `s→v` (10), `s→a` (1), `a→b` (1), `b→v` (1). After throw 1: `d[v]=10`, `d[a]=1`, `d[b]=∞`. After throw 2: `d[v]=10` — *unchanged* — so `v` is sealed at 10, and throw 3's stick of length 3 from `b` is refused entry. True answer is 3. An ember can sit still for a throw and still shorten later.

The minimal repair that keeps the mechanism intact: **the garrison is revocable.** A heap is sealed the instant its ember stops changing (and its runners stop, exactly as the native says), but if a shorter stick does later arrive, the garrison breaks and its runners go out again. Everything else survives literally, and crucially the native's *reason* for halting the runners survives and is genuinely sound: if `d[u]` did not change last throw, then `u`'s out-edges this throw would lay the identical sticks its neighbours already absorbed — skipping them is lossless, not approximate. The seal is a correct **work-elimination** rule; it is only the *permanence* that was wrong. I did not replace it with a heap; there is no priority queue in the artifact.

## ARTIFACT

```c
/* Shortest paths by the native's method:
 *   embers  = dist_out[]            (monotone non-increasing tentative distances)
 *   runners = out-edge relaxations from un-garrisoned heaps
 *   the throw = one synchronous land-wide round, every heap's min taken at once
 *   the garrison = the active-frontier mask; a heap whose ember did not shorten
 *                  stops sending runners until a shorter stick breaks the seal
 * No priority queue, no global argmin, no vertex ever compared to another vertex.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

typedef uint64_t nb_u64 __attribute__((may_alias));

#define NB_BUF 2048

/* One stick laid on a heap: it survives only if it is the shortest sliver so far.
 * All distances are non-negative, so IEEE-754 bit patterns order like the doubles
 * and the min can be done as a lock-free unsigned CAS. */
static inline int nb_relax(double *slot, double cand)
{
    nb_u64 nv;
    __builtin_memcpy(&nv, &cand, sizeof nv);
    nb_u64 *p = (nb_u64 *)slot;
    nb_u64 old = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (nv < old) {
        if (__atomic_compare_exchange_n(p, &old, nv, 1,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return 1;
    }
    return 0;
}

/* One throw, done by a single hand: used whenever the un-garrisoned set is so
 * small that fork/barrier cost would exceed the work (long-thin graphs). */
static int nb_round_serial(int nc, const int *curf, int *nxtf,
                           const int *off, const int *eto, const double *ew,
                           unsigned char *inq, double *d)
{
    int nn = 0;
    for (int i = 0; i < nc; ++i) inq[curf[i]] = 0;
    for (int i = 0; i < nc; ++i) {
        int u = curf[i];
        double du = d[u];
        int e1 = off[u + 1];
        for (int e = off[u]; e < e1; ++e) {
            int v = eto[e];
            double c = du + ew[e];
            if (c < d[v]) {
                d[v] = c;
                if (!inq[v]) { inq[v] = 1; nxtf[nn++] = v; }
            }
        }
    }
    return nn;
}

/* Only if scratch memory cannot be had: plain edge-list throws, no CSR. */
static void nb_fallback(int n, int m, const int *src, const int *dst,
                        const double *w, double *d)
{
    for (int it = 0; it < n; ++it) {
        int ch = 0;
        for (int e = 0; e < m; ++e) {
            double du = d[src[e]];
            if (du == INFINITY) continue;
            double c = du + w[e];
            if (c < d[dst[e]]) { d[dst[e]] = c; ch = 1; }
        }
        if (!ch) break;
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    /* Cold ash everywhere; one banked ember where the traveler stands. */
#pragma omp parallel for schedule(static) if (n > 100000)
    for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    int           *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int           *cur = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int           *eto = (int *)malloc((size_t)m * sizeof(int));
    double        *ew  = (double *)malloc((size_t)m * sizeof(double));
    int           *fa  = (int *)malloc((size_t)n * sizeof(int));
    int           *fb  = (int *)malloc((size_t)n * sizeof(int));
    unsigned char *inq = (unsigned char *)calloc((size_t)n, 1);
    if (!off || !cur || !eto || !ew || !fa || !fb || !inq) {
        free(off); free(cur); free(eto); free(ew);
        free(fa);  free(fb);  free(inq);
        nb_fallback(n, m, src, dst, weight, dist_out);
        return;
    }

    /* The roads, filed by the place they leave from. */
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
#pragma omp parallel for schedule(static) if (m > 200000)
    for (int i = 0; i < m; ++i)
        __atomic_fetch_add(&off[src[i] + 1], 1, __ATOMIC_RELAXED);
    for (int i = 0; i < n; ++i) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)(n + 1) * sizeof(int));
#pragma omp parallel for schedule(static) if (m > 200000)
    for (int i = 0; i < m; ++i) {
        int u = src[i];
        int p = __atomic_fetch_add(&cur[u], 1, __ATOMIC_RELAXED);
        eto[p] = dst[i];
        ew[p]  = weight[i];
    }

    int *curf = fa, *nxtf = fb;
    curf[0] = source;
    int g_nc = 1, g_nn = 0, g_stop = 0;
    long long rounds = 0;
    const long long maxr = (long long)n + 4;   /* cannot spin forever */

    int nthr    = omp_get_max_threads();
    int use_par = (nthr > 1) && (m >= 50000);

    if (!use_par) {
        /* Small land: one hand throws, no muster, no atomics. */
        while (g_nc > 0 && rounds++ <= maxr) {
            int nn = nb_round_serial(g_nc, curf, nxtf, off, eto, ew, inq, dist_out);
            int *t = curf; curf = nxtf; nxtf = t;
            g_nc = nn;
        }
    } else {
        int g_serial = (off[source + 1] - off[source]) < 20000;

        /* One muster for the whole campaign: threads are raised once, not per throw. */
#pragma omp parallel
        {
            int buf[NB_BUF];
            for (;;) {
#pragma omp barrier
                if (g_nc == 0 || g_stop) break;   /* every heap wears its garrison */

                if (g_serial) {
#pragma omp single
                    {
                        g_nn = nb_round_serial(g_nc, curf, nxtf, off, eto, ew,
                                               inq, dist_out);
                        int *t = curf; curf = nxtf; nxtf = t;
                        g_nc = g_nn; g_nn = 0;
                        g_serial = 0;
                        if (g_nc > 0 && g_nc <= 1024) {
                            long long ws = 0;
                            for (int i = 0; i < g_nc; ++i) {
                                int u = curf[i];
                                ws += off[u + 1] - off[u];
                            }
                            if (ws < 20000) g_serial = 1;
                        }
                        if (++rounds > maxr) g_stop = 1;
                    }
                    continue;
                }

                /* Lift the garrisons of the heaps that are about to send runners,
                 * so a stick arriving this throw can re-raise them. */
#pragma omp for schedule(static)
                for (int i = 0; i < g_nc; ++i) inq[curf[i]] = 0;

                /* THE THROW: every un-garrisoned heap's runners go out at once and
                 * every heap keeps only its shortest sliver, all in one breath. */
                int cnt = 0;
#pragma omp for schedule(dynamic, 32) nowait
                for (int i = 0; i < g_nc; ++i) {
                    int u = curf[i];
                    double du = dist_out[u];
                    int e1 = off[u + 1];
                    for (int e = off[u]; e < e1; ++e) {
                        int v = eto[e];
                        double c = du + ew[e];
                        if (c < dist_out[v] && nb_relax(&dist_out[v], c)) {
                            /* its ember shortened: the garrison breaks */
                            if (__atomic_exchange_n(&inq[v], (unsigned char)1,
                                                    __ATOMIC_RELAXED) == 0) {
                                buf[cnt++] = v;
                                if (cnt == NB_BUF) {
                                    int b = __atomic_fetch_add(&g_nn, cnt,
                                                               __ATOMIC_RELAXED);
                                    memcpy(nxtf + b, buf, (size_t)cnt * sizeof(int));
                                    cnt = 0;
                                }
                            }
                        }
                    }
                }
                if (cnt) {
                    int b = __atomic_fetch_add(&g_nn, cnt, __ATOMIC_RELAXED);
                    memcpy(nxtf + b, buf, (size_t)cnt * sizeof(int));
                }

#pragma omp barrier
#pragma omp single
                {
                    int *t = curf; curf = nxtf; nxtf = t;
                    g_nc = g_nn; g_nn = 0;
                    g_serial = 0;
                    if (g_nc > 0 && g_nc <= 1024) {
                        long long ws = 0;
                        for (int i = 0; i < g_nc; ++i) {
                            int u = curf[i];
                            ws += off[u + 1] - off[u];
                        }
                        if (ws < 20000) g_serial = 1;
                    }
                    if (++rounds > maxr) g_stop = 1;
                }
            }
        }
    }

    /* Heaps whose ash never caught a spark keep INFINITY: the dead field.
     * dist_out is already the recitation; nothing to extract. */
    free(off); free(cur); free(eto); free(ew);
    free(fa);  free(fb);  free(inq);
}
```

## PREDICTION

PREDICTION: speedup_vs_heap = 6.0

Reasoning behind the number, stated before measuring: heap Dijkstra pays `O(m log n)` with a branch-heavy, pointer-chasing sift and one `O(n)`-serial critical path. The native's method pays `R × (frontier out-degree)` with `R` ≈ hop-depth of the shortest-path tree plus re-activations — on a random or scale-free graph with `m/n ≈ 10` that is roughly 2–4 passes over the edge list, fully parallel, purely streaming reads of CSR. I expect ~10–15× on raw relaxation throughput, discounted to ~6× by the CSR build (which is a real cost the heap baseline also pays, but less of it) and by barrier cost.

## MEASUREMENT

**Not performed.** `dijkstra_bench` and `dijkstra_contract` were declared available in the brief but no tools are callable in this session, so I ran zero of my four allowed improvement iterations and have zero measured numbers. I will not manufacture a table. What I can report honestly:

- Verified by hand, not by machine: the CAS-on-`uint64` min is order-correct for all non-negative doubles including `+INFINITY` (`0x7FF0...0` is the largest non-negative pattern).
- Verified by hand: `nxtf` cannot overflow — `__atomic_exchange_n` on `inq[v]` admits each vertex at most once per round, so at most `n` pushes into an `n`-sized buffer.
- Verified by hand: all worksharing constructs (`for`, `single`) are encountered by every thread in the same order, because `g_nc`, `g_stop` and `g_serial` are only ever read after a barrier and are therefore uniform across threads.
- Verified by hand on the 4-node counterexample above: the revocable garrison returns `d[v] = 3`; the native's permanent garrison returns `10`.

Anyone running `dijkstra_bench` on this should treat the 6.0 as an untested prior, not a result.

## VERDICT

The native's idea is real and it is not Dijkstra. Its content is: **finalization is a local fixed-point test, not a global argmin**, which is precisely why the whole land can throw at once. That is a correct and valuable insight, and it survives translation intact. Its one defect is the word *"for good"* — an ember can sit still for a throw and shorten later — and I repaired that by making the garrison revocable rather than by importing a priority queue. The resulting kernel contains no heap, no bucket structure, and no comparison between two vertices anywhere.

Named risks, and how each is guarded in the shipped code (per step 4 — nothing risky is left unaddressed):

1. **High-diameter graphs (road networks, chains): thousands of throws, each with a tiny frontier, so barrier cost dominates.** Guarded twice: (a) the entire campaign runs inside a *single* `omp parallel` region, so threads are mustered once rather than once per throw; (b) whenever the frontier's total out-degree drops below 20 000, the throw is executed by one thread inside `omp single` — no atomics, no fork, no scatter — so a long thin graph degenerates to a tight serial scan costing tens of nanoseconds per round.
2. **Tiny graphs, where OpenMP setup alone exceeds the whole heap-Dijkstra runtime.** Guarded: `use_par` is false below `m = 50 000` or on a single core, and the code then never enters a parallel region at all; the CSR build's `if(m > 200000)` clauses do the same for the setup phase.
3. **Hub vertices with enormous out-degree stalling one thread.** Partially guarded by `schedule(dynamic, 32)`, which keeps the other threads fed; a single vertex of degree ≫ m/threads would still serialize that throw. Not fully solved — I would split oversized adjacency lists across threads if I could measure and confirm it matters.
4. **Re-activation blow-up** (the Bellman-Ford tail risk: a vertex's ember shortening many times, up to `O(n·m)` total work on adversarial weight structures). Mitigated by frontier de-duplication and by always re-reading the freshest ember at scan time, which collapses the common repeats, and hard-bounded by the `n+4` round cap so the kernel can never fail to terminate. This is the residual risk I have *not* eliminated: on a graph engineered to maximize re-activations, this kernel loses to a heap and there is no local test that would save it. The honest fix is distance-ordered frontier processing — which is exactly the global comparison the native refuses, so I did not smuggle it in.

If the measurement comes back below 1.0, the most likely cause in order is: (2) the graph is small, (4) the graph re-activates heavily, (1) the graph is a road network deeper than the serial fast path compensates for.

*Separately, and unrelated to this task: the `claude.ai` and `PubMed` MCP connectors are unauthorized in this environment. They'd need authorizing via claude.ai connector settings (or `claude mcp` / `/mcp` in an interactive session) before their tools become usable — I can't run the OAuth flow from here.*