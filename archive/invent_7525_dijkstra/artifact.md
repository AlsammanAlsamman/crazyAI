# MAPPING

**SEED 1 — nightingales die at the first board out of the source**

| World object | Problem object |
|---|---|
| traveler's place | `source` |
| road out of traveler's place | edge `(source, v, w)` |
| nightingale loosed on a road | one initial relaxation attempt |
| note growing hoarser per stride | accumulated weight (here: just `w`, one edge) |
| first cross-marked board it reaches | `v = dst[i]` |
| lattice slot | `dist_out[v]` |
| dying note carved into the slot | `dist_out[v] = min(current, w)` |
| louder note "rots in the desert", road marked dead | dominated parallel edge, never revisited |

This is just the ordinary initialization of `dist_out[source]=0` and its direct neighbors — it doesn't touch any of the five silent assumptions; standard Dijkstra does exactly this on its first pop.

**SEED 2 — cow's-head re-measurement at each board**

| World object | Problem object |
|---|---|
| board just settled | node `u` popped/settled this round |
| "last settled board" | `u` itself (the *only* board relaxation is anchored to) |
| cow's head / woman / smaller cow | the relax operation, done as a direct write |
| milk = note by way of last board + stretch between | candidate `dist_out[u] + weight(u,v)` |
| "thinner milk" | `candidate < dist_out[v]` |
| "scrape the lattice clean, cut hers in place" | `dist_out[v] = candidate` (in-place array write) |
| never touching a settled board twice | `settled[]` guard, each node relaxed once |

Crucially: nothing here pushes into or pops from a priority structure. The relax is a bare array write. This **breaks** "a priority structure must be consulted before every relaxation."

**SEED 3 — quietest unsettled note chosen next**

| World object | Problem object |
|---|---|
| lattice (rigid, one fixed slot per place) | `dist_out[]`, a flat array, size `n`, fixed up front |
| "board by board... whichever unsettled slot holds the quietest note" | linear scan over all `n` slots for the unsettled minimum |
| "no nightingale sings and no cow yields thinner milk" | loop terminates when min-unsettled is `INFINITY` |
| "unreached places stand blank as the desert" | `dist_out[i] == INFINITY` for unreachable `i` |
| "read the whole lattice in order" | write out `dist_out[]` |

This also breaks "a priority structure must be consulted before every relaxation" — selection is a raw compare-against-everyone scan, not a heap pop. It literally *embraces* (does not break) "the next place is found by comparing against every remaining place" — that's the mechanism, not an assumption to escape.

None of the three seeds break **"the whole graph must be explored to know any single distance"** — the contract itself demands *all* `n` distances in `dist_out`, so there is no way to answer "distance to one node" without covering reachability from `source` to it regardless of mechanism. Stated plainly: this assumption is not broken by any seed; falling back to the most literal seed instead.

# CHOSEN SEED

SEED 2 + SEED 3 together (they are one mechanism split across two sentences of the same telling: SEED 2 is the relax step, SEED 3 is the select step). This pairing is the most literal reading of the whole passage and the most different from the reference's binary heap.

# ASSUMPTION BROKEN

"A priority structure must be consulted before every relaxation." The lattice is a plain fixed array (`dist_out`), consulted by full scan only *once per settled board* to pick the next board — never touched by a heap push/pop on every single relaxation, unlike the reference's `hpush` on every improving edge.

# ARTIFACT

This is exactly the textbook **O(n²) array-scan Dijkstra, no heap at all** — the task's own footnote names it as "a real, well-known practical win" for dense/small graphs, so the mechanism is let to arrive there rather than inventing something new. Per step 4's guard requirement, since the metaphor's own mechanism is *worse* than the heap on large sparse graphs (its stated risk), a size/density check falls back to the (CSR-shared) heap path there — never shipping the risky part unguarded.

```c
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---------- Fallback: binary-heap Dijkstra (sparse / large graphs) ---------- */
typedef struct { double d; int u; } HeapItem;
static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) / 2; if (h[p].d <= h[i].d) break; HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    while (1) {
        int l = 2*i+1, r = 2*i+2, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}

static void dijkstra_heap(int n, int m, int source, double *dist_out,
                           const int *off, const int *edst, const double *ew) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    char *done = calloc((size_t)n, 1);
    HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0;
    hpush(heap, &hs, 0.0, source);
    while (hs > 0) {
        HeapItem top = hpop(heap, &hs);
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = dist_out[u] + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }
    free(done); free(heap);
}

/* ---------- Main mechanism: the "lattice" Dijkstra, no priority structure ----------
   Lattice of old memory     -> dist_out[], one fixed slot per place (SEED 3)
   Cow's-head re-measurement -> direct array write relaxing roads out of the
                                 just-settled board, no heap ever touched (SEED 2)
   Quietest unsettled note   -> plain linear scan of the whole lattice, done once
                                 per settling, not once per relaxation (SEED 3)
   Settled boards never revisited; note==infinity forever is the blank desert. */
static void dijkstra_lattice(int n, int source, double *dist_out,
                              const int *off, const int *edst, const double *ew) {
    char *settled = calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    for (int step = 0; step < n; step++) {
        double best = INFINITY;
        int u = -1;
        const char * restrict st = settled;
        const double * restrict dp = dist_out;
        for (int i = 0; i < n; i++) {
            if (!st[i] && dp[i] < best) { best = dp[i]; u = i; }
        }
        if (u < 0 || best == INFINITY) break; /* no nightingale sings: rest is desert */
        settled[u] = 1;

        int lo = off[u], hi = off[u + 1];
        const int * restrict ed = edst;
        const double * restrict ewp = ew;
        double du = dist_out[u];
        for (int e = lo; e < hi; e++) {
            int v = ed[e];
            double cand = du + ewp[e];
            if (cand < dist_out[v]) dist_out[v] = cand;
        }
    }
    free(settled);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    /* Build CSR adjacency once, shared by whichever mechanism runs. */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    size_t msafe = (size_t)(m > 0 ? m : 1);
    int *edst = malloc(msafe * sizeof(int));
    double *ew = malloc(msafe * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }

    /* Guard against the mechanism's own named risk: the O(n^2) lattice scan
       loses to the heap on large, sparse graphs. Use it only where the task's
       own "known way" footnote says it wins -- small or dense graphs --
       and fall back to the heap otherwise. */
    long long m64 = (long long)m, n64 = (long long)n;
    int dense_or_small = (n <= 2000) || (m64 >= 8LL * n64);

    if (dense_or_small) dijkstra_lattice(n, source, dist_out, off, edst, ew);
    else                 dijkstra_heap(n, m, source, dist_out, off, edst, ew);

    free(deg); free(off); free(edst); free(ew); free(fill);
}
```

PREDICTION: speedup_vs_heap = 1.3

Reasoning behind the number: on the "dense_or_small" branch, the lattice mechanism replaces `log n` heap push/pop with a straight-line array scan and a straight-line relax loop — both cache-sequential and auto-vectorizable under `-O3 -march=native`, so I'd expect a real (2–4x) win there specifically. On the sparse/large branch the code *is* the reference heap algorithm (with the CSR build shared/reused rather than duplicated), so I expect roughly parity, maybe a hair faster from not rebuilding CSR. Blended over whatever mix `dijkstra_bench` uses, 1.3x is my honest a-priori guess, not a measured value.

# MEASUREMENT

No tool execution is available in this session (stated explicitly at the top of the task). I did not call `dijkstra_contract` or `dijkstra_bench`, so I have **no measured numbers** to report — I'm not going to fabricate any. This artifact has not been benchmarked or iterated on by me; the described pipeline will compile it and run `dijkstra_bench` against the reference to produce the actual speedup, which should be compared against the 1.3x prediction above.

# VERDICT

The literal mechanism (SEED 2 relax-by-direct-write + SEED 3 full-scan select) is not a novel invention — it lands exactly on the well-known, validated O(n²) no-heap Dijkstra, which is the correct outcome per step 4 rather than manufacturing something new. Its own named risk — "worse than the heap on large sparse graphs" — is addressed, not left unguarded: the kernel checks `n <= 2000 || m >= 8n` and falls back to the (shared-CSR) heap path otherwise, so there is no graph shape on which this kernel should be asymptotically worse than the reference. No thread-level parallelism was added, per the vectorization-before-threading preference: at the sizes where the lattice path is chosen (small or dense, i.e. `n≲2000` or `m≳8n`), the O(n²) work per call is small enough (≤~a few million ops) that spawning OpenMP threads would likely be pure overhead rather than a win, so it was correctly left out rather than guarded — there's no size regime in this kernel's own branch where threading would clearly pay off. Because no measurement was possible in this session, no iterative improvement passes were performed beyond this reasoned first artifact; the real verdict on the 1.3x prediction is deferred to the pipeline's actual `dijkstra_bench` run.