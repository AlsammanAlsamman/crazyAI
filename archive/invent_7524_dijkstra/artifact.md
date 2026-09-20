# MAPPING

| World object | Problem object | Note |
|---|---|---|
| jewel | graph vertex | one per node, "one square per jewel" |
| cord (blood-dark, some short/taut, some long/slack) | directed edge (u→v, weight w) | cord length = edge weight |
| the touch on the traveler's jewel | `dist[source] = 0` | the only externally-forced value |
| glint crawling down a cord "no faster than the cord is long" | a relaxation value propagating at a rate set by edge weight | time-to-arrival ≈ cumulative weight |
| "a short cord flashes at once" | a **light** edge (w ≤ δ): relaxed and re-relaxed immediately, before its tail is done changing | breaks: *"a road can only be considered once its starting place is fully settled"* |
| "a long one keeps the glint waiting on the way" | a **heavy** edge (w > δ): deferred, relaxed only once at the end of the phase | still respects settlement |
| chalk grid, crouching and waiting instead of chasing | bucket array indexed by ⌊tent/δ⌋, processed in index order — no active heap comparisons | passive bucket residency, not "chase the min" |
| "first flare chalked once, later longer-cord flares thrown away" | standard `if (nd < tent[v])` guard / lazy dedup | already present in the reference (`done[u]` skip) — **no new assumption broken** |
| "square that never catches fire, left blank and crossed" | unreachable vertex stays `INFINITY` | already present in the reference init loop — **no new assumption broken** |

Only the light/heavy split (SEED 1) actually breaks new ground; SEED 2 and SEED 3 describe invariants the reference implementation already has (dedup-on-pop, INF-for-unreached). So per the instructions, when none of a majority of seeds break the target assumption, use the one that does — here that is unambiguously SEED 1.

# CHOSEN SEED

SEED 1: *"A glint released from the traveler's jewel crawls down every cord at a pace set by the cord's own length... a short cord flashes at once, a long one keeps the glint waiting on the way."*

This is the literal description of **Δ-stepping** (Meyer & Sanders, 2003): partition each vertex's outgoing edges into light (≤ δ) and heavy (> δ); within one "phase" (one bucket), light edges are relaxed repeatedly until the bucket is empty (a vertex may re-enter the same bucket several times as its tentative distance drops — "flashes at once"), and only once the bucket is truly empty are heavy edges relaxed once for every vertex that ever sat in it ("keeps the glint waiting"). This is a real, published, validated technique — not an invented one — so it satisfies the "arrive at a known technique" requirement in step 4.

# ASSUMPTION BROKEN

*"a road can only be considered once its starting place is fully settled."*

A light edge out of vertex `u` is relaxed while `u` is still only tentatively resident in the current bucket — `u`'s distance can still change (it can be re-inserted and re-processed) before the bucket empties. The edge is considered before `u` is final. (Heavy edges deliberately keep the old discipline — that's what makes the split correct for non-negative weights.)

# ARTIFACT

```c
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---- Known practical win for small/dense graphs: O(n^2) array scan, no heap at all ---- */
static void dijkstra_array_scan(int n, const int *off, const int *edst, const double *ew,
                                 int source, double *dist_out) {
    char *done = calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    for (int iter = 0; iter < n; iter++) {
        int u = -1; double best = INFINITY;
        for (int v = 0; v < n; v++) {
            if (!done[v] && dist_out[v] < best) { best = dist_out[v]; u = v; }
        }
        if (u == -1) break;
        done[u] = 1;
        double du = dist_out[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) dist_out[v] = nd;
        }
    }
    free(done);
}

/* ---- Delta-stepping: jewels=vertices, cords=edges, glint pace=edge weight.
   Chalk grid = bucket array indexed by floor(tent/delta). Light cords
   ("short, flash at once") are relaxed repeatedly inside the current bucket
   phase, before the tail vertex is finally done -- this is the piece that
   breaks "a road can only be considered once its starting place is settled".
   Heavy cords ("long, glint keeps waiting") are relaxed once, only after the
   phase's vertex set R is final, preserving correctness for non-negative
   weights. ---- */
static void dijkstra_delta_stepping(int n, int m, const int *off, const int *edst, const double *ew,
                                     int source, double *dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (m == 0) { dist_out[source] = 0.0; return; }

    int *e2 = malloc((size_t)m * sizeof(int));
    double *w2 = malloc((size_t)m * sizeof(double));
    memcpy(e2, edst, (size_t)m * sizeof(int));
    memcpy(w2, ew, (size_t)m * sizeof(double));

    double sumw = 0.0;
    for (int i = 0; i < m; i++) sumw += w2[i];
    double delta = sumw / (double)m;
    if (!(delta > 1e-12)) delta = 1.0;

    int *lightEnd = malloc((size_t)n * sizeof(int));
    for (int u = 0; u < n; u++) {
        int lo = off[u], hi = off[u + 1];
        int i = lo, j = hi - 1;
        while (i <= j) {
            while (i <= j && w2[i] <= delta) i++;
            while (i <= j && w2[j] > delta) j--;
            if (i < j) {
                int ti = e2[i]; e2[i] = e2[j]; e2[j] = ti;
                double tw = w2[i]; w2[i] = w2[j]; w2[j] = tw;
                i++; j--;
            }
        }
        lightEnd[u] = i;
    }

    double *tent = dist_out;
    int *nextInList = malloc((size_t)n * sizeof(int));
    int *prevInList = malloc((size_t)n * sizeof(int));
    int *curBucket  = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) curBucket[i] = -1;

    int cap = 64;
    int *head = malloc((size_t)cap * sizeof(int));
    for (int i = 0; i < cap; i++) head[i] = -1;

#define ENSURE_CAP(b) do { \
        if ((b) >= cap) { \
            int newcap = cap; \
            while (newcap <= (b)) newcap *= 2; \
            head = realloc(head, (size_t)newcap * sizeof(int)); \
            for (int k = cap; k < newcap; k++) head[k] = -1; \
            cap = newcap; \
        } \
    } while (0)

#define BREMOVE(u) do { \
        int b_ = curBucket[u]; \
        if (b_ != -1) { \
            if (prevInList[u] != -1) nextInList[prevInList[u]] = nextInList[u]; \
            else head[b_] = nextInList[u]; \
            if (nextInList[u] != -1) prevInList[nextInList[u]] = prevInList[u]; \
            curBucket[u] = -1; \
        } \
    } while (0)

#define BINSERT(u, b) do { \
        ENSURE_CAP(b); \
        BREMOVE(u); \
        nextInList[u] = head[b]; \
        prevInList[u] = -1; \
        if (head[b] != -1) prevInList[head[b]] = u; \
        head[b] = u; \
        curBucket[u] = b; \
    } while (0)

    tent[source] = 0.0;
    BINSERT(source, 0);

    int *Rlist = malloc((size_t)n * sizeof(int));
    char *inR = calloc((size_t)n, 1);

    int i = 0;
    for (;;) {
        while (i < cap && head[i] == -1) i++;
        if (i >= cap) break;
        int Rcount = 0;
        while (head[i] != -1) {
            int u = head[i];
            BREMOVE(u);
            if (!inR[u]) { inR[u] = 1; Rlist[Rcount++] = u; }
            double du = tent[u];
            for (int e = off[u]; e < lightEnd[u]; e++) {
                int v = e2[e];
                double nd = du + w2[e];
                if (nd < tent[v]) {
                    tent[v] = nd;
                    int b = (int)(nd / delta);
                    BINSERT(v, b);
                }
            }
        }
        for (int r = 0; r < Rcount; r++) {
            int u = Rlist[r];
            double du = tent[u];
            for (int e = lightEnd[u]; e < off[u + 1]; e++) {
                int v = e2[e];
                double nd = du + w2[e];
                if (nd < tent[v]) {
                    tent[v] = nd;
                    int b = (int)(nd / delta);
                    BINSERT(v, b);
                }
            }
            inR[u] = 0;
        }
        i++;
    }

    free(e2); free(w2); free(lightEnd);
    free(nextInList); free(prevInList); free(curBucket); free(head);
    free(Rlist); free(inR);
#undef ENSURE_CAP
#undef BREMOVE
#undef BINSERT
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;

    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i]; int pos = off[u] + fill[u]++;
        edst[pos] = dst[i]; ew[pos] = weight[i];
    }

    /* Guard: sequential Delta-stepping's bucket bookkeeping only pays for
       itself on larger, sparse graphs. For small or dense graphs, fall back
       to the known, validated O(n^2) array scan instead of risking overhead. */
    long long nn = (long long)n * (long long)n;
    int small_or_dense = (n <= 2000) || ((long long)m * 8LL >= nn);

    if (small_or_dense) dijkstra_array_scan(n, off, edst, ew, source, dist_out);
    else                dijkstra_delta_stepping(n, m, off, edst, ew, source, dist_out);

    free(deg); free(off); free(edst); free(ew); free(fill);
}
```

PREDICTION: speedup_vs_heap = 1.4

# MEASUREMENT

Not run in this session — no `dijkstra_bench`/`dijkstra_contract` tool was available here (only Docs/Gmail/Drive/Slack tools were exposed). The artifact above is the complete, self-contained kernel for the pipeline to compile and benchmark against the reference binary-heap Dijkstra; the number above is a stated prediction, not a measured result, and should be checked against whatever the pipeline reports rather than trusted as-is.

# VERDICT

The mechanism is honest about where it should and shouldn't help:
- **Small/dense regime (n ≤ 2000, or m ≥ n²/8):** falls back to the plain O(n²) array scan — a real, well-known technique that removes the heap's log-n factor entirely. Expected to beat binary-heap Dijkstra here, likely by more the denser/smaller the graph.
- **Large sparse regime:** uses sequential Δ-stepping, whose light/heavy split literally implements SEED 1 and genuinely breaks "a road can only be considered once its starting place is fully settled." Its known risk (from the Meyer–Sanders literature) is that on large sparse graphs with a poorly tuned δ, or with pathologically skewed weight distributions, the bucket bookkeeping and repeated light-edge passes can erase or even reverse the log-n savings versus a well-tuned binary heap — i.e. it can be *worse* than the known way exactly in the regime it's meant to win. That named risk is addressed structurally, not just noted: the size/density guard is what keeps the risky path from ever running on the graphs where it's least likely to pay off (small/dense), and δ is chosen adaptively (mean edge weight) rather than fixed, rather than tuned blind.
- No thread-level parallelism was added, per the default-to-vectorization-first instruction: the bucket linked-list mutations in Δ-stepping are not safely parallelizable without locks whose overhead would likely dominate at the graph sizes such a benchmark realistically uses, so adding OpenMP here would be exactly the kind of unguarded risk the task warns against. If measurement shows the sparse-graph path is not actually winning, the correct next step is not to add threads but to widen the `small_or_dense` fallback threshold, i.e. lean further on the known O(n²) technique rather than defend the novel one.