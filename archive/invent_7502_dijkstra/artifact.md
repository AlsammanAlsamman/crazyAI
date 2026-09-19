# MAPPING

**SEED 1** — "Water poured all at once from the traveler's peg races through bore-matched hollow reeds, and first-arrival wetness marks the shortest road to each other peg."

| World object | Problem object |
|---|---|
| traveler's peg (rented, detachable stand driven in among the others) | the source node, placed among the fixed node set |
| every other peg, set out on the suitcase-shaped square | every other node `0..n-1` |
| hollow reed between two pegs sharing a road | a directed edge `src[i]->dst[i]` |
| reed's bore-length, cut to match the road's true walking-length, never guessed | `weight[i]`, used exactly (not quantized/approximated) |
| pouring the same water into every reed mouth at the peg at once, piped not carried | relaxing **all** outgoing edges of a just-wetted node **simultaneously**, driven by a physical clock (distance-as-time), not fetched one at a time by a comparison structure |
| the water racing at a constant rate | edge traversal cost accrues at unit rate per unit bore-length = weight |
| first wet mark at a peg | the minimum accumulated distance to that node |

Breaks: *"a priority structure must be consulted before every relaxation"* and *"the next place to finalize is found by comparing against every remaining place."* Nothing is ever compared against the whole remaining set — arrival order is produced by the water's own physics (time/distance), not by an explicit min-search.

**SEED 2** — "Wherever two streams reach a peg together, the later wet mark gets scraped off before it dries, keeping only the first."

| World object | Problem object |
|---|---|
| two streams reaching one peg | two candidate relaxations `dist[u]+w` arriving at the same node |
| scraping the later mark, keeping the first | `if (nd < dist[v]) dist[v] = nd` |

Breaks: nothing new — this is literally the standard relax/min-take step already present in the reference kernel. Least novel seed; almost a restatement of the known way.

**SEED 3** — "Reeds that never wet a far peg are pulled out whole and laid aside dry; a peg the water never reaches is marked hollow with no road home."

| World object | Problem object |
|---|---|
| reed that never wets its far end | an edge that never contributes to any shortest-path improvement |
| pulling it out, laying it aside dry | simply never re-examining it (already the reference's natural behavior — it's never dequeued) |
| peg the water never reaches | node left at `INFINITY` |

Breaks: *"the whole graph must be explored to know any single distance"* only mildly — this is basically what any correct SSSP implementation already does (unreachable components are never touched). Not much departure from the known way either.

Seed 1 is the most literal and the most different from binary-heap Dijkstra: it removes the comparison-based priority structure entirely and replaces it with a **physical clock** (bucketed by "time/distance") plus **simultaneous, parallel firing of every outgoing reed at once**.

# CHOSEN SEED
SEED 1.

# ASSUMPTION BROKEN
"A priority structure must be consulted before every relaxation" and "the next place to finalize is found by comparing against every remaining place." Instead of a binary heap (or an O(n) linear scan) picking one min-node at a time, the water's own travel time sorts nodes into discretized "arrival time" buckets. All nodes whose water arrived in the same time-window are drained and have **all** their outgoing reeds opened **at once** (parallelized with OpenMP — "piped, not carried"), repeating a bucket only until it truly runs dry before the clock advances to the next bucket. This is a literal Δ-stepping-style bucket queue rather than a comparison-based heap.

Concrete objects:
- place = graph node
- road = directed edge, road's "true walking length" = edge weight
- peg = an integer time-bucket slot `floor(dist/Δ)` a node currently sits in
- pouring water into every reed mouth at once = relaxing every outgoing edge of every node in the current bucket in an `omp parallel for`
- scraping off the later wet mark = the double-checked `if (nd < dist_out[w])` inside a critical section (only the smaller candidate survives)
- waiting only as long as the slowest wetted reed in this peg's cluster = repeat-draining the current bucket until it produces no more insertions into itself before moving the clock forward
- a peg the water never reaches = `dist_out[v] == INFINITY`

# ARTIFACT

```c
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* a "peg cluster": the set of nodes currently sitting in one time-bucket */
typedef struct { int *a; int len; int cap; } Bucket;

static void bucket_push(Bucket *b, int v) {
    if (b->len == b->cap) {
        int ncap = b->cap ? b->cap * 2 : 8;
        b->a = realloc(b->a, (size_t)ncap * sizeof(int));
        b->cap = ncap;
    }
    b->a[b->len++] = v;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n <= 0) return;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0 || n == 1) return;

    /* CSR: peg -> its outgoing reeds */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(deg); free(fill);

    /* the widest reed's bore sets the pegboard's clock granularity */
    double maxW = 0.0;
    for (int i = 0; i < m; i++) if (weight[i] > maxW) maxW = weight[i];
    const int BUCKET_TARGET = 256;
    double delta = (maxW > 0.0) ? (maxW / (double)BUCKET_TARGET) : 1.0;
    if (!(delta > 0.0)) delta = 1.0;
    int L = BUCKET_TARGET + 4; /* circular pegboard: a bucket can never be re-touched
                                  once the clock has moved past its whole reachable window */

    Bucket *buckets = calloc((size_t)L, sizeof(Bucket));
    double *lastVal = malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) lastVal[i] = -1.0;

    /* pour water into the traveler's own peg */
    bucket_push(&buckets[0], source);

    long long curBucket = 0;
    long long emptyStreak = 0;
    int *batch = NULL; int batchCap = 0;

    while (emptyStreak < L) {
        int slot = (int)(((curBucket % L) + L) % L);
        if (buckets[slot].len == 0) { emptyStreak++; curBucket++; continue; }
        emptyStreak = 0;

        for (;;) {
            int blen = buckets[slot].len;
            if (blen == 0) break;
            if (blen > batchCap) { batch = realloc(batch, (size_t)blen * sizeof(int)); batchCap = blen; }
            memcpy(batch, buckets[slot].a, (size_t)blen * sizeof(int));
            buckets[slot].len = 0; /* drained; fresh pours during this round land here anew */

            #pragma omp parallel for schedule(dynamic, 64)
            for (int bi = 0; bi < blen; bi++) {
                int u = batch[bi];
                double du; int doWork = 0;
                #pragma omp critical(pegboard)
                {
                    du = dist_out[u];
                    if (lastVal[u] != du) { lastVal[u] = du; doWork = 1; }
                }
                if (!doWork) continue;
                for (int e = off[u]; e < off[u + 1]; e++) {
                    int w = edst[e];
                    double nd = du + ew[e];
                    if (nd < dist_out[w]) {           /* dirty pre-check, safe: dist only shrinks */
                        #pragma omp critical(pegboard)
                        {
                            if (nd < dist_out[w]) {   /* first wet mark wins; scrape the rest */
                                dist_out[w] = nd;
                                long long g = (long long)floor(nd / delta);
                                if (g < curBucket) g = curBucket; /* fp-boundary safety clamp */
                                int wslot = (int)(((g % L) + L) % L);
                                bucket_push(&buckets[wslot], w);
                            }
                        }
                    }
                }
            }
        }
        curBucket++;
    }

    for (int i = 0; i < L; i++) free(buckets[i].a);
    free(buckets); free(lastVal); free(batch);
    free(off); free(edst); free(ew);
}
```

# PREDICTION: speedup_vs_heap = 0.5

Reasoning for the prediction: every actual state change (the dedup check *and* every successful relaxation) goes through a single named OpenMP critical section, so parallelism only pays off when many candidate relaxations fail the pre-check cheaply in parallel; on typical sparse benchmark graphs the lock will be entered often relative to useful work, plus there is bucket bookkeeping (realloc-based push, repeated drain-until-stable per bucket, up to `L=260` empty-bucket scans at the tail) that a tuned binary heap doesn't pay. I expect this to be honestly *slower* than heap Dijkstra on most cases, roughly half the speed, with a chance of being closer to parity or even ahead only on dense, high-degree graphs where the parallel batch relaxation has real work to amortize the lock cost.

# MEASUREMENT
No `dijkstra_bench` / `dijkstra_contract` tool was available to me in this session (the tool list I was actually given contains only Docs/Gmail/Drive/Slack tools, not the numeric ones named in the task). I did not fabricate a number. Per the task setup, the artifact above is meant to be compiled and measured by the surrounding pipeline; the prediction line was recorded before any such run could occur, as required.

# VERDICT
Unmeasured in this session — reported plainly rather than invented. The construction is a faithful, literal translation of the "pour water into every reed at once, first wet mark wins, wait for the slowest reed in this cluster before advancing the clock" story into a Δ-stepping-style circular bucket queue with OpenMP-parallel batch relaxation, replacing the heap's per-node comparison with a physical-time (distance) clock. It is genuinely different from both the reference binary-heap Dijkstra and the O(n²) array-scan variant. My own honest expectation, based on the single shared critical section and the bucket bookkeeping overhead, is that it will underperform the heap on typical sparse test cases (predicted ~0.5×) and should be re-measured — and, if the pipeline's actual numbers come back, the next iteration should try splitting the single global critical section into per-node-bucket locks or lock-free atomic-min on `dist_out`'s bit pattern to reduce contention, which was not attempted here due to no measurement feedback being available to justify the added complexity.