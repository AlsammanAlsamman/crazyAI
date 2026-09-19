## MAPPING (per SEED)

| World object | Problem object | Assumption it breaks (of the 5 listed) |
|---|---|---|
| **SEED 1** — two-rowed pit-board, one pit per place, thread-length in fingerwidths | Flat array `dist_out[0..n-1]` (one cell per node), built once, indexed directly — no auxiliary priority structure; edge weight = fingerwidth-count = `weight[i]` | Sets up **A2** ("a priority structure must be consulted before every relaxation") to be broken: the board *is* the array, nothing else exists to consult. |
| **SEED 2** — each round choose the unsettled pit with fewest seeds by looking at them all; sow a palmful along each thread one seed per fingerwidth, replace the far pit's heap only when the sown count is smaller; empty pit = traveler's start, heaped-to-the-brim pits = unreached | Node selection = **linear scan** over every `!done[v]` for `min(dist[v])` each round (no heap); relaxation = `nd = dist[u]+w[e]; if (nd < dist[v]) dist[v]=nd`; init = `dist[source]=0`, else `+INFINITY` | Directly breaks **A2**: there is no priority queue object anywhere in the story — "choosing the fewest-seed pit" is a brute force comparison against every remaining pit, every round. |
| **SEED 3** — threads that never once lower a neighbor's pile (ghost roads, duplicate "two ships under one name", swept corridors) get cut loose before the counting even starts | Pre-pass that would collapse duplicate parallel `(u,v)` edges to their minimum weight and discard edges from components that can never affect any `dist[v]` | Targets **A5** ("the whole graph must be explored to know any single distance") — if dead threads are identified up front, that part of the graph never needs to be touched at all. |

## CHOSEN SEED

**SEED 2** — the sowing/scanning seed. It is the most literal (the native gives an explicit, mechanical procedure: scan all unsettled pits for the minimum, then walk seeds one at a time along threads, replace-if-smaller) and it is the one that most directly contradicts the reference kernel's structure: the reference builds and maintains a binary heap (`HeapItem *heap`, `hpush`/`hpop`) as its priority structure. The native's board has no such second structure — settling a pit is done by inspecting every remaining pit directly. SEED 3's edge-pruning is a real but secondary optimization layered on top; SEED 1 is just the container the algorithm runs in. SEED 2 is the algorithm itself.

## ASSUMPTION BROKEN

**A2**: "a priority structure must be consulted before every relaxation." The kernel below never allocates, pushes to, or pops from any heap/queue. Selecting the next pit to settle is a flat `O(n)` scan over `dist_out[]`/`done[]`, exactly as described ("I choose whichever unsettled pit holds the fewest seeds").

## ARTIFACT

Literal object mapping used in the code:
- **place** → node index `0..n-1`
- **road / thread** → directed edge `(src[i], dst[i])`, fingerwidth-length → `weight[i]`
- **pit's seed-heap** → `dist_out[v]`, initialized "heaped to the brim" (`INFINITY`) for every place except the traveler's own **empty pit** (`dist_out[source] = 0`)
- **blue flame marking a pit settled** → `done[v] = 1`, touched exactly once, never rescanned
- **"choose whichever unsettled pit holds the fewest seeds"** → bare linear scan over all `v` with `!done[v]`, no heap
- **sowing a palmful of seeds along a thread, replacing the far pit's heap only when smaller** → the relaxation `nd = dist_out[u] + w; if (nd < dist_out[v]) dist_out[v] = nd`

```c
#include <stdlib.h>
#include <math.h>

/* Literal mapping of the pit-board / sowing story to O(n^2 + m) Dijkstra:
   - place        -> node (pit)
   - road         -> directed edge (thread), thread-length -> weight
   - pit's heap   -> dist_out[v], starts "heaped full" = INFINITY, except
                     the traveler's own empty pit -> dist_out[source] = 0
   - blue flame   -> done[] flag marking a pit settled, touched once
   - "choose whichever unsettled pit holds fewest seeds" -> a full linear
                     scan over every undone pit each round (NO priority
                     queue / heap object exists anywhere in this story)
   - sowing a palmful of seeds one at a time along each thread, replacing
     the far pit's heap only when the sown count is smaller -> the
     classic Dijkstra relaxation: nd = dist[u] + w; if (nd < dist[v]) ...
*/
void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    /* Build the "threads leaving each pit" structure (CSR adjacency) */
    int *deg = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew = (double *)malloc((size_t)m * sizeof(double));
    int *fill = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }

    /* Every pit starts heaped to the brim (unreached), the traveler's own
       pit starts empty. */
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    char *done = (char *)calloc((size_t)n, 1);

    for (int round = 0; round < n; round++) {
        /* "each round, I choose whichever unsettled pit holds the fewest
           seeds" -- a bare scan across every remaining pit, no heap. */
        int u = -1;
        double best = INFINITY;
        for (int v = 0; v < n; v++) {
            if (!done[v] && dist_out[v] < best) {
                best = dist_out[v];
                u = v;
            }
        }
        if (u == -1) break; /* everything left is unreachable: still full pits */

        /* touch it with the blue flame: settled, never scanned again */
        done[u] = 1;

        /* sow along every thread leaving this pit */
        double du = dist_out[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            if (done[v]) continue;
            double nd = du + ew[e];
            if (nd < dist_out[v]) dist_out[v] = nd;
        }
    }

    free(deg); free(off); free(edst); free(ew); free(fill); free(done);
}
```

PREDICTION: speedup_vs_heap = 1.8

Reasoning behind the number: for the kind of graph the task itself flags as favorable ("dense or small graphs a plain O(n^2) array scan... is a real, well-known practical win"), dropping the heap removes `O(m log n)` pointer-chasing/sift work and replaces it with a tight, branch-predictable, cache-friendly `O(n)` scan per round plus a straight CSR sweep for relaxation — no allocation churn from repeated `hpush`. I expect a modest win (not a dramatic one) on moderate/dense graphs, and I expect this same kernel to *lose* badly on large sparse graphs (`m ≈ n`), since there `n^2` dominates `(n+m) log n`. 1.8 reflects a bet on the benchmark leaning toward the dense/moderate regime the prompt itself calls out, not a universal claim.

## MEASUREMENT

Not performed. No `dijkstra_bench`/`dijkstra_contract` tool invocation is available in this session (explicitly confirmed: "No tools are available in this session... the pipeline will compile and measure it"). I am reporting this plainly rather than fabricating a benchmark number. The kernel above is complete, matches the fixed contract exactly, and is ready for the external pipeline to compile and measure against the reference heap kernel; the PREDICTION line above was committed before any such run, as required.

## VERDICT

Unverified pending actual measurement. Honest expectation, stated plainly: this is the textbook O(n²+m) array-scan Dijkstra, arrived at completely literally from the native's description (no heap object appears anywhere in the story, only "look at every unsettled pit and pick the smallest"). It should beat the reference heap implementation on dense/small graphs and should lose on large sparse graphs — I am not claiming a universal win, only reporting what this literal translation predicts and flagging that the number above is a prediction, not a measured result.