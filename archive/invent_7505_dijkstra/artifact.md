## MAPPING (world‑object → problem‑object, per SEED)

**SEED 1** — *"A heap of sticks stands for a place's distance, and crossing its chessboard on foot with a pawn turns that distance into a fixed queen."*

| World object | Problem object |
|---|---|
| heart / candle at traveler's place | source node, dist=0 |
| heap of sticks | tentative distance dist[v] |
| laying the heap flat as a chessboard | grouping tentative distances into a bounded-width **bucket** (range [i·Δ,(i+1)·Δ)) |
| walking the pawn one square per stick | repeatedly relaxing *light* edges (weight ≤ Δ) out of nodes currently sitting in that bucket, until the bucket stops producing new entries (a fixed point) |
| pawn reaching the far rank → queen, fixed forever | the batch of nodes in that stabilized bucket is finalized (done=1) all at once — no further argument/revision |

Broken assumption: **A** — "each place's distance must be finalized before its neighbors are explored." Here, light edges out of *not‑yet‑finalized* nodes are relaxed repeatedly, possibly several times, before any of those nodes is declared final. Exploration precedes, and even causes revision of, the eventual settling.

**SEED 2** — *"The smallest heap still without a queen is always queened next... only replaced if it beats what is already there."*

| World object | Problem object |
|---|---|
| smallest un-queened heap | min-distance unfinalized node |
| "replaced only if it beats what's there" | relax rule dist[x] = min(dist[x], dist[u]+w) |

This is literally the greedy correctness invariant shared by *every* Dijkstra variant (heap or array). It doesn't itself demand a heap — it only demands *some* way of finding the minimum — so it breaks **B** ("a priority structure must be consulted") only in the degenerate way of licensing a brute-force scan, which is exactly the already-known O(n²) array variant. Not novel.

**SEED 3** — *"A road that dead-ends... leaves its heap forever bare of sticks and pawn, marking a place unreached."*

| World object | Problem object |
|---|---|
| dead-end road, boy in the oak | node with no path from source |
| bare heap, no pawn sent | dist[v] stays +∞, node never enters any bucket/heap |

Breaks **E** — "the whole graph must be explored to know any single distance": the algorithm is licensed to stop once nothing remains pending, touching only the reachable component.

## CHOSEN SEED
SEED 1 — most literal (chessboard = bucket, pawn steps = relaxation rounds, queening = batch finalize) and most different from *both* stated known ways (binary heap, O(n²) scan): it is neither.

## ASSUMPTION BROKEN
**A** — distances get relaxed into neighbors before those neighbors (indeed even the source nodes of the relaxation) are finalized; finalization happens in *batches* only once a bucket reaches a fixed point, "no argument, no revision" from then on.

## ARTIFACT
This is a literal, from-scratch implementation of **Δ-stepping** (Meyer & Sanders): buckets = heaps of sticks reused like a small board, light-edge relaxation = pawn steps within one bucket-board until stable, batch finalize = queening, heavy-edge relaxation (only after queening, since it provably lands in a strictly later bucket) = "look down every road leading out." One improvement pass already folded in: buckets are consumed with a cursor over a growing array (worklist style) instead of malloc/free "snapshot" churn per pass, to avoid unnecessary allocation overhead.

```c
#include <stdlib.h>
#include <math.h>

typedef struct { int *arr; int cnt; int cap; } Bucket;
typedef struct { Bucket *b; int cap; } Buckets;

static void buckets_ensure(Buckets *B, int idx) {
    if (idx < B->cap) return;
    int newcap = B->cap ? B->cap : 16;
    while (idx >= newcap) newcap *= 2;
    B->b = (Bucket*)realloc(B->b, (size_t)newcap * sizeof(Bucket));
    for (int i = B->cap; i < newcap; i++) { B->b[i].arr = NULL; B->b[i].cnt = 0; B->b[i].cap = 0; }
    B->cap = newcap;
}

static void bucket_add(Buckets *B, int idx, int v) {
    buckets_ensure(B, idx);
    Bucket *bk = &B->b[idx];
    if (bk->cnt == bk->cap) {
        bk->cap = bk->cap ? bk->cap * 2 : 4;
        bk->arr = (int*)realloc(bk->arr, (size_t)bk->cap * sizeof(int));
    }
    bk->arr[bk->cnt++] = v;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    int *deg = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fillp = (int*)calloc((size_t)n, sizeof(int));
    double wmax = 0.0, wsum = 0.0;
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fillp[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
        if (weight[i] > wmax) wmax = weight[i];
        wsum += weight[i];
    }
    free(fillp);

    double delta = (m > 0) ? (wsum / (double)m) : 1.0;
    if (!(delta > 0.0)) delta = (wmax > 0.0) ? wmax : 1.0;

    int *lcount = (int*)calloc((size_t)n, sizeof(int));
    for (int u = 0; u < n; u++) {
        int c = 0;
        for (int e = off[u]; e < off[u + 1]; e++) if (ew[e] <= delta) c++;
        lcount[u] = c;
    }
    for (int u = 0; u < n; u++) {
        int lo = off[u], hi = off[u + 1] - 1;
        while (lo < hi) {
            while (lo < hi && ew[lo] <= delta) lo++;
            while (lo < hi && ew[hi] > delta) hi--;
            if (lo < hi) {
                int td = edst[lo]; edst[lo] = edst[hi]; edst[hi] = td;
                double tw = ew[lo]; ew[lo] = ew[hi]; ew[hi] = tw;
            }
        }
    }
    int *lend = (int*)malloc((size_t)n * sizeof(int));
    for (int u = 0; u < n; u++) lend[u] = off[u] + lcount[u];
    free(lcount);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n > 0) dist_out[source] = 0.0;

    int *bucket_of = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) bucket_of[i] = -1;
    char *done = (char*)calloc((size_t)n, 1);
    char *inR = (char*)calloc((size_t)n, 1);
    int *Rlist = (int*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));

    Buckets B; B.b = NULL; B.cap = 0;
    int maxb = 0;
    if (n > 0) {
        bucket_add(&B, 0, source);
        bucket_of[source] = 0;
    }

    for (int i = 0; i <= maxb; i++) {
        if (i >= B.cap || B.b[i].cnt == 0) continue;
        int Rn = 0;
        int cursor = 0;
        for (;;) {
            if (cursor >= B.b[i].cnt) break;
            int v = B.b[i].arr[cursor++];
            if (done[v] || bucket_of[v] != i) continue;
            bucket_of[v] = -2;
            if (!inR[v]) { inR[v] = 1; Rlist[Rn++] = v; }
            double dv = dist_out[v];
            for (int e = off[v]; e < lend[v]; e++) {
                int x = edst[e];
                double nd = dv + ew[e];
                if (nd < dist_out[x]) {
                    dist_out[x] = nd;
                    int nb = (int)(nd / delta);
                    if (nb < 0) nb = 0;
                    if (nb > maxb) maxb = nb;
                    bucket_of[x] = nb;
                    bucket_add(&B, nb, x);
                }
            }
        }
        for (int k = 0; k < Rn; k++) {
            int v = Rlist[k];
            done[v] = 1;
            inR[v] = 0;
            double dv = dist_out[v];
            for (int e = lend[v]; e < off[v + 1]; e++) {
                int x = edst[e];
                double nd = dv + ew[e];
                if (nd < dist_out[x]) {
                    dist_out[x] = nd;
                    int nb = (int)(nd / delta);
                    if (nb < 0) nb = 0;
                    if (nb > maxb) maxb = nb;
                    bucket_of[x] = nb;
                    bucket_add(&B, nb, x);
                }
            }
        }
    }

    for (int i = 0; i < B.cap; i++) free(B.b[i].arr);
    free(B.b);
    free(deg); free(off); free(edst); free(ew); free(lend);
    free(bucket_of); free(done); free(inR); free(Rlist);
}
```

PREDICTION: speedup_vs_heap = 0.5

Reasoning for the number: correctness is guaranteed for arbitrary non-negative real weights (unlike naive Dial's-algorithm bucketing, which silently breaks when two different true distances land in the same bucket), but that safety costs machinery a binary heap doesn't need: an up-front light/heavy edge partition pass, per-node dynamic bucket arrays grown by `realloc`, and pointer chasing through `Buckets`/`Bucket` structs instead of one flat array. Single-threaded, on typical sparse random-weight graphs, I expect this bucket bookkeeping to outweigh the "no log n" win, landing this kernel at roughly **half the speed of (2× slower than)** the reference binary heap — with the caveat that it could close the gap or win on graphs whose weights cluster tightly around the mean Δ (few, dense buckets, few re-relaxations) and would need OpenMP over each bucket's worklist to actually surpass the heap on multi-core hardware, which this first submission does not attempt.

## MEASUREMENT
Not executed — `dijkstra_bench` was not among the tools exposed in this session, so no empirical number was collected here. The prediction above is un-verified and should be treated as a hypothesis for the pipeline to test, not a result.

## VERDICT
Honest status: artifact compiles-by-inspection and is, to the best of my manual trace, correct (Δ-stepping's published proof covers the light/heavy split and batch finalization; I traced the stale-entry and pointer-invalidation cases explicitly since `realloc` can move the bucket array mid-loop). Whether it actually clears the predicted 0.5× bar — or misses it in either direction — is unknown until the pipeline runs `dijkstra_bench`; I am not asserting a measured result I don't have. If the real number comes back near parity or better, the natural next (unused) improvement would be OpenMP-parallelizing the per-bucket relax loops across `Rlist`/light-edge worklist entries, which this literal single-threaded translation deliberately left out to keep the first honest measurement clean.