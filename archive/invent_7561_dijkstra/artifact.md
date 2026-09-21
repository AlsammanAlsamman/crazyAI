# MAPPING

| World object | Problem object | Silent assumption it touches |
|---|---|---|
| **SEED 1** — standing‑stone hub / settled stone; road; chant sent down every road at once; bird carrying the road's length as its cry | settled stone = a node `u` just marked finalized; road = outgoing edge `(u,v,w)`; chant = a relaxation attempt; bird's cry = candidate distance `dist[u]+w` arriving at `v` | This is *exactly* standard Dijkstra edge‑relaxation, done only *after* the stone (node) is settled. It leaves assumption 1 ("finalize before exploring neighbors") and assumption 4 ("road only considered once source settled") **intact**. Least novel of the three. |
| **SEED 2** — traveler claims the shortest‑cried unclaimed bird among *every* bird now perched, nails its name to the calendar, closing that place forever | "every bird now perched and unclaimed" = the whole field of not‑yet‑settled nodes that currently hold a finite tentative distance; "shortest‑cried" = minimum tentative distance; "nailing to the calendar, closing forever" = `done[u]=1`, `dist_out[u]` fixed | The traveler finds the minimum by **sweeping the whole field**, not by consulting an indexed structure. This breaks assumption 2 ("a priority structure must be consulted before every relaxation") — there is no heap at all, relaxation is a bare array write, and the "next to finalize" is found by one linear scan per round. |
| **SEED 3** — roads into the never‑visited maze send no bird back; blank calendar day; thread thrown away | unreachable component; `dist_out[v]=INFINITY`; no retry, no wasted bookkeeping for that node | Same as the reference's `INFINITY` initialization — pure correctness bookkeeping already present in the known way. Breaks none of the five assumptions. |

# CHOSEN SEED

**None of the three seeds breaks assumption 1** ("each place's distance must be finalized before its neighbors are explored") — every chant in this whole account is sent *from the settled stone*, i.e., strictly after settling, in all three seeds. Stated plainly per the instructions, and falling back to the most literal seed: **SEED 2**, because it is the central, load‑bearing mechanism of the whole account (the selection/closing step), and it is the one whose literal reading ("every bird now perched," i.e., the *whole field*, scanned each round) diverges furthest from the heap example given.

# ASSUMPTION BROKEN

Assumption 2: *"a priority structure must be consulted before every relaxation."* Read literally, SEED 2's traveler keeps no indexed structure — relaxation (SEED 1) is a bare write, and "next to finalize" comes from a plain O(n) sweep of the whole field once per round. This is not a novel invention: it is precisely the well‑known, validated **O(n²) array‑scan Dijkstra**, which the `known_way` text itself names as "a real, well‑known practical win" for dense/small graphs — so per step 4 the mechanism is made to *arrive at* that established technique rather than something untested.

The `known_way` text names **two regimes** (sparse/large → heap; dense/small → array scan), so per step 5 the kernel must recognize which regime it is in and fall back. I encode that with a runtime check comparing the O(n²) sweep cost against the O((n+m) log n) heap cost, falling back to the **exact reference heap algorithm** (unchanged, so no regression) whenever the graph is sparse/large. This directly addresses the risk named in `known_way` ("O(n²) is bad for large sparse graphs") rather than shipping it unguarded.

# ARTIFACT

```c
#include <stdlib.h>
#include <math.h>

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    /* Build CSR adjacency once; shared by both regimes. */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    {
        int *fill = calloc((size_t)n, sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = off[u] + fill[u]++;
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
        free(fill);
    }
    free(deg);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    /* Regime check mirroring the known_way's two named regimes:
       dense/small  -> "of every bird now perched and unclaimed I take the
                        one whose chant is shortest": sweep the whole field
                        each round, no priority structure at all (O(n^2)).
       sparse/large -> fall back to the validated binary-heap Dijkstra
                        (identical to the reference), guarding the risk
                        that an O(n^2) sweep is bad when the graph is big
                        and sparse. */
    double logn = log((double)n + 2.0) / log(2.0);
    double heap_cost = 3.0 * ((double)n + (double)m) * logn;
    int use_array_scan = ((double)n * (double)n) <= heap_cost;

    char *done = calloc((size_t)n, 1);

    if (use_array_scan) {
        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            double best = INFINITY;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist_out[v] < best) { best = dist_out[v]; u = v; }
            }
            if (u < 0) break; /* rest of the field is the never-visited maze: leave blank, throw the thread away */
            done[u] = 1;
            double du = dist_out[u];
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = du + ew[e];
                if (nd < dist_out[v]) dist_out[v] = nd;
            }
        }
    } else {
        typedef struct { double d; int u; } HeapItem;
        HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
        int hs = 0;
        {
            int i = hs++; heap[i].d = 0.0; heap[i].u = source;
            while (i > 0) {
                int p = (i - 1) / 2;
                if (heap[p].d <= heap[i].d) break;
                HeapItem t = heap[p]; heap[p] = heap[i]; heap[i] = t; i = p;
            }
        }
        while (hs > 0) {
            HeapItem top = heap[0]; hs--; heap[0] = heap[hs];
            {
                int i = 0;
                while (1) {
                    int l = 2 * i + 1, r = 2 * i + 2, s = i;
                    if (l < hs && heap[l].d < heap[s].d) s = l;
                    if (r < hs && heap[r].d < heap[s].d) s = r;
                    if (s == i) break;
                    HeapItem t = heap[s]; heap[s] = heap[i]; heap[i] = t; i = s;
                }
            }
            int u = top.u;
            if (done[u]) continue;
            done[u] = 1;
            for (int e = off[u]; e < off[u + 1]; e++) {
                int v = edst[e];
                double nd = dist_out[u] + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    int i = hs++; heap[i].d = nd; heap[i].u = v;
                    while (i > 0) {
                        int p = (i - 1) / 2;
                        if (heap[p].d <= heap[i].d) break;
                        HeapItem t = heap[p]; heap[p] = heap[i]; heap[i] = t; i = p;
                    }
                }
            }
        }
        free(heap);
    }

    free(done);
    free(off); free(edst); free(ew);
}
```

PREDICTION: speedup_vs_heap = 1.3

# MEASUREMENT

No tools were available in this session (explicitly stated: "No tools are available in this session... the pipeline will compile and measure it"), so `dijkstra_bench` could not actually be run here. Reporting this plainly rather than fabricating numbers: **no empirical measurement was performed by me.** My prediction above is reasoning-only, based on: (a) for sparse/large graphs the kernel takes the `else` branch and is byte‑for‑byte the same algorithm as the reference heap Dijkstra, so speedup ≈ 1.0 there; (b) for graphs the runtime check classifies as dense/small, the O(n²) sweep removes all heap bookkeeping (allocation, sift‑up/down, pointer chasing) in favor of a branch‑light linear scan that auto‑vectorizes well under `-O3 -march=native`, which typically yields a real 1.5–3× win in that regime; 1.3 reflects a blended, conservative guess without knowing the harness's default graph shape.

# VERDICT

The metaphor did not, on honest reading, break the "settle before exploring neighbors" assumption — no seed here abandons Dijkstra's greedy order. What SEED 2 legitimately breaks is the priority‑structure assumption, and it does so by converging on a real, textbook technique (O(n²) array‑scan Dijkstra) rather than inventing something new, exactly as step 4 requires. Because the `known_way` text itself names the risk ("dense or small graphs" as the regime where this wins, implying it's a loss otherwise), I did not ship it unguarded: the kernel checks `n²` against a scaled `(n+m) log n` estimate at runtime and falls back to the untouched, validated heap path for the sparse/large regime, so the known way's own risk is addressed rather than ignored. No thread‑level parallelism was added — per‑node out‑degree is far too small a unit of work at realistic benchmark sizes to amortize thread launch, and the one loop worth accelerating (the linear min‑scan) is already vectorization‑friendly by construction, consistent with preferring vectorization hints over threading. The honest gap in this answer is the untested threshold constant (`3.0`) in the regime check and the complete absence of an actual `dijkstra_bench` run — both should be the first thing corrected once tool access is restored, before trusting the 1.3 prediction.