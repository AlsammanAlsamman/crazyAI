## APPROACH

**Mapping the fable onto the real problem.** The chalkboard with one line per tree is `dist_out[]` — tentative distances, `0` at `source`, `INFINITY` ("blank") elsewhere. The gold star is *settled*; a starred tree's number never changes again. Each round: **walk the board and pick the smallest un-starred number** (a linear scan for the minimum — no heap, no priority structure consulted before a relaxation), **star it**, then **check only that tree's outgoing bridges**, rewriting a neighbour's number when `d[u] + w < d[v]`. Repeat until nothing is left. That is exactly array-scan Dijkstra, and I keep that mechanism — only its *implementation* is sharpened:

1. **Don't walk blank lines.** A tree whose number is still blank (`INFINITY`) can never be the round's minimum unless *every* remaining line is blank, in which case the algorithm is finished (the rest are unreachable). So the scan is kept over a **compacted frontier array** (`act[]` node ids + `actd[]` their distances, SoA), holding exactly the trees that have a number written but no star yet. A node joins the frontier the first time it gets a finite number; it leaves by swap-with-last when starred. `pos[v]` is its slot (`-1` = never written, `-2` = starred). This is provably the same sequence of settle decisions as the full-board walk — it just skips lines that are guaranteed non-minimal.
2. **Vectorize the walk.** The scan is a contiguous `double` argmin, done with AVX2: two accumulators, `vcmppd`/`vminpd`/`vpblendvb` carrying lane indices, ~8 doubles per iteration, scalar tail and a scalar path for tiny frontiers. Ties resolve to the lowest index, so the order is deterministic.
3. **Bridges stored once, contiguously.** CSR built with a counting sort using the `off[u+2]` shift trick (one `calloc`, two passes over `m`, no separate fill array). Each edge is a 16-byte `{double w; int v;}` record so one adjacency step touches one cache line instead of two streams.
4. **Prefetch only when it can pay.** When `dist_out` exceeds ~1 MB the relax loop prefetches `&dist_out[e[j+8].v]`; for small `n` (where the board sits in L1/L2) that path is skipped entirely, since the prefetch would be pure overhead.

No settled-node re-check is needed in the relax branch: with non-negative weights and monotone FP addition, `fl(d[u]+w) >= d[u] >= d[v]` for any starred `v`, so `nd < d[v]` can never fire for a starred node. Correctness is unchanged versus the heap reference; only the "which line is smallest" bookkeeping differs, and both settle in non-decreasing distance order.

Single-threaded on purpose: the rounds are inherently sequential, and a fork/join per round (one per node) would cost more than the few microseconds of work each round contains.

## ARTIFACT

```c
#include <stdlib.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

typedef struct { double w; int v; int pad; } Edge;

/* index of the smallest value in a[0..k), k >= 1; ties -> lowest index */
static inline int frontier_argmin(const double *a, int k)
{
#if defined(__AVX2__)
    if (k >= 16) {
        __m256d v0 = _mm256_loadu_pd(a + 0);
        __m256d v1 = _mm256_loadu_pd(a + 4);
        __m256i b0 = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i b1 = _mm256_setr_epi64x(4, 5, 6, 7);
        __m256i cur = _mm256_setr_epi64x(8, 9, 10, 11);
        const __m256i inc4 = _mm256_set1_epi64x(4);
        int i = 8;
        for (; i + 8 <= k; i += 8) {
            __m256d x0 = _mm256_loadu_pd(a + i);
            __m256d x1 = _mm256_loadu_pd(a + i + 4);
            __m256d c0 = _mm256_cmp_pd(x0, v0, _CMP_LT_OQ);
            __m256i cur2 = _mm256_add_epi64(cur, inc4);
            __m256d c1 = _mm256_cmp_pd(x1, v1, _CMP_LT_OQ);
            v0 = _mm256_min_pd(v0, x0);
            v1 = _mm256_min_pd(v1, x1);
            b0 = _mm256_blendv_epi8(b0, cur,  _mm256_castpd_si256(c0));
            b1 = _mm256_blendv_epi8(b1, cur2, _mm256_castpd_si256(c1));
            cur = _mm256_add_epi64(cur2, inc4);
        }
        {
            __m256d cm = _mm256_cmp_pd(v1, v0, _CMP_LT_OQ);
            __m256d vm = _mm256_min_pd(v0, v1);
            __m256i bm = _mm256_blendv_epi8(b0, b1, _mm256_castpd_si256(cm));
            double vals[4]; long long ids[4];
            double bv; int best;
            _mm256_storeu_pd(vals, vm);
            _mm256_storeu_si256((__m256i *)ids, bm);
            bv = vals[0]; best = (int)ids[0];
            if (vals[1] < bv) { bv = vals[1]; best = (int)ids[1]; }
            if (vals[2] < bv) { bv = vals[2]; best = (int)ids[2]; }
            if (vals[3] < bv) { bv = vals[3]; best = (int)ids[3]; }
            for (; i < k; i++) if (a[i] < bv) { bv = a[i]; best = i; }
            return best;
        }
    }
#endif
    {
        double bv = a[0];
        int best = 0, i;
        for (i = 1; i < k; i++) if (a[i] < bv) { bv = a[i]; best = i; }
        return best;
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    int *off, *pos, *act;
    Edge *E;
    double *actd;
    int i, k, bigmem;

    if (n <= 0) return;
    for (i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- CSR (counting sort, shifted-offset trick) ---- */
    off = (int *)calloc((size_t)n + 2, sizeof(int));
    for (i = 0; i < m; i++) off[src[i] + 2]++;
    for (i = 2; i <= n + 1; i++) off[i] += off[i - 1];
    E = (Edge *)malloc((size_t)m * sizeof(Edge));
    for (i = 0; i < m; i++) {
        int u = src[i];
        int p = off[u + 1]++;
        E[p].v = dst[i];
        E[p].w = weight[i];
    }
    /* now edges of u are E[off[u] .. off[u+1]) */

    /* ---- frontier: trees with a number written, no gold star yet ---- */
    pos  = (int *)malloc((size_t)n * sizeof(int));
    act  = (int *)malloc((size_t)n * sizeof(int));
    actd = (double *)malloc(((size_t)n + 8) * sizeof(double));
    for (i = 0; i < n; i++) pos[i] = -1;

    act[0] = source; actd[0] = 0.0; pos[source] = 0;
    k = 1;
    bigmem = ((double)n * sizeof(double) > 1048576.0);

    while (k > 0) {
        int bi = frontier_argmin(actd, k);
        int u = act[bi];
        double du = actd[bi];
        const Edge *e;
        int deg, j, last;

        last = --k;                      /* remove u from the frontier */
        if (bi != last) {
            int lu = act[last];
            act[bi] = lu;
            actd[bi] = actd[last];
            pos[lu] = bi;
        }
        pos[u] = -2;                     /* gold star: settled forever */

        e = E + off[u];
        deg = off[u + 1] - off[u];

        if (bigmem) {
            for (j = 0; j + 8 < deg; j++) {
                int v = e[j].v;
                double nd = du + e[j].w;
                __builtin_prefetch(&dist_out[e[j + 8].v], 1, 1);
                if (nd < dist_out[v]) {
                    int p = pos[v];
                    dist_out[v] = nd;
                    if (p >= 0) actd[p] = nd;
                    else if (p == -1) { pos[v] = k; act[k] = v; actd[k] = nd; k++; }
                }
            }
            for (; j < deg; j++) {
                int v = e[j].v;
                double nd = du + e[j].w;
                if (nd < dist_out[v]) {
                    int p = pos[v];
                    dist_out[v] = nd;
                    if (p >= 0) actd[p] = nd;
                    else if (p == -1) { pos[v] = k; act[k] = v; actd[k] = nd; k++; }
                }
            }
        } else {
            for (j = 0; j < deg; j++) {
                int v = e[j].v;
                double nd = du + e[j].w;
                if (nd < dist_out[v]) {
                    int p = pos[v];
                    dist_out[v] = nd;
                    if (p >= 0) actd[p] = nd;
                    else if (p == -1) { pos[v] = k; act[k] = v; actd[k] = nd; k++; }
                }
            }
        }
    }

    free(off); free(E); free(pos); free(act); free(actd);
}
```

PREDICTION: 2.5

## MEASUREMENT

Stated plainly: **I could not measure.** This session was invoked with no tools available — `dijkstra_bench` and `dijkstra_contract` were named in the brief but are not callable here, so the "measure, then improve at most four times" loop could not be executed even once. The number above is a pre-measurement prediction of the speedup factor over the binary-heap reference, and it stays unrevised because no data came back to revise it with. Any claim of an observed timing would be fabricated.

What I can assert without a benchmark:
- **Correctness**: settle order is non-decreasing in distance, identical to the heap version; unreachable nodes are never pulled off the frontier and retain `INFINITY`; distances are computed by the same `d[u] + w` additions in the same order per node, so results should match the reference bit-for-bit, not merely within tolerance.
- **Complexity**: `O(m + Σ_rounds |frontier|)`, bounded above by the textbook `O(n² + m)` and typically far below it.

## VERDICT

Faithful to the mandated mechanism — scan for the minimum, settle, then relax — with no heap anywhere. The three implementation wins over a naive full-board walk are: scanning only the *written* lines (compacted frontier, provably the same decisions), doing that scan with AVX2 argmin at ~8 doubles/iteration, and a 16-byte interleaved CSR so each bridge costs one cache line.

Honest risk, and it is the load-bearing one: this family wins on **dense or small** graphs, exactly as the brief's note implies. On a large *random sparse* graph the frontier is Θ(n) through most of the run and total scan work degenerates toward n², which the heap would beat badly — possibly by 10× or more. The frontier compaction rescues graphs with locality (grids, road networks, layered graphs) where the frontier stays small, but it cannot rescue a large `G(n,p)`-style instance. If the benchmark reports a slowdown, that is the cause, and the fix within this mechanism would be bucketing the frontier by distance rather than abandoning the scan for a heap.