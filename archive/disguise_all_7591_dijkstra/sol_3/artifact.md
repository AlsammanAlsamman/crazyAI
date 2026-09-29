## APPROACH

**Mapping the disguised solution onto the real problem.** The "mice" solution is *label‑correcting* SSSP — Bellman–Ford–Moore — as opposed to the *label‑setting* Dijkstra:

| Mouse story | Real implementation |
|---|---|
| No burrow is ever "locked/stamped" | No `done[]` array, no settled set, no finalization step ever happens |
| No one compares every unlocked burrow to pick the next | No priority queue at all — no heap, no array scan for a minimum |
| All mice shout down all their tunnels at once, in waves | One wave = relax all outgoing edges of all currently-active nodes, in parallel (`omp parallel for`) |
| "is that better than what you've got?" → writes it down | `if (du + w < dist[v]) dist[v] = du + w`, done as a lock-free atomic min |
| Wave repeats because a guess can still be beaten later | A node whose value drops becomes active again for the next wave — even a node already processed in this same wave |
| A whole wave with no change ⇒ everything is quietly final | Loop terminates when the wave produces an empty active set; values are the fixed point, never "declared" final |

**How I made the mechanism fast instead of replacing it.** Textbook Bellman–Ford sweeps *all m edges* every wave, `O(n·m)`. Three implementation-level accelerations, none of which changes the mechanism:

1. **Active frontier.** An edge `u→v` can only improve `dist[v]` in wave `k` if `dist[u]` changed in wave `k−1` (its candidate value is otherwise unchanged and already rejected). So each wave scans only the out-edges of nodes whose guess changed — mathematically the identical wave, with provably-useless shouting skipped. Total work drops from `n·m` to (re-activations)·m, typically 2–3·m.
2. **Genuine parallel waves** (the story's literal "all at once"). Distances are held as `uint64_t` bit patterns: all labels are non-negative doubles (0, finite sums of non-negative weights, or `+inf`), and for non-negative doubles IEEE-754 bit order *is* numeric order — so "better than what you've got" becomes a lock-free `uint64` CAS-min with no locks and no floating-point atomics. A per-node claim flag (`__atomic_exchange`) dedups the next frontier; threads batch pushes into thread-local buffers flushed with one `fetch_add`. A stale (too large) read of `dist[u]` is harmless: whoever lowered it re-activates `u`, so the wave structure self-heals.
3. **Adaptive serial path.** Waves with few frontier edges (`< 30000`) run branch-free serial — critical for high-diameter graphs (grids, chains) that produce hundreds of tiny waves where a parallel barrier would cost more than the work.

**Exactness.** IEEE round-to-nearest addition is monotone: `a ≤ b ⇒ fl(a+w) ≤ fl(b+w)` for `w ≥ 0`. Therefore the label-correcting fixed point equals Dijkstra's answer *bit-for-bit*, not merely within tolerance — the greedy prefix-minimal path Dijkstra follows is also the argmin of the wave process. Unreachable nodes never leave `+inf`.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define INF_BITS 0x7FF0000000000000ULL
#define LBUF 512
#define PAR_EDGES 30000
#define PAR_NODES 16

static inline double b2d(uint64_t b) { double d; memcpy(&d, &b, sizeof d); return d; }
static inline uint64_t d2b(double d) { uint64_t b; memcpy(&b, &d, sizeof b); return b; }

/* lock-free "is that better than what you've got?" on non-negative doubles,
   compared as uint64 bit patterns (order-preserving for non-negative IEEE-754). */
static inline int amin_u64(uint64_t *p, uint64_t val)
{
    uint64_t old = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (val < old) {
        if (__atomic_compare_exchange_n(p, &old, val, 0,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return 1;
    }
    return 0;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m < 0) m = 0;

    size_t ms = (size_t)(m > 0 ? m : 1);
    int      *off  = (int *)calloc((size_t)n + 1, sizeof(int));
    int      *cur  = (int *)malloc((size_t)n * sizeof(int));
    int      *edst = (int *)malloc(ms * sizeof(int));
    double   *ew   = (double *)malloc(ms * sizeof(double));
    uint64_t *D    = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    unsigned char *inq = (unsigned char *)calloc((size_t)n, 1);
    int *fr = (int *)malloc((size_t)n * sizeof(int));
    int *nx = (int *)malloc((size_t)n * sizeof(int));

    if (!off || !cur || !edst || !ew || !D || !inq || !fr || !nx) {
        /* degenerate safety: cannot proceed; leave INF except source */
        dist_out[source] = 0.0;
        free(off); free(cur); free(edst); free(ew); free(D); free(inq);
        free(fr); free(nx);
        return;
    }

    /* ---- CSR build: two passes over the edge list, no heap anywhere ---- */
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int u = 0; u < n; u++) off[u + 1] += off[u];
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int p = cur[u]++;
        edst[p] = dst[i];
        ew[p]   = weight[i];
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n > 200000)
#endif
    for (int i = 0; i < n; i++) D[i] = INF_BITS;

    D[source] = 0ULL;                 /* bits of +0.0 */
    fr[0] = source;
    int fsize = 1;

    /* ---- waves: no node is ever finalized, no priority structure ---- */
    while (fsize > 0) {
        long long fe = 0;
        for (int k = 0; k < fsize; k++) {
            int u = fr[k];
            inq[u] = 0;               /* u may legitimately re-activate this wave */
            fe += (long long)(off[u + 1] - off[u]);
        }
        int nsz = 0;

#ifdef _OPENMP
        if (fe >= PAR_EDGES && fsize >= PAR_NODES) {
            #pragma omp parallel
            {
                int lbuf[LBUF];
                int lc = 0;
                #pragma omp for schedule(guided) nowait
                for (int k = 0; k < fsize; k++) {
                    int u = fr[k];
                    double du = b2d(__atomic_load_n(&D[u], __ATOMIC_RELAXED));
                    int e0 = off[u], e1 = off[u + 1];
                    for (int e = e0; e < e1; e++) {
                        int v = edst[e];
                        uint64_t nb = d2b(du + ew[e]);
                        if (nb < __atomic_load_n(&D[v], __ATOMIC_RELAXED)) {
                            if (amin_u64(&D[v], nb)) {
                                if (__atomic_exchange_n(&inq[v], (unsigned char)1,
                                                        __ATOMIC_RELAXED) == 0) {
                                    lbuf[lc++] = v;
                                    if (lc == LBUF) {
                                        int base = __atomic_fetch_add(&nsz, lc,
                                                        __ATOMIC_RELAXED);
                                        memcpy(nx + base, lbuf,
                                               (size_t)lc * sizeof(int));
                                        lc = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                if (lc) {
                    int base = __atomic_fetch_add(&nsz, lc, __ATOMIC_RELAXED);
                    memcpy(nx + base, lbuf, (size_t)lc * sizeof(int));
                }
            }
        } else
#endif
        {
            for (int k = 0; k < fsize; k++) {
                int u = fr[k];
                double du = b2d(D[u]);
                int e0 = off[u], e1 = off[u + 1];
                for (int e = e0; e < e1; e++) {
                    int v = edst[e];
                    uint64_t nb = d2b(du + ew[e]);
                    if (nb < D[v]) {
                        D[v] = nb;
                        if (!inq[v]) { inq[v] = 1; nx[nsz++] = v; }
                    }
                }
            }
        }

        int *t = fr; fr = nx; nx = t;
        fsize = nsz;
    }

    memcpy(dist_out, D, (size_t)n * sizeof(double));   /* bits are valid doubles */

    free(off); free(cur); free(edst); free(ew); free(D); free(inq);
    free(fr); free(nx);
}
```

PREDICTION: 3.0

(Interpreting the number as median speedup over the reference binary-heap Dijkstra: I expect ~2–4× serial from eliminating all `O(m log n)` heap traffic, and more on large sparse graphs where the parallel wave path engages; well under 1.5× on tiny or very high-diameter graphs where wave count dominates.)

## MEASUREMENT

**Not measured.** This session was invoked with no tools available — `dijkstra_bench` and `dijkstra_contract` were listed but are not callable here, and there is no local harness for the fixed contract. So I have *zero* measured numbers, and the four allowed improvement rounds did not happen. Reporting that plainly rather than inventing a table.

What I did verify by reasoning only, and what remains open:

- **Contract**: signature, argument order, `dist_out` semantics (`INFINITY` for unreachable, `0` at source) match exactly; only `dist_out` is written; all scratch freed.
- **Exactness**: argued bit-identical to heap Dijkstra via monotonicity of round-to-nearest addition with non-negative weights.
- **Termination**: every re-activation requires a *strict* decrease of some label; the label set is finite, so zero-weight cycles and duplicate/self edges cannot loop forever.
- **Unvalidated risk 1**: the `PAR_EDGES = 30000` / `PAR_NODES = 16` thresholds are reasoned, not tuned. On a 1000×1000 grid this deliberately keeps every wave serial; if the bench's graphs sit near the boundary this is the first knob I would sweep.
- **Unvalidated risk 2**: total re-activation factor. If the bench includes an adversarial high-diameter graph, label-correcting work can far exceed `2–3·m` and this loses to the heap. The in-mechanism fix would be distance-bucketed frontier ordering (still no finalization, still label-correcting), which I did not add because I cannot measure whether it pays.

## VERDICT

Artifact is complete, contract-conformant, and faithful to the disguised mechanism — repeated all-at-once relaxation waves with no node ever finalized and no priority structure consulted — accelerated by active-frontier pruning, lock-free `uint64` atomic-min parallel waves, and an adaptive serial path. **The performance claim is unverified: prediction 3.0× stands as a prediction only, with no measurement behind it.** Until it is benched, treat the speedup as an argument, not a result; the correctness argument is the part I am confident in.