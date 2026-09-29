## APPROACH

**Mapping the disguised solution onto the real problem.** Gus's bureaucracy is exactly the set of assumptions the standard solution smuggles in, and Mimi's rule is a *label-correcting* (Bellman–Ford–Moore / SPFA) relaxation loop:

| Forest | Implementation |
|---|---|
| Hollow trees, rope bridges with painted times | `n` nodes, CSR-compressed out-edge lists built once from `(src,dst,weight)` |
| "Crossing a bridge never makes you arrive earlier" | `weight[i] >= 0` ⟹ `dist` is monotone non-increasing per node, so a corrected label never needs to be revisited upward, and a self-cycle can never satisfy `nd < dist[u]` (this is what makes the cached `du` safe) |
| No "settled" stamp, no gold star | **no `done[]`/visited array at all, no finalization step** |
| "The moment you hear a better time, run across every bridge leaving that tree" | on `dist[v]` improvement, `v` is put on a work list; popping `v` scans all of `v`'s out-edges |
| "A report that doesn't beat the current number → that scout stops" | `if (nd < dist[v])` is the *only* gate; failing it propagates nothing (this is the termination argument) |
| No council roll-call over every unsettled tree | no priority queue and no O(n) scan — the work list is an O(1) circular deque; a node is enqueued at most once at a time via an `inq` flag |
| "Eventually no scout has anything new to say" | loop ends when the deque is empty; every label is then final |

**Making this mechanism fast rather than swapping it out:**
1. **Deque with SLF (small-label-first)**, not plain FIFO. A newly improved node whose label beats the label at the front goes to the *front*, otherwise the back. This is still pure gossip — no finalization, no key comparisons against a heap, O(1) per push — but it cuts the number of times a node is re-relaxed by a large factor, which is where naive SPFA loses.
2. **One arena allocation** for offsets, deque, edges and flags; the deque buffer doubles as the CSR scatter cursor, so no separate `deg`/`fill` arrays (the baseline allocates seven blocks).
3. **16-byte `{double w; int v;}` edge records**, so an adjacency scan is a single sequential stream (one prefetch stream, one TLB walk) instead of two.
4. **Software prefetch** of `dist[E[e+4].v]` on long adjacency lists — the random `dist[v]` load is the only cache miss left in the inner loop.
5. **`restrict` locals** so the compiler knows the edge array and `dist_out` don't alias and can keep `du` in a register.
6. **Pathology guard (honest disclosure):** label-correcting relaxation is O(nm) on adversarial layered graphs. A per-pop work counter (cost: one add + one branch per *node*, not per edge) aborts if scanned edges exceed `32*(m+n)` — ~10–20× the normal budget — and reruns heap Dijkstra so a contrived input can't turn into a timeout. It should never fire on random or structured-but-sane graphs.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { double w; int v; int pad; } Edge;

/* Safety valve only: bounded-work fallback for adversarial graphs. */
static void dijkstra_fallback(int n, int m, const int *off, const Edge *E,
                              int source, double *dist, unsigned char *done)
{
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    memset(done, 0, (size_t)n);
    Edge *h = (Edge *)malloc(((size_t)m + 2) * sizeof(Edge));
    if (!h) return;
    int hs = 1;
    h[0].w = 0.0; h[0].v = source;
    while (hs > 0) {
        int u = h[0].v;
        hs--;
        h[0] = h[hs];
        int i = 0;
        for (;;) {
            int l = 2 * i + 1;
            if (l >= hs) break;
            int s = l;
            if (l + 1 < hs && h[l + 1].w < h[l].w) s = l + 1;
            if (!(h[s].w < h[i].w)) break;
            Edge t = h[s]; h[s] = h[i]; h[i] = t;
            i = s;
        }
        if (done[u]) continue;
        done[u] = 1;
        double du = dist[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = E[e].v;
            double nd = du + E[e].w;
            if (nd < dist[v]) {
                dist[v] = nd;
                int j = hs++;
                h[j].w = nd; h[j].v = v;
                while (j > 0) {
                    int p = (j - 1) >> 1;
                    if (h[p].w <= h[j].w) break;
                    Edge t = h[p]; h[p] = h[j]; h[j] = t;
                    j = p;
                }
            }
        }
    }
    free(h);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    const size_t nn = (size_t)n;
    const size_t A  = 64;
    size_t b_off = (((nn + 2) * sizeof(int) + A - 1) / A) * A;
    size_t b_dq  = (((nn + 2) * sizeof(int) + A - 1) / A) * A;
    size_t b_ed  = (((size_t)m * sizeof(Edge) + A - 1) / A) * A;
    size_t b_inq = (((nn + A) + A - 1) / A) * A;

    char *arena = (char *)malloc(b_off + b_dq + b_ed + b_inq + A);
    if (!arena) return;
    char *p = arena;
    size_t mis = (size_t)p & (A - 1);
    if (mis) p += A - mis;

    int  *off = (int *)p;
    int  *dq  = (int *)(p + b_off);                       /* CSR cursor, then deque */
    Edge *E   = (Edge *)(p + b_off + b_dq);
    unsigned char *inq = (unsigned char *)(p + b_off + b_dq + b_ed);

    /* ---- CSR build: count, prefix, scatter (cursor aliases the deque buffer) ---- */
    memset(off, 0, (nn + 2) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(dq, off, nn * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int q = dq[u]++;
        E[q].v = dst[i];
        E[q].w = weight[i];
    }
    memset(inq, 0, nn);

    /* ---- pure label-correcting gossip: no "settled" set, no priority queue ---- */
    double     * restrict D  = dist_out;
    const Edge * restrict EA = E;

    const int cap = n + 1;
    int head = 0, tail = 0;
    dq[tail++] = source;
    inq[source] = 1;

    const long long budget = 32LL * ((long long)m + (long long)n) + 4096LL;
    long long work = 0;
    int overflow = 0;

    while (head != tail) {
        int u = dq[head];
        head++; if (head == cap) head = 0;
        inq[u] = 0;

        const double du = D[u];
        int e = off[u], ee = off[u + 1];

        work += (long long)(ee - e) + 1;
        if (work > budget) { overflow = 1; break; }

        const int elim = ee - 4;
        for (; e < ee; e++) {
            if (e < elim) __builtin_prefetch(&D[EA[e + 4].v], 1, 1);
            int v = EA[e].v;
            double nd = du + EA[e].w;
            if (nd < D[v]) {                 /* the one and only gate */
                D[v] = nd;
                if (!inq[v]) {
                    inq[v] = 1;
                    /* small-label-first: better-than-front goes to the front */
                    if (head != tail && nd < D[dq[head]]) {
                        head = (head == 0) ? cap - 1 : head - 1;
                        dq[head] = v;
                    } else {
                        dq[tail] = v;
                        tail++; if (tail == cap) tail = 0;
                    }
                }
            }
        }
    }

    if (overflow) dijkstra_fallback(n, m, off, E, source, dist_out, inq);

    free(arena);
}
```

Correctness notes: the deque holds exactly the nodes with `inq==1`, at most `n`, so capacity `n+1` never overflows and the front slot needed by a push-front is always free. `du` is safe to cache because a path leaving `u` and returning cannot satisfy `nd < D[u]` under non-negative weights (zero-weight cycles give `nd == du`, which fails the strict test — no infinite churn). Per-path floating-point accumulation order is identical to Dijkstra's (left-to-right prefix sums), and relaxing from a smaller prefix is monotone in the rounded result, so labels agree with the reference to well within tolerance.

## PREDICTION

PREDICTION: 3.0

(i.e. ≈3.0× faster than the given binary-heap Dijkstra baseline on the benchmark's graphs; I expect 2–4× on random graphs with average degree ≲32, closer to 1.5× for very dense small graphs where the baseline's heap barely matters and CSR construction dominates.)

## MEASUREMENT

**Not measured — I could not run the benchmark.** This session was invoked with no tools available (`dijkstra_bench` and `dijkstra_contract` are listed but not callable here), so the "measure, then improve at most four times" loop did not happen: I got one shot and spent it on the variant with the best expected-value-under-uncertainty, not the most aggressive one. Stating that plainly rather than dressing up an estimate as a result:

- Measured speedup: **none obtained**.
- Contract check: **not executed**; verified only by inspection (signature, argument order, `INFINITY` for unreachable, `dist_out[source] = 0`, no writes outside `dist_out[0..n-1]`, all scratch freed on every exit path except the `malloc`-failure early return, which frees nothing because nothing was allocated).
- Consequences of the missing measurement: I deliberately did **not** add the OpenMP frontier variant (parallel Bellman–Ford rounds with CAS-based atomic min on the `uint64` bit patterns of non-negative doubles — valid because IEEE ordering matches numeric ordering for non-negative values including `+INF`). It is the natural next step for `m ≳ 10^7` and would plausibly add 3–6× on top, but unmeasured it is a coin flip: it loses the SLF ordering advantage and degrades on high-diameter graphs, and I will not ship an unverified parallel rewrite of a hot loop.
- If the pipeline measures this, the numbers worth reporting are: total time vs. baseline, and edges-scanned/`m` (the ratio that says whether SLF is doing its job — expect 1.2–3; if it is >8 the deque discipline, not the mechanism, is what needs work, and if the fallback fires at all the graph is adversarial and that fact is the finding).

## VERDICT

The mechanism translates cleanly and the disguised solution's central claim is true: the "settle first, then announce" ceremony is not needed for correctness — non-negative weights alone guarantee monotone labels, so `if (nd < dist[v])` is a sufficient stopping rule and the entire `done[]` set, the priority structure, and the O(n) scan for the next node to finalize all disappear. What replaces them is a work list, and the *only* thing that decides whether this beats heap Dijkstra is how many times a node gets re-relaxed; that is why SLF, not the removal of the heap, is the real optimization here.

Honest status: **a well-founded implementation with an unverified performance claim.** I believe it is faster than the baseline and I have stated 3.0× before any measurement, as required, but I have no evidence and the prediction may be wrong — most plausibly on the low side for dense graphs (baseline heap cost shrinks relative to the shared O(n+m) build) and on the high side if the harness feeds a layered graph that makes label-correcting thrash, in which case the guard fires and the result is *slower* than baseline by the wasted-work factor plus one full Dijkstra.