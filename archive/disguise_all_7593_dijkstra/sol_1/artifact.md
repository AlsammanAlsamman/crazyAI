## APPROACH

**Faithfulness to Pip's mechanism.** The disguised solution is *linear-scan (O(n²)) Dijkstra with no heap at all*: one chalkboard of tentative distances, and each round a straight sweep over the board to find the smallest unlocked entry, lock it, relax its out-edges, repeat. I keep exactly that mechanism — there is no priority queue, no heap, no bucket structure anywhere in the artifact. The only thing I change is *how the sweep is executed*, not *what it computes*.

Mapping, piece by piece:

| Pip | Implementation |
|---|---|
| chalkboard line per tree | `cd[]` (tentative distance) + `ci[]` (which tree that line is about) |
| "giant question mark" | node never touched → it has **no line on the board yet** (`pos[v] == -1`) |
| "locked in", question mark crossed out | `pos[u] == -2`, final value written to `dist_out[u]` |
| walk the *entire* board to find the smallest | `scan_min()`: one contiguous SIMD sweep of `cd[0..k)`, no ordering maintained |
| walk the bridges out of the locked tree | CSR adjacency sweep `off[u]..off[u+1]` |
| "update the guess written for the tree across the bridge" | overwrite `cd[pos[v]]` in place — O(1) decrease-key, which is exactly what a chalkboard *is* (erase and rewrite one line), and precisely what a heap cannot do cheaply |
| repeat until every tree is locked | loop until the board is empty |

Two board-keeping refinements, both semantics-preserving rather than mechanism-changing:

1. **Question-marked trees aren't written on the board at all.** A line reading ∞ can never be the strict minimum while any finite line exists, and if *every* remaining line is ∞ those trees are unreachable and keep `INFINITY`. So I only materialise a line when a tree first gets a finite guess. For sparse graphs this shrinks the sweep from `n` to the live frontier; for dense graphs it degenerates back to the honest O(n²) and costs nothing.
2. **Locked lines are erased by swap-with-last**, keeping the live lines contiguous — so the sweep is a dense, unit-stride `double` stream with zero pointer chasing and zero gathers.

**Why this should beat the binary heap.** The heap reference pays `log n` random-access swaps *per relaxation* and pushes up to `m` stale entries (lazy deletion), so its working set is O(m) of 16-byte items touched in a cache-hostile pattern. The scan version does 4–8 `minpd` lanes per cycle over a perfectly sequential array and relaxes in O(1). The sweep is also branchless: `_mm256_min_pd` reductions, then one `_mm256_cmp_pd`/`movemask` pass to recover the index (early-exit, ~half the array on average). Ties resolve to the lowest index, which is fine — any minimum-distance node is a valid next lock-in.

The remaining exposure is the genuine one: a very large *sparse* graph whose frontier stays O(n) makes the sum-of-sweeps quadratic and the heap wins. That's inherent to the mechanism I was asked to implement, and I'd rather report it than swap in a heap.

Other details: CSR is built with a counting sort in two passes over `m`, reusing `pos` as the histogram and `ci` as the fill cursor (no extra allocations); edges are packed as a 16-byte `{double w; int v;}` record so each adjacency walk is a *single* memory stream rather than two.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <immintrin.h>
#endif

/* 16-byte edge record: one memory stream per adjacency walk. */
typedef struct { double w; int v; int pad; } Edge;

/* Pip's full sweep of the board: return index of the smallest entry in a[0..k). */
static inline int scan_min(const double *a, int k)
{
    int b;
    double bv;

    if (k < 8) {
        b = 0; bv = a[0];
        for (int i = 1; i < k; i++) if (a[i] < bv) { bv = a[i]; b = i; }
        return b;
    }

#if defined(__AVX2__)
    {
        __m256d m0 = _mm256_loadu_pd(a);
        __m256d m1 = _mm256_loadu_pd(a + 4);
        int i = 8;
        for (; i + 16 <= k; i += 16) {
            __m256d x0 = _mm256_loadu_pd(a + i);
            __m256d x1 = _mm256_loadu_pd(a + i + 4);
            __m256d x2 = _mm256_loadu_pd(a + i + 8);
            __m256d x3 = _mm256_loadu_pd(a + i + 12);
            m0 = _mm256_min_pd(m0, x0);
            m1 = _mm256_min_pd(m1, x1);
            m0 = _mm256_min_pd(m0, x2);
            m1 = _mm256_min_pd(m1, x3);
        }
        for (; i + 8 <= k; i += 8) {
            m0 = _mm256_min_pd(m0, _mm256_loadu_pd(a + i));
            m1 = _mm256_min_pd(m1, _mm256_loadu_pd(a + i + 4));
        }
        {
            __m256d mv = _mm256_min_pd(m0, m1);
            __m128d lo = _mm256_castpd256_pd128(mv);
            __m128d hi = _mm256_extractf128_pd(mv, 1);
            __m128d mm = _mm_min_pd(lo, hi);
            mm = _mm_min_sd(mm, _mm_unpackhi_pd(mm, mm));
            bv = _mm_cvtsd_f64(mm);
        }
        for (; i < k; i++) if (a[i] < bv) bv = a[i];

        /* locate the line holding that value */
        {
            __m256d bvv = _mm256_set1_pd(bv);
            int j = 0;
            for (; j + 4 <= k; j += 4) {
                __m256d x = _mm256_loadu_pd(a + j);
                int msk = _mm256_movemask_pd(_mm256_cmp_pd(x, bvv, _CMP_EQ_OQ));
                if (msk) return j + __builtin_ctz((unsigned)msk);
            }
            for (; j < k; j++) if (a[j] == bv) return j;
        }
        return 0;
    }
#else
    b = 0; bv = a[0];
    for (int i = 1; i < k; i++) if (a[i] < bv) { bv = a[i]; b = i; }
    return b;
#endif
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m < 0) m = 0;

    int    *off = (int *)   malloc((size_t)(n + 1) * sizeof(int));
    int    *pos = (int *)   malloc((size_t)n * sizeof(int));   /* board slot / -1 blank / -2 locked */
    int    *ci  = (int *)   malloc((size_t)n * sizeof(int));   /* which tree each board line is */
    double *cd  = (double *)malloc((size_t)n * sizeof(double));/* the guesses, contiguous */
    Edge   *E   = (Edge *)  malloc((size_t)(m > 0 ? m : 1) * sizeof(Edge));

    if (!off || !pos || !ci || !cd || !E) {
        free(off); free(pos); free(ci); free(cd); free(E);
        return;
    }

    /* ---- CSR by counting sort; `pos` doubles as histogram, `ci` as fill cursor ---- */
    memset(pos, 0, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) pos[src[i]]++;
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + pos[i];
    memcpy(ci, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int p = ci[u]++;
        E[p].w = weight[i];
        E[p].v = dst[i];
    }
    memset(pos, 0xFF, (size_t)n * sizeof(int));   /* every tree: blank line (-1) */

    /* ---- the board starts with one line: the Big Oak at zero ---- */
    int k = 1;
    cd[0] = 0.0; ci[0] = source; pos[source] = 0;

    while (k > 0) {
        /* full sweep of every live line, smallest wins */
        int j = scan_min(cd, k);
        int u = ci[j];
        double du = cd[j];

        /* lock it in and erase its line (swap-with-last keeps lines contiguous) */
        k--;
        cd[j] = cd[k];
        ci[j] = ci[k];
        pos[ci[j]] = j;
        pos[u] = -2;
        dist_out[u] = du;

        /* walk the bridges out of it */
        {
            int e   = off[u];
            int end = off[u + 1];
            for (; e < end; e++) {
                int v = E[e].v;
                int p = pos[v];
                if (p == -2) continue;                 /* already locked */
                double nd = du + E[e].w;
                if (p >= 0) {
                    if (nd < cd[p]) cd[p] = nd;        /* erase + rewrite one line */
                } else {
                    pos[v] = k; cd[k] = nd; ci[k] = v; k++;   /* first real guess */
                }
            }
        }
    }

    free(off); free(pos); free(ci); free(cd); free(E);
}
```

## PREDICTION

PREDICTION: 2.5

(Stated before any measurement: I expect ~2.5× the binary-heap reference on the benchmark's graph mix, driven by O(1) decrease-key plus a vectorised unit-stride min sweep replacing O(log n) random-access sift operations per relaxation.)

## MEASUREMENT

**Not measured.** No tools were available in this session — `dijkstra_bench` and `dijkstra_contract` could not be invoked, so the prediction above stands unverified and the "improve it at most four times" loop never ran. I am reporting that plainly rather than inventing numbers: there is no measured speedup, no correctness run against the reference, and no tolerance check behind this artifact.

What I can assert without the harness is contract conformance by construction: signature and argument order copied verbatim; `dist_out` is written for all `n` nodes (`INFINITY` for unreachable, since a node with no finite guess never gets a board line and is never overwritten); `dist_out[source] = 0` is written on the first lock-in; all scratch memory is freed on every exit path including allocation failure; `n <= 0`, `m == 0`, self-loops, parallel edges, zero-weight edges, and out-of-range `source` are all handled. Arithmetic is bit-identical to the reference's: same `du + w` accumulation order along whatever shortest path Dijkstra settles, so agreement should be exact, not merely within tolerance, except where distinct equal-cost paths exist — and there the reference's own tie-break is equally arbitrary.

## VERDICT

The mechanism was translated faithfully — heapless linear-scan Dijkstra, one sweep of the live board per lock-in — and implemented about as tightly as that mechanism allows: AVX2 min-reduction over a contiguous, swap-compacted distance array, O(1) decrease-key by in-place rewrite, single-stream 16-byte CSR edges, five allocations with the histogram and fill-cursor arrays aliased onto working storage.

Honest assessment of where it stands: I believe it beats the binary heap on dense and small-to-medium graphs, which is the well-known result the problem statement itself flags. I am *not* confident it beats the heap on a large sparse graph with a persistently wide frontier — there the sum of sweeps goes quadratic and the heap's `log n` wins. That risk is intrinsic to the requested mechanism, not to this implementation of it, and the frontier-only sweep is the strongest mitigation available without abandoning the mechanism.

Bottom line: artifact complete and, I believe, correct; prediction of 2.5× recorded before measurement and left untested. Treat the speedup claim as unsubstantiated until the benchmark runs.