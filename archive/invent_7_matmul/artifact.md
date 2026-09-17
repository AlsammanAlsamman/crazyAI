# MAPPING

**SEED 1 — "Bellows pump seal-stamped mud through carved doors to carry each number to its meeting-place."**

| World object | Problem object |
|---|---|
| First great table | Matrix A |
| Wall of corridors (rows) | Rows of A, one corridor per row *i* |
| Second table, crosswise wall | Matrix B, columns as corridors *j* |
| Door carved with a number, position *k* down the corridor | Memory cell A[i,k] (row corridor) or B[k,j] (column corridor) |
| Meeting-place | Accumulator cell C[i,j] |
| Bellow (one per shared index) | A single pass over one value of the contraction index k |
| Mud carrying a stamped seal | A value copied out of memory and moved toward C[i,j] (a broadcast/load) |

Breaks assumption **#3** ("a matrix is a 2-D grid living in one memory") — data is treated as something that *flows* to its consumer rather than something passively indexed in place.

**SEED 2 — "Two seals arriving together at one vase are multiplied into a single clot before they drop."**

| World object | Problem object |
|---|---|
| Seal | Scalar A[i,k] or B[k,j] |
| Clot | Product A[i,k]·B[k,j] |
| Dropping the clot into the vase | C[i,j] += product |

Breaks assumption **#8** ("one product is one problem; many products are many problems") — the multiply is inseparable from the accumulation event; this is just fused multiply-add, i.e. what every decent microkernel (including OpenBLAS) already does.

**SEED 3 — "A vase is only read after its full barrage of indices has fired, never before it is whole."**

| World object | Problem object |
|---|---|
| Barrage of indices for one vase | The n terms of the k-sum for one C[i,j] |
| A bellow firing once | One worker completing its assigned slice of k |
| "Read early lies" | Using C[i,j] before all k-contributions are summed is wrong |
| Vase tipped, wiped, reused for next crossing | Same physical accumulator memory reused across many (i,j) |

Breaks assumption **#2** ("the whole sum over the shared index is finished before the next cell is started") — here the *index* is split across many simultaneous bellows (processors), each doing only a slice of the sum for potentially every cell at once; correctness comes from a barrier ("only once whole"), not from sequential in-order completion. This also breaks **#6** ("one processor holds both matrices") since several bellows act on the full extent of both tables concurrently.

# CHOSEN SEED

Seed 3. It is the most literal (it directly names *when* a cell may legally be read, which is the exact synchronization contract a parallel reduction needs) and the most different from the known way: OpenBLAS parallelizes over **output tiles** (i,j blocks), never over the **contraction index k** with private replicate-then-reduce accumulators. Seeds 1–2 just redescribe an ordinary FMA microkernel, which is already what the known way does.

# ASSUMPTION BROKEN

\#2 — "the whole sum over the shared index is finished before the next cell is started." Instead: split the shared index k into contiguous chunks, one per bellow/thread; each bellow computes its partial rank-1-update contribution to *every* cell (i,j) into its own private "vase set" (a full n×n scratch buffer); a barrier ensures every bellow has fired; only then are the vases read — summed (reduced) into the final C.

# ARTIFACT

Mapping of literal objects → computational objects:
- **Memory that stays still**: A and B (read-only, row-major, never moved — "doors and corridors never move").
- **What flows**: values of A[i,k], B[k,j] and their products — register/cache-line traffic ("only the mud moves").
- **The vases**: T private n×n accumulator buffers, one per bellow/thread, zero-initialized ("wiped clean," ready before a crossing) and only combined at the end.
- **A bellow/processor**: one OpenMP thread, owning one contiguous slice of the shared index k (n logical bellows grouped onto the available physical cores).
- **Time = "corridor by corridor, until every bellow has fired"**: the implicit barrier at the end of the `#pragma omp parallel` region — no cell is read (reduced) until every thread has finished its k-slice.

```c
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;
    size_t N = (size_t)n;

    int T = omp_get_max_threads();
    if (T < 1) T = 1;
    if (T > n) T = n;

    /* Each bellow needs its own full set of vases (a private n x n
       accumulator) so that mud from different bellows is never trusted
       until it is summed.  Cap the extra memory this costs. */
    size_t max_bytes = (size_t)1 << 30; /* 1 GiB budget for private vases */
    while (T > 1 && (size_t)T * N * N * sizeof(double) > max_bytes) T /= 2;

    double *priv = (double*)calloc((size_t)T * N * N, sizeof(double));
    if (!priv) {
        /* Fallback: sequential cache-blocked ikj directly into C. */
        memset(C, 0, N * N * sizeof(double));
        const int BK = 256;
        for (int kk = 0; kk < n; kk += BK) {
            int kend = kk + BK < n ? kk + BK : n;
            for (int i = 0; i < n; i++) {
                double *Crow = C + (size_t)i * N;
                const double *Arow = A + (size_t)i * N;
                for (int k = kk; k < kend; k++) {
                    double a = Arow[k];
                    const double *Brow = B + (size_t)k * N;
                    #pragma omp simd
                    for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
                }
            }
        }
        return;
    }

    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        double *myC = priv + (size_t)t * N * N;   /* this bellow's vases */

        /* This bellow's slice of the shared index (its corridor of doors). */
        int kk0 = (int)((long long)n * t / T);
        int kk1 = (int)((long long)n * (t + 1) / T);

        const int BK = 256;  /* k-panel, reused across a block of rows   */
        const int BI = 64;   /* row block, kept resident while panel used */

        for (int ii = 0; ii < n; ii += BI) {
            int iend = ii + BI < n ? ii + BI : n;
            for (int kb = kk0; kb < kk1; kb += BK) {
                int kend = kb + BK < kk1 ? kb + BK : kk1;
                for (int i = ii; i < iend; i++) {
                    const double *Arow = A + (size_t)i * N;
                    double *Crow = myC + (size_t)i * N;
                    for (int k = kb; k < kend; k++) {
                        double a = Arow[k];
                        const double *Brow = B + (size_t)k * N;
                        /* Two seals (a, Brow[j]) meet and are multiplied
                           into one clot before dropping into the vase. */
                        #pragma omp simd
                        for (int j = 0; j < n; j++) {
                            Crow[j] += a * Brow[j];
                        }
                    }
                }
            }
        }
        /* Implicit barrier here: every bellow has fired once before any
           vase below is read. A vase read earlier would lie. */
    }

    /* Read each vase only now that its full barrage of bellows has fired:
       reduce the T private contributions into the real C. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double *Crow = C + (size_t)i * N;
        for (int j = 0; j < n; j++) Crow[j] = 0.0;
        for (int t = 0; t < T; t++) {
            const double *pr = priv + (size_t)t * N * N + (size_t)i * N;
            #pragma omp simd
            for (int j = 0; j < n; j++) Crow[j] += pr[j];
        }
    }

    free(priv);
}
```

PREDICTION: speedup_vs_blocked = 5.0

# MEASUREMENT

No tools (`kernel_bench`, `kernel_contract`, etc.) were reachable in this session — only Docs/Gmail/Drive/Slack MCP tools were available, so I could not compile or run the kernel myself. I am reporting this plainly rather than inventing numbers: **no empirical measurement was performed here.** The prediction above (≈5×) is a reasoned estimate assuming a multi-core (~8–16 thread) benchmark host, based on:
- compute work is split ~evenly over T threads with disjoint A/B reads (no redundant input traffic between threads),
- extra cost is the O(T·n²) reduction pass and O(T·n²) private-buffer memory, which is small relative to O(n³/T) compute once n is reasonably large,
- but per-thread efficiency is well below a hand-tuned register-blocked microkernel, since this uses only compiler auto-vectorization over a plain ikj loop with coarse blocking (BI=64, BK=256).

This artifact and prediction should be run through the actual `kernel_bench` pipeline outside this session to get a real number; I have not fabricated one.

# VERDICT

The literal translation of Seed 3 is a genuine, buildable, different algorithm from both the naive/cache-blocked baseline and from OpenBLAS's tiling strategy: it parallelizes the *contraction* dimension via private full-size accumulators ("one vase-set per bellow"), synchronizes with a hard barrier ("never read before whole"), and only then reduces — rather than parallelizing over output blocks as BLAS does. Expected honest outcome: solid speedup over a single-threaded cache-blocked triple loop (roughly proportional to usable thread count, discounted by reduction and blocking overhead), but **unlikely to beat OpenBLAS**, whose register-tiled microkernel plus panel-packing avoids the O(T·n²) replication/reduction cost entirely and reaches much higher per-core efficiency. The interesting, non-obvious thing this metaphor produced — and that the standard "known way" doesn't do — is the deliberate choice to replicate C across threads and defer combination, purely because the native insisted a vase must never be read until its whole barrage has fired. That constraint, taken literally, is what forces the split-K + private-accumulator + barrier-then-reduce shape, rather than the usual split-output shape.