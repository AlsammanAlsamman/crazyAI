# MAPPING

Per-seed world→problem tables:

**SEED 1 — island-tiling**

| World object | Problem object |
|---|---|
| campsite ground | main memory (RAM) |
| one campsite bed | one cache level's working set (L2/L1) |
| the two great tables | matrices A, B |
| an island | one output tile: a row-strip of A × a column-strip of B |
| strip of first table's rows | a block of rows of A (size MC) |
| strip of second table's columns | a block of columns of B (size NC) |

Assumption broken: **"a matrix is a two-dimensional grid living in one memory"** — the matrix is treated as a deck of independent, disjoint sub-problems, not one flat addressable grid.

**SEED 2 — one bowl per corner**

| World object | Problem object |
|---|---|
| a corner of the third table | one element (or micro-tile) of C |
| the bowl | the accumulator holding the running sum for that C entry |
| "sleeps it, folding into what's already there" | in-place `+=` / FMA accumulation, no separate partial-sum array |
| "never open a second bowl for the same corner" | exactly one owner (thread/register) per output element — no atomics, no reduction-then-merge |

Assumption broken: **"the output is produced one cell at a time, row by row"** — many corners (many bowls, across many islands) can be filled concurrently; row-order is not privileged.

**SEED 3 — burn spent products, no suitcase between islands**

| World object | Problem object |
|---|---|
| a seam-product | one `A[i,k]*B[k,j]` term |
| the fire | discarding the product the instant it's folded into the bowl (never stored as its own array entry) |
| the suitcase | a packed/repacked scratch buffer of A or B data |
| "a suitcase can't carry to the next island what the last one already spent" | forbids OpenBLAS-style **panel packing that is reused across many tiles** — each island reloads straight from A/B, nothing persists past the island's lifetime |
| "wait at each window only until its own island's bowl has finished... never for a neighbor's" | no barrier between tiles; each thread finishes its own tile and moves on (dynamic scheduling, not lock-step) |

Assumption broken: **"one product is one problem; many products are many problems"** — a product is not a stored, individually-tracked unit of work; it's a transient event that must vanish the instant it's consumed (this is literally the semantics of a fused multiply-add: the product never occupies its own memory slot).

# CHOSEN SEED

SEED 3 (burn spent products / no suitcase between islands), read together with SEED 2's "one bowl per corner" (they compose naturally: SEED 3 tells you *how* a product dies, SEED 2 tells you *where* the sum lives).

SEED 3 is the most literal (fire = FMA, no intermediate storage) and the most different from the known way: OpenBLAS's entire performance story is built on the opposite move — pack a panel of A or B once into a contiguous, cache/TLB-friendly buffer and **reuse that suitcase across many tiles** to amortize the packing cost. The native explicitly forbids that: nothing survives an island's boundary. So the artifact below deliberately does **not** pack/copy A or B into scratch buffers; each tile streams directly from the original arrays and throws away everything but the finished corner values.

# ASSUMPTION BROKEN

"One product is one problem; many products are many problems" (products are transient, never individually stored or tracked — folded via FMA straight into the one permanent bowl for that corner), together with "the output is produced one cell at a time, row by row" (many corners/islands are filled concurrently, no cross-tile synchronization).

# ARTIFACT

Objects, made literal:
- **campsite ground** = RAM; **a bed** = the L2 working set of one tile.
- **island** = one (row-block of A) × (col-block of B) pair → one C tile, owned by exactly one thread for its whole lifetime.
- **bowl** = a `__m256d` register (4 running sums for a 1×4 strip of the corner) or, at the edges, a scalar `double`, living only in registers/`C` itself — never spilled to an intermediate array.
- **seam** = the index k where `A[i,k]` and `B[k,j]` meet; the k-loop is the seam search.
- **fire** = the product is formed by `_mm256_fmadd_pd` and consumed in the same instruction — it never exists as a separately addressable value.
- **suitcase forbidden between islands** = no packing buffer for A or B; each tile reads straight from the original row-major arrays.
- **window / no idling for a neighbor** = `#pragma omp parallel for collapse(2) schedule(dynamic)` over tiles — each thread finishes its own tile's full k-reduction and immediately grabs the next, with no barrier tying it to any other tile.
- **trunk** = C itself, written once per tile when that tile's bowls are done.
- **processor** = one OpenMP thread = one islander, working one island start-to-finish.
- **time** = the k-loop inside a tile (sequential seam search); tiles themselves are unordered/parallel.

```c
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#ifndef MC
#define MC 64   /* island row-strip height  (bed size, rows of A)   */
#endif
#ifndef NC
#define NC 64   /* island col-strip width   (bed size, cols of B)   */
#endif
#ifndef KC
#define KC 256  /* how much of the seam we search before re-checking cache residency */
#endif

/* C = A * B, row-major, n x n. C may be overwritten. */
void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    int nbi = (n + MC - 1) / MC;
    int nbj = (n + NC - 1) / NC;

    /* Each (bi,bj) is one island: a strip of A's rows paired with a
       strip of B's columns. One thread owns one island for its whole
       life -- no island ever waits on a neighbor's trunk. */
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < nbi; bi++) {
        for (int bj = 0; bj < nbj; bj++) {
            int i0 = bi * MC, i1 = (i0 + MC < n) ? i0 + MC : n;
            int j0 = bj * NC, j1 = (j0 + NC < n) ? j0 + NC : n;

            /* Walk the seam (k) in chunks so the relevant slice of B
               stays hot in cache -- but nothing here is packed into
               a scratch "suitcase" that outlives this island. */
            for (int k0 = 0; k0 < n; k0 += KC) {
                int k1 = (k0 + KC < n) ? k0 + KC : n;

                for (int i = i0; i < i1; i++) {
                    const double *Arow = A + (size_t)i * n;
                    double *Crow = C + (size_t)i * n;
                    int j = j0;

                    /* one bowl (4-wide register) per corner-group */
                    for (; j + 4 <= j1; j += 4) {
                        __m256d bowl = _mm256_loadu_pd(&Crow[j]);
                        for (int k = k0; k < k1; k++) {
                            __m256d a = _mm256_set1_pd(Arow[k]);
                            __m256d b = _mm256_loadu_pd(&B[(size_t)k * n + j]);
                            /* fold, then burn: the product a*b never
                               exists as its own stored value */
                            bowl = _mm256_fmadd_pd(a, b, bowl);
                        }
                        _mm256_storeu_pd(&Crow[j], bowl); /* lift bowl into trunk */
                    }
                    /* ragged edge: scalar bowl */
                    for (; j < j1; j++) {
                        double bowl = Crow[j];
                        for (int k = k0; k < k1; k++)
                            bowl += Arow[k] * B[(size_t)k * n + j];
                        Crow[j] = bowl;
                    }
                }
            }
        }
    }
}
```

# PREDICTION

PREDICTION: speedup_vs_blocked = 12

Reasoning: vs. a single-core scalar cache-blocked triple loop, this gets ~4x from AVX2 (4 doubles/FMA) and roughly another ~3–4x from multithreading across islands on a typical 4–8 core desktop/laptop (parallel efficiency hurt by memory bandwidth since we deliberately refuse to pack panels — SEED 3's constraint). I do **not** expect this to beat OpenBLAS, which packs panels precisely to avoid the redundant re-reads of B that this design accepts as the cost of taking the seed literally.

# MEASUREMENT

Not run in this session — `kernel_bench` is not among the tools actually available to me here (my tool list is Claude Docs/Gmail/Drive/Slack only; the task's own instructions confirm "No tools are available in this session... the pipeline will compile and measure it"). I'm reporting the prediction above honestly as a prediction, not as a measured result, and have not fabricated a measurement number. If the pipeline runs this and reports a number, the next step would be to compare it against the 12x prediction and, if it's far off (especially if bandwidth-bound from the no-packing rule), the first fix to try would be reintroducing a *bounded* pack of just the current KC×NC slice of B (still discarded at the island's end, so it doesn't violate "no suitcase carried *between* islands," only reused *within* one).

# VERDICT

Artifact delivered and prediction stated as required; empirical verdict is pending actual execution of `kernel_bench`, which is outside this session's available toolset. The honest status is: **untested, prediction only**.