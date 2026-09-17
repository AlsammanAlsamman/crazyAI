## Step 1 — Mapping table for each SEED

| SEED | World object | Problem object | Silent assumption it breaks |
|---|---|---|---|
| **1** "I hold every door open at once instead of walking them in sequence, so all products form simultaneously." | Doors = values of the shared index k. Walking them "one after another" = a scalar accumulator march `for k: acc += a*b`. Holding all doors open = exposing many k's to the hardware at once (SIMD lanes, independent accumulator chains) instead of one serial dependency chain. | The k-loop reduction is restructured from *one* long serial FMA dependency chain into *several* independent chains that execute concurrently and are combined only once at the end. | **"one product is one problem; many products are many problems."** Many k-products are treated as a single simultaneous operation, not n sequential ones. |
| **2** "Sparrows-as-windows physically transport each door's product into the granary that shares its crossing, performing the accumulation as flight rather than arithmetic." | Sparrow = a SIMD/FMA lane; "window on a plane" = a vector register sliding across a contiguous row of B. Granary-stalk = C[i][j], sitting at the row/column crossing. "Accumulation as flight" = the multiply-add is fused (FMA) into the act of moving data into the register, not a separate load→multiply→add→store sequence. | Use fused multiply-add so the product and its addition into the destination happen as one hardware "flight" (single rounding), and let many (i,j) cells fill concurrently rather than the output being finished one cell at a time. | **"the output is produced one cell at a time, row by row."** Many granaries are filled concurrently by independent flights, not in strict row-major cell order. |
| **3** "I discard the spent product-bodies once a granary is full, keeping only the final sum and the untouched original stalks for verification." | Untouched stalks = A, B never rewritten (already required by the contract). Discarded product-bodies = individual a·b terms never materialized/stored once folded into the sum. | Products stay transient (registers only); only the reduced sum is written to memory; A/B stay byte-identical so a door can be "walked again" for a check. | This mostly **restates** an assumption already honored by the naive kernel (products aren't stored) — it doesn't break anything new. Weakest, least novel seed. |

## Step 2 — Choice

SEED 1 is picked: it is the most literal (doors = k-index, walking = serial accumulator dependency, holding-open = simultaneity) and the most different from the known way. OpenBLAS's "register-tiled microkernel" already gets throughput from vector *width*; it does **not** generally attack the *serial FMA dependency chain* itself, and it always **repacks** A and B into panels — which the narrator explicitly forbids ("what stays still... I never bend or replant them"). So the artifact must (a) never copy/pack A or B, and (b) literally split the k-walk into independent, concurrently-live accumulator chains that are combined only once.

## Step 3 — Full object mapping

| World object | Computational object |
|---|---|
| Two sun-disc houses (A, B) | The two input matrices, row-major, **read in place, never repacked** |
| Field of stalks ranked by side-count | Rows of A indexed by i, columns of B indexed by j |
| Door (shared index) | One value of k |
| "Present at every door at once" | Two independent FMA accumulator chains per output row, alternating even/odd k, executed concurrently (no single-chain latency bound) |
| Stalks meeting → strawberry (product) | `a * b`, formed via `_mm256_fmadd_pd` (single rounding — the literal "monstrous, larger-than-parent" exact fused step) |
| Sparrow / window on a plane | An AVX2 `__m256d` register sliding across 4 contiguous doubles of a row of B |
| Granary-stalk (third table, at the shadow-crossing) | `C[i][j]` accumulator register, one per row of the 4×4 tile |
| "As many sparrows as doors along that crossing" | n FMA landings per C-cell (contraction length) |
| Burying the granary / discarding spent bodies | Registers are combined and stored; no n³ product tensor is ever materialized |
| Untouched fields | `const double *A, *B` — never mutated |
| My "all-at-once sight" | OpenMP across cores (one thread owns a disjoint 4-row band for its whole lifetime — no races) |
| "Last sparrow lands, third house finished" | k is chunked (KC) purely to bound the working set touched per pass — no packing buffer, just loop restructuring |

## ARTIFACT

```c
#include <immintrin.h>
#include <string.h>
#include <omp.h>

/* Register-tiled micro-kernel for a 4x4 tile of C, rows i0..i0+3, cols j0..j0+3,
   over k in [k0, k0+kc).
   "Doors held open at once": two independent accumulator chains per row
   (acc0 for even k, acc1 for odd k) run concurrently instead of one serial
   FMA dependency chain, and are combined only once, when the last door
   in this chunk has been visited. A and B are read directly from their
   original layout -- never packed/copied ("what stays still, stays still"). */
static inline void microkernel_4x4(int n, const double *A, const double *B,
                                    double *C, int i0, int j0, int k0, int kc,
                                    int first_block) {
    __m256d acc0[4], acc1[4];
    if (first_block) {
        for (int r = 0; r < 4; r++) { acc0[r] = _mm256_setzero_pd(); acc1[r] = _mm256_setzero_pd(); }
    } else {
        for (int r = 0; r < 4; r++) {
            acc0[r] = _mm256_loadu_pd(&C[(size_t)(i0 + r) * n + j0]);
            acc1[r] = _mm256_setzero_pd();
        }
    }

    int k = k0;
    int kend = k0 + kc;
    for (; k + 1 < kend; k += 2) {
        __m256d b0 = _mm256_loadu_pd(&B[(size_t)k * n + j0]);
        __m256d b1 = _mm256_loadu_pd(&B[(size_t)(k + 1) * n + j0]);
        for (int r = 0; r < 4; r++) {
            double a0 = A[(size_t)(i0 + r) * n + k];
            double a1 = A[(size_t)(i0 + r) * n + k + 1];
            acc0[r] = _mm256_fmadd_pd(_mm256_set1_pd(a0), b0, acc0[r]);
            acc1[r] = _mm256_fmadd_pd(_mm256_set1_pd(a1), b1, acc1[r]);
        }
    }
    for (; k < kend; k++) {
        __m256d b0 = _mm256_loadu_pd(&B[(size_t)k * n + j0]);
        for (int r = 0; r < 4; r++) {
            double a0 = A[(size_t)(i0 + r) * n + k];
            acc0[r] = _mm256_fmadd_pd(_mm256_set1_pd(a0), b0, acc0[r]);
        }
    }
    for (int r = 0; r < 4; r++) {
        __m256d sum = _mm256_add_pd(acc0[r], acc1[r]);
        _mm256_storeu_pd(&C[(size_t)(i0 + r) * n + j0], sum);
    }
}

static void scalar_block(int n, const double *A, const double *B, double *C,
                          int i0, int i1, int j0, int j1, int k0, int k1,
                          int first_block) {
    for (int i = i0; i < i1; i++) {
        for (int j = j0; j < j1; j++) {
            double s = first_block ? 0.0 : C[(size_t)i * n + j];
            for (int k = k0; k < k1; k++)
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    int mb = n - (n % 4);
    int nb = n - (n % 4);
    const int KC = 256; /* bound working set per pass; no packing buffer */

    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < mb; i0 += 4) {
        for (int k0 = 0; k0 < n; k0 += KC) {
            int kc = (k0 + KC <= n) ? KC : (n - k0);
            int first_block = (k0 == 0);
            for (int j0 = 0; j0 < nb; j0 += 4)
                microkernel_4x4(n, A, B, C, i0, j0, k0, kc, first_block);
            if (nb < n)
                scalar_block(n, A, B, C, i0, i0 + 4, nb, n, k0, k0 + kc, first_block);
        }
    }

    if (mb < n) {
        #pragma omp parallel for schedule(static)
        for (int i = mb; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
                C[(size_t)i * n + j] = s;
            }
        }
    }
}
```

## PREDICTION

PREDICTION: speedup_vs_blocked = 6.0

Reasoning for the number: an all-cores OpenMP fan-out over the naive/blocked single-thread baseline should give roughly a factor equal to available cores discounted for memory-bandwidth sharing (call it ~4–8x on a typical 8–16 core box), and the 4×4 FMA register tile with the dual-chain k-unroll should add a modest further factor over a compiler-autovectorized scalar-accumulator loop (maybe 1.2–1.5x, since -O3 -march=native may already autovectorize the given baseline's inner j-loop). Multiplying a conservative middle estimate of both gives ~6x. I expect this to fall well short of OpenBLAS (no packing means A's column-strided loads and no true panel reuse across the k-blocks), and I state that honestly rather than inflate the number.

## MEASUREMENT

Not run in this session — no benchmarking tool (`kernel_bench`) was invocable here; per the session's explicit instruction, the reasoning, artifact, and pre-measurement prediction above are provided for the downstream pipeline to compile and measure. I have not fabricated a measured number.

## VERDICT

Honest status: unmeasured. The artifact is a literal translation of SEED 1 — it breaks the serial "march through doors" reduction into concurrently-live accumulator chains and refuses to repack A/B ("what stays still... I never replant them"), which is the one concrete way it diverges from OpenBLAS's known packed-panel microkernel rather than quietly re-deriving it. Because no measurement tool was available in this session, the requested "improve up to four times" loop against real numbers could not be executed here; that iteration should happen once the pipeline reports actual timings against the cache-blocked baseline, at which point the two obvious next levers (in priority order) are: (1) widen the k-unroll from 2 to 4 independent chains if the FMA-port count on the target CPU supports it, and (2) if OpenBLAS-class speed is the goal, revisit the "never replant them" constraint specifically for A (a single, cheap row-panel copy of the current i-block only, not a full BLAS-style pack) since that is where the no-packing choice is most likely to cost the most on real hardware.