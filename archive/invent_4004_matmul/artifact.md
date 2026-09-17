# MAPPING

**SEED 1 — "A clock-shaped chest ticks once per shared index at each row-column crossing to pace the count."**

| World object | Problem object |
|---|---|
| row of table 1 / column of table 2 | row `i` of A / column `j` of B |
| crossing of row and column | output cell `(i,j)` |
| clock-shaped chest | a private counter/state for iterating the shared index `k = 0..n-1`, instantiated fresh per `(i,j)` |
| "wind it once for the shared index" | set loop bound = `n` |
| one tick | one `k` step: fetch `A[i][k]` and `B[k][j]` |
| "a thousand chests can tick at once, side by side" | many `(i,j)` crossings processed concurrently, not row-by-row in sequence |

Assumption broken: **"the output is produced one cell at a time, row by row."**

**SEED 2 — "A candle is burned completely for every single argument, so no unfinished claim can ever be forwarded."**

| World object | Problem object |
|---|---|
| the spoken "argument" (paired bone circles) | the scalar term `A[i][k]*B[k][j]` |
| candle burned completely, never rushed | the multiply must fully retire (full IEEE double), no shortcuts |
| wax dripping into a bowl | a *local register accumulator*, filled by pooling finished products, not by an incremental in-place update |
| "none darkening another's candle" | independent product computations that don't share mutable state — safe for parallel/ILP execution |
| weighing the pool once the clock comes full round | one final reduction after ALL `k` are done, not a running fold interleaved with something else |

Assumption broken: partially reframes **"the whole sum over the shared index is finished before the next cell is started"** into "sum built as an independent pool + single weigh," which argues for decoupled multiply/accumulate register chains (ILP) rather than one serial `+=`.

**SEED 3 — "Finished sums are carried one-way by runners to the island of Conclusions, since nothing in this world travels backward."**

| World object | Problem object |
|---|---|
| finished sum / fresh bone circle | the completed scalar `C[i][j]` after the *entire* k-reduction |
| runner | a single one-directional store instruction moving that value into C's memory |
| "nothing may be carried backward" | C is **never read back** once any part of it is finished — no read-modify-write passes over C |
| "I do not chase the runner; I wait ... for the empty-handed return" | issue a non-blocking/streaming store, confirm completion with a fence, don't otherwise block |
| "break the spent chest, sweep... into flat dust" | per-cell scratch state is discarded instantly, O(1) live state per crossing |
| "a million crossings ... none needing a landmark, only its row and column" | crossings addressed purely by `(i,j)`, trivially parallel, no shared state / false sharing |
| "the third table is simply arrived at, whole" | C emerges complete only at a single final barrier — no incremental multi-pass build-up |

Assumption broken: **the implicit multi-pass read-modify-write of C** that both the naive loop (`C[...] += a*B[...]` executed **n times per cell**) and classic k-blocked GEMM (accumulate into C once per k-block) perform. Seed 3 forbids ever revisiting C, forcing the *entire* K-reduction to happen in private register/scratch state and C to be touched exactly once, by a write-only, one-directional store.

# CHOSEN SEED

**SEED 3** — most literal (a runner that only ever goes forward maps directly onto "no RMW of C, one-way non-temporal store") and most different from OpenBLAS's approach, which *does* pack panels and *does* revisit C tiles across K-blocking passes. Seed 3 explicitly forbids exactly that revisiting.

# ASSUMPTION BROKEN

"The output is produced one cell at a time, row by row" **and**, more specifically, the hidden assumption that accumulating into C requires repeated backward (read-then-write) traffic to C's memory. The minimal baseline literally does `C[i*n+j] += a*B[...]` **n times per cell** — n reads + n writes to C per output element. Seed 3 says: do the whole shared-index reduction in local state, touch C exactly once, and never look at it again (motivating a non-temporal store).

# ARTIFACT

Mapping of every world object → computational object used in the code:
- flat, shadowless ground → per-crossing private register state (no false sharing between threads/lanes)
- table rows/columns walked forward, never packed into a new box → A/B read directly from original memory, no panel-copy buffer
- clock-shaped chest → the `k` loop, restarted fresh per `(i,j)` (or per row-block/column-block)
- candles, "none darkening another's" → two independent FMA accumulator chains (`acc0`,`acc1`) for ILP, plus 4-wide AVX2 lanes across `j`
- wax pooled, weighed once → `acc0+acc1` computed only after the full `k` sweep
- runner, one-way, never chased → `_mm256_stream_pd` non-temporal store (fallback to normal store if misaligned), with a single `_mm_sfence()` at the very end standing for "waiting for every runner's empty-handed return"
- "a thousand chests tick at once, side by side" → OpenMP over row-blocks (cores) + SIMD lanes (vector width)
- B-column reused by many rows before the chest is discarded → row-block (`IB_ROWS`) amortizes the B column-panel across many `i`, giving cache reuse without ever packing/copying

```c
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#ifndef IB_ROWS
#define IB_ROWS 64
#endif

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;

#if defined(__AVX2__) && defined(__FMA__)
    const int n4 = n - (n % 4); /* aligned j-bound for 4-wide vector lanes */

    #pragma omp parallel for schedule(dynamic)
    for (int ib = 0; ib < n; ib += IB_ROWS) {
        int i_end = ib + IB_ROWS < n ? ib + IB_ROWS : n;

        /* B column-panel B[*, j..j+3] ("the column of bone circles") is
           reused across every row i in this block ("a thousand chests
           tick at once, side by side") before we move to the next
           column-panel. Nothing is packed/copied -- read straight from
           the original table. */
        for (int j = 0; j < n4; j += 4) {
            for (int i = ib; i < i_end; i++) {
                const double *Ai = A + (size_t)i * n;
                __m256d acc0 = _mm256_setzero_pd(); /* two independent   */
                __m256d acc1 = _mm256_setzero_pd(); /* "candles" -> ILP  */
                int k = 0;
                int k2 = n - (n % 2);
                for (; k < k2; k += 2) {
                    __m256d a0 = _mm256_set1_pd(Ai[k]);
                    __m256d a1 = _mm256_set1_pd(Ai[k + 1]);
                    __m256d b0 = _mm256_loadu_pd(B + (size_t)k * n + j);
                    __m256d b1 = _mm256_loadu_pd(B + (size_t)(k + 1) * n + j);
                    acc0 = _mm256_fmadd_pd(a0, b0, acc0); /* burn candle */
                    acc1 = _mm256_fmadd_pd(a1, b1, acc1); /* burn candle */
                }
                if (k < n) {
                    __m256d a0 = _mm256_set1_pd(Ai[k]);
                    __m256d b0 = _mm256_loadu_pd(B + (size_t)k * n + j);
                    acc0 = _mm256_fmadd_pd(a0, b0, acc0);
                }
                __m256d sum = _mm256_add_pd(acc0, acc1); /* weigh the pool */
                double *dst = C + (size_t)i * n + j;
                if ((((uintptr_t)dst) & 31u) == 0) {
                    _mm256_stream_pd(dst, sum);  /* one-way runner: never read back */
                } else {
                    _mm256_storeu_pd(dst, sum);
                }
            }
        }

        /* remainder columns: still register-accumulated, single forward
           write -- never a read-modify-write of C. */
        for (int i = ib; i < i_end; i++) {
            const double *Ai = A + (size_t)i * n;
            for (int j = n4; j < n; j++) {
                double acc0 = 0.0, acc1 = 0.0;
                int k = 0;
                int k2 = n - (n % 2);
                for (; k < k2; k += 2) {
                    acc0 += Ai[k]     * B[(size_t)k * n + j];
                    acc1 += Ai[k + 1] * B[(size_t)(k + 1) * n + j];
                }
                if (k < n) acc0 += Ai[k] * B[(size_t)k * n + j];
                C[(size_t)i * n + j] = acc0 + acc1;
            }
        }
    }
    _mm_sfence(); /* wait, still, for every runner's landing before returning */
#else
    /* No AVX2/FMA: still avoid repeated read-modify-write of C. */
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        const double *Ai = A + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int k = 0; k < n; k++)
                acc += Ai[k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = acc;
        }
    }
#endif
}
```

Numerical note: sums are reordered (pairwise via `acc0/acc1`) relative to a strict left-to-right fold, giving ordinary IEEE-754 rounding-order differences (~1e-13 relative for moderate `n`) — "exact up to floating-point reassociation," not bit-identical to a naive accumulation.

PREDICTION: speedup_vs_blocked = 8.0

# MEASUREMENT

No `kernel_bench` tool is available in this session (explicitly stated in the task setup — "No tools are available in this session"). The prediction above is registered *before* any measurement, as required. Actual compilation/benchmarking must be performed by the pipeline; I have not executed it and am not fabricating a number for it.

# VERDICT

Unmeasured — reported honestly as such rather than inventing a result. Reasoning behind the prediction: the artifact removes the two clearest sources of waste in the naive/blocked baseline (per-`k` read-modify-write of C, and an initial `memset`), adds genuine ILP via two independent accumulator chains ("candles that don't darken each other"), 4-wide AVX2/FMA vectorization, row-block reuse of the B column panel without any packing step, non-temporal stores exploiting the fact that C is provably write-once/one-way, and multi-core parallelism via OpenMP. Main risks to the 8× prediction: (1) if the reference "cache-blocked triple loop" is already OpenMP-parallel and auto-vectorized well by `-O3 -march=native`, the gap shrinks toward 2–4×; (2) the row-block size `IB_ROWS=64` is a guess, not tuned against real cache sizes, and is the most likely first thing to retune if measurement shows a smaller-than-predicted gain; (3) unaligned fallback stores (non-multiple-of-4 `n`, or non-32-byte-aligned `C`) lose the streaming-store benefit for those rows. This design is very unlikely to beat OpenBLAS itself, which additionally packs panels and register-tiles far more aggressively across all cache levels — that comparison was not attempted here.