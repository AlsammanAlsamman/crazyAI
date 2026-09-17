# MAPPING

**SEED 1 — dreamer-blue pens, one per shared index**

| World object | Problem object |
|---|---|
| Two towels of nerve-thread pinned to two fixed desks | Matrices A, B, stored row-major, **never repacked/copied** |
| Row of pens along the shared seam, one per index | One scalar-broadcast "operator" per value of k |
| A west nerve and an east nerve that answer to the same pen | A[i,k] and B[k,j] — the two operands sharing index k |
| The pen trembling out the product ("I never do the multiplying with my own hand") | The multiply is delegated to a single elementary FMA op, not hand-coded as part of a bigger fused loop |
| One pen serving *every* cell that needs it, not one pen per cell | The same k is reused across **all** (i,j) — a rank-1 structure, not a per-cell inner loop |

Breaks: "one product is one problem; many products are many problems" (many cells share one pen/index) and "every product is computed exactly once" is actually reinforced, not broken, by seed 3.

**SEED 2 — the moon's single continuous turning**

| World object | Problem object |
|---|---|
| One drop born of shared index k, "however far-flung across the towel" | The entire rank-1 contribution A[:,k]⊗B[k,:] to C, spread over **all** n² cells |
| The moon's one continuous turning | One synchronized sweep (parallel-for + barrier) that applies that whole rank-1 update to all of C **before** the next index starts |
| "I do not add the drops one at a time, cell by patient cell" | Output is **not** produced one cell at a time, row by row — it's produced one shared-index layer at a time, across the whole matrix |
| "Moon-turn after moon-turn, once per shared index, never rush or split" | The k-loop is the outermost, sequential loop; no two k's contributions are ever interleaved or partially computed |
| "Pours, doesn't count — cannot skip or double" | Use a data-parallel (race-free) sweep: disjoint output rows per thread, so nothing is skipped or double-counted |

Breaks directly and explicitly:
- "the output is produced one cell at a time, row by row" — **broken**: output is produced one *index-layer* at a time, whole-matrix at a time.
- "the whole sum over the shared index is finished before the next cell is started" — **broken**: the opposite — no cell's sum is finished until the *last* index-layer (k = n−1) has been poured.

**SEED 3 — spent pens thrown away after exactly one trembling**

| World object | Problem object |
|---|---|
| A pen drawn on twice would "lie about the product" | No persistent per-k accumulator/register is reused across iterations — each k's scalar a = A[i,k] is a fresh, disposable temporary |
| Two original towels stay fixed for auditability | A and B are read-only, in place, never overwritten or transformed (no packing buffers) |

Breaks: mild reinforcement of "every product computed exactly once", not a new assumption.

# CHOSEN SEED

**Seed 2 — the moon.** It is the most literal (it dictates an exact loop-order and synchronization discipline, not just a vague "parallelize" idea) and it is maximally different from the known way: OpenBLAS's microkernel philosophy is "finish a small C-tile completely, keeping it in registers, before moving to the next tile" — i.e., finish sums early for register/cache reuse. The moon does the *opposite* on purpose: it refuses to finish *any* cell until every shared index has poured its drop, moving instead in complete, synchronized, whole-matrix layers. That is a genuine structural inversion, not a cosmetic rename of blocking.

# ASSUMPTION BROKEN

"The whole sum over the shared index is finished before the next cell is started" **and** "the output is produced one cell at a time, row by row." The moon-kernel produces C as a sequence of n complete rank-1 (outer-product) updates over the *entire* matrix, each one a synchronized, all-cells-at-once "pour," rather than as n² independent dot-product accumulations.

# ARTIFACT

Mapping of every world-object to a computational one, literally:
- **Desks (immovable)** → A and B stay in their original row-major memory, untouched, unpacked, never transposed or copied into panels (unlike OpenBLAS).
- **Pens, one per shared index** → the scalar `a = A[i*n+k]` and the row pointer `B+k*n`, recomputed fresh each k.
- **Pinching nerves at the nib / pen trembles the product** → the elementary FMA `crow[j] += a*brow[j]`.
- **The moon's one continuous turning** → one `#pragma omp parallel for` sweep over all rows i for fixed k, which touches the *entire* C matrix in that turn, with an implicit barrier at its end (the moon-turn "finishes" before the next begins).
- **"Never rush or split" the moon** → the k-loop itself stays strictly sequential/outermost — we do *not* parallelize across k (that would split one index's pour into a race across turns).
- **Spent pens thrown away** → `a` and `brow` are turn-local temporaries, never carried over or reused across k.
- **Processor** → each OpenMP thread is "a hand" pouring into disjoint rows (no two hands ever touch the same cell in the same turn → no counting error).
- **Time** → the outer k-loop is the clock; one tick = one full-matrix layer applied.

```c
#include <string.h>
#include <omp.h>

/* C = A*B, row-major, n x n. Literal translation of "the moon":
   one synchronized whole-matrix pass per shared index k, never
   finishing a cell early, never packing/moving the two fixed
   towels (A, B), never reusing a spent pen (per-k temporaries). */
void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    for (int k = 0; k < n; k++) {                    /* one moon-turn per shared index */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            double a = A[(size_t)i * n + k];          /* west-desk nerve for this pen */
            const double *brow = B + (size_t)k * n;   /* east-desk nerve row for this pen */
            double *crow = C + (size_t)i * n;
            #pragma omp simd
            for (int j = 0; j < n; j++) {
                crow[j] += a * brow[j];                /* pen trembles out the product, poured not counted */
            }
            /* 'a' and brow are the spent pen: discarded here, never reused for another k */
        }
        /* implicit OpenMP barrier: this moon-turn is fully poured into every
           rightful cell of the whole third towel before the next k begins */
    }
}
```

# PREDICTION

Reasoning: this is the k-i-j (outer-product) loop order, which is known to have worse asymptotic memory traffic on C than a cache-blocked kernel (every k re-reads/re-writes the *entire* C matrix — O(n) full sweeps of C instead of O(n/√cache) sweeps), and it also forces a strided (column) read of A for every k. Its only real lever is the multicore parallel-for inside each moon-turn, which a single-threaded "cache-blocked triple loop" reference does not have. For moderate n where C still mostly survives in L3 across turns, I expect the multicore win to outweigh the worse traffic pattern by a modest factor, but not to approach OpenBLAS, and to degrade (possibly go below 1×) as n grows past L3 capacity, since then every one of the n moon-turns pays a full DRAM sweep of C.

PREDICTION: speedup_vs_blocked = 3.0

# MEASUREMENT

Not performed. Tool access (`kernel_bench`) was explicitly disabled for this session ("No tools are available in this session... the pipeline will compile and measure it"). I am reporting this plainly rather than inventing numbers: the prediction above is the pre-registered estimate; actual speedup vs. the cache-blocked baseline (and vs. OpenBLAS) must come from the external pipeline run, not from me.

# VERDICT

Unmeasured in this session, so no verdict on absolute performance can honestly be claimed yet. Structurally, the artifact is a faithful, non-hand-wavy translation of the native's description — it genuinely inverts the "finish each cell's sum before moving on" assumption via a synchronized outer-product sweep, which the known blocked/microkernel approach never does. The predicted risk is real and specific (O(n) full C-matrix traffic instead of O(n/√cache)), so if measurement disagrees with the 3.0× prediction, the most informative next step (not yet taken, since it would require the disabled bench tool) would be to test whether the failure mode is exactly the predicted one: speedup shrinking or inverting as n exceeds L3 size — which would confirm the mechanism rather than just the outcome.