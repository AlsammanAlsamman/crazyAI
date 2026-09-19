## APPROACH

The naive baseline is O(n·k): it rebuilds each k-mer's 2-bit code from scratch, one base at a time, for every one of the ~n starting positions. This violates no correctness requirement but wastes k−1 of the k base-lookups at every step, since consecutive windows share k−1 bases.

Optimizations applied, each guided by the "silent assumptions" listed:
1. **Rolling code (kills "built fully from scratch")** — maintain a running 2k-bit code; shift left 2 bits, OR in the new base, mask to width, at every step. This turns the inner loop into O(n) total instead of O(n·k).
2. **Branchless base decode** — a 256-entry lookup table replaces the `switch`, removing per-base branch mispredictions.
3. **Parallelize across the sequence (kills "must be read once, start to end, in order")** — split the n−k+1 windows into contiguous chunks, one per OpenMP thread. Each thread reconstructs its chunk's *first* window directly (k lookups, done once) then rolls forward independently — the windows are read out of global order but each window is still counted correctly exactly once.
4. **Per-thread private histograms + parallel reduction (kills "count table updated one position at a time" / avoids contention)** — when the table (4^k entries) is small enough to afford one private copy per thread within a memory budget, each thread accumulates into its own histogram (no synchronization at all in the hot loop), then a final parallel pass sums the private tables into the caller's `counts`. This is correct and avoids both locks and cache-line ping-pong.
5. **Fallback for huge tables** — if 4^k is too large to replicate per thread, fall back to `#pragma omp atomic` increments directly into the shared table (still correct, still parallel, no giant memory blowup), and to single-threaded rolling for small inputs where thread setup overhead would dominate.

All four assumptions instructed to overturn are addressed: no full from-scratch rebuild, table updates happen per-thread/batched not strictly one-at-a-time, and traversal is not forced start-to-end-in-order.

## ARTIFACT

```c
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static unsigned char g_b2c[256];
static int g_b2c_init = 0;

static void init_b2c(void) {
    for (int i = 0; i < 256; i++) g_b2c[i] = 3;
    g_b2c[(unsigned char)'A'] = 0;
    g_b2c[(unsigned char)'C'] = 1;
    g_b2c[(unsigned char)'G'] = 2;
    g_b2c_init = 1;
}

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (!g_b2c_init) init_b2c();
    if (k <= 0 || n < k) return;

    long total = (long)n - (long)k + 1; /* number of windows */
    uint64_t mask, entries;
    if (k >= 32) { mask = ~0ULL; entries = 0; /* not realistically reachable */ }
    else { mask = (1ULL << (2 * k)) - 1; entries = 1ULL << (2 * k); }

    const unsigned char *ub2c = g_b2c;
    const unsigned char *useq = (const unsigned char *)seq;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#endif
    if (total < 200000L) nthreads = 1; /* not worth parallel overhead */

    if (nthreads <= 1) {
        uint64_t code = 0;
        for (int j = 0; j < k - 1; j++) code = (code << 2) | ub2c[useq[j]];
        for (long i = 0; i < total; i++) {
            code = ((code << 2) | ub2c[useq[i + k - 1]]) & mask;
            counts[code]++;
        }
        return;
    }

    const uint64_t BUDGET_BYTES = (uint64_t)256 * 1024 * 1024; /* 256MB extra budget */
    int use_private = (k < 32) &&
        (entries <= (BUDGET_BYTES / (uint64_t)sizeof(uint64_t) / (uint64_t)nthreads));

    if (use_private) {
        uint64_t **locals = (uint64_t **)malloc(sizeof(uint64_t *) * (size_t)nthreads);
        for (int t = 0; t < nthreads; t++)
            locals[t] = (uint64_t *)calloc((size_t)entries, sizeof(uint64_t));

#ifdef _OPENMP
        #pragma omp parallel num_threads(nthreads)
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            long chunk = (total + nthreads - 1) / nthreads;
            long start = (long)tid * chunk;
            long end = start + chunk;
            if (end > total) end = total;
            if (start < end) {
                uint64_t *lc = locals[tid];
                uint64_t code = 0;
                for (int j = 0; j < k; j++) code = (code << 2) | ub2c[useq[start + j]];
                code &= mask;
                lc[code]++;
                for (long i = start + 1; i < end; i++) {
                    code = ((code << 2) | ub2c[useq[i + k - 1]]) & mask;
                    lc[code]++;
                }
            }
        }

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long c = 0; c < (long)entries; c++) {
            uint64_t sum = 0;
            for (int t = 0; t < nthreads; t++) sum += locals[t][c];
            counts[c] = sum;
        }
        for (int t = 0; t < nthreads; t++) free(locals[t]);
        free(locals);
    } else {
#ifdef _OPENMP
        #pragma omp parallel num_threads(nthreads)
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            long chunk = (total + nthreads - 1) / nthreads;
            long start = (long)tid * chunk;
            long end = start + chunk;
            if (end > total) end = total;
            if (start < end) {
                uint64_t code = 0;
                for (int j = 0; j < k; j++) code = (code << 2) | ub2c[useq[start + j]];
                code &= mask;
#ifdef _OPENMP
                #pragma omp atomic
#endif
                counts[code]++;
                for (long i = start + 1; i < end; i++) {
                    code = ((code << 2) | ub2c[useq[i + k - 1]]) & mask;
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    counts[code]++;
                }
            }
        }
    }
}
```

PREDICTION: 10

## MEASUREMENT

No execution tools (`kmer_bench`, etc.) were actually available in this session — the tool list shown was aspirational but the runtime environment explicitly stated "No tools are available in this session," so I could not run the benchmark myself. I am reporting this plainly rather than fabricating numbers. The 10× prediction above is derived purely from algorithmic reasoning: the rolling-code rewrite alone removes a factor of ~k from the inner-loop work (each base is looked up once instead of up to k times), and for k in the range typically used for benchmarking such kernels (roughly 8–16), that alone predicts ~8–16× on a single core; multithreading with private per-thread histograms should add a further near-linear factor for large n on machines with several cores, bounded by memory bandwidth for the final reduction pass. Actual measured numbers depend on the harness's n, k, and thread count, which I don't have visibility into here.

## VERDICT

Unverified in this session — the implementation is algorithmically sound (O(n) instead of O(n·k), correct exact counts via either private-table reduction or atomic fallback, all edge cases like n<k and large k handled without UB), and should beat the naive kernel whenever the pipeline actually compiles and runs it, but I have no empirical confirmation to report and am not claiming one.