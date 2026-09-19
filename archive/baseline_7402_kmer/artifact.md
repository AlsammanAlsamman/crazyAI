## APPROACH

The naive solution re-derives every k-mer's 2-bit-packed code from scratch (`O(n·k)` work): for every start position `i` it re-scans all `k` bases. That repeated work is exactly the inefficiency to remove.

Key optimizations, in order of expected impact:

1. **Rolling code (O(n) instead of O(n·k))** — a k-mer's code differs from its predecessor by one shifted-in base and a mask to drop the oldest base's bits: `code = ((code << 2) | lut[next_base]) & mask`. This turns the inner re-scan into a single O(1) update per position, so the whole scan becomes O(n) regardless of k.
2. **Branchless base lookup** — a 256-entry compile-time constant table (`BASE_LUT`) replaces the `switch` in `base_code`, avoiding branches/mispredicts on every base.
3. **Parallelization, with a memory-aware strategy** since `counts` has `4^k` entries and can range from tiny to huge:
   - *Small alphabet size* (`4^k ≤ 2^20`, i.e. k≤10): give each OpenMP thread its own private counts buffer (fits comfortably in cache), scan disjoint contiguous ranges of the sequence independently (each thread rebuilds its own starting window in O(k), negligible), then reduce the per-thread buffers into `counts`. No contention at all during the scan.
   - *Large alphabet size* (k>10) or too few windows to amortize thread-local buffer allocation: parallel scan writing directly into the shared `counts` array via `#pragma omp atomic`. Collisions are rare because the table is large, so atomic overhead stays low while we still get parallel scanning + overlapped memory latency.
   - *Otherwise* (too few windows, single thread, or allocation failure): plain serial O(n) rolling pass — already strictly better than the naive baseline.

This preserves exact contract semantics (same code packing, high base first, `counts` arrives pre-zeroed and we only add to it) while removing all redundant re-derivation work and exploiting available cores when it's safe memory-wise.

## ARTIFACT

```c
#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static const unsigned char BASE_LUT[256] = {
    ['A'] = 0, ['C'] = 1, ['G'] = 2, ['T'] = 3
};

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || n < k) return;

    uint64_t mask = (k >= 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);
    uint64_t numCounts = (k >= 32) ? 0ULL /* unused in this branch */ : (1ULL << (2 * k));
    long long numWindows = (long long)n - (long long)k + 1;
    if (numWindows <= 0) return;

#ifdef _OPENMP
    int maxThreads = omp_get_max_threads();
#else
    int maxThreads = 1;
#endif

    int useLocalTables = 0;
    int useAtomicParallel = 0;
    if (maxThreads > 1 && k < 32) {
        if (numCounts <= (1ULL << 20) && numWindows > 100000) {
            useLocalTables = 1;
        } else if (numWindows > 1000000) {
            useAtomicParallel = 1;
        }
    }

    if (useLocalTables) {
        int nthreads = maxThreads;
        uint64_t *local_all = (uint64_t *)calloc((size_t)nthreads * (size_t)numCounts, sizeof(uint64_t));
        if (local_all) {
#ifdef _OPENMP
            #pragma omp parallel num_threads(nthreads)
#endif
            {
#ifdef _OPENMP
                int tid = omp_get_thread_num();
#else
                int tid = 0;
#endif
                uint64_t *lc = local_all + (size_t)tid * (size_t)numCounts;
                long long chunk = (numWindows + nthreads - 1) / nthreads;
                long long start = (long long)tid * chunk;
                long long end = start + chunk;
                if (end > numWindows) end = numWindows;
                if (start < end) {
                    uint64_t code = 0;
                    for (int j = 0; j < k - 1; j++) {
                        code = (code << 2) | BASE_LUT[(unsigned char)seq[start + j]];
                    }
                    for (long long i = start; i < end; i++) {
                        code = ((code << 2) | BASE_LUT[(unsigned char)seq[i + k - 1]]) & mask;
                        lc[code]++;
                    }
                }
            }
            for (int t = 0; t < nthreads; t++) {
                uint64_t *lc = local_all + (size_t)t * (size_t)numCounts;
                for (uint64_t c = 0; c < numCounts; c++) {
                    counts[c] += lc[c];
                }
            }
            free(local_all);
            return;
        }
        /* allocation failed: fall through to serial path below */
    } else if (useAtomicParallel) {
        int nthreads = maxThreads;
#ifdef _OPENMP
        #pragma omp parallel num_threads(nthreads)
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            long long chunk = (numWindows + nthreads - 1) / nthreads;
            long long start = (long long)tid * chunk;
            long long end = start + chunk;
            if (end > numWindows) end = numWindows;
            if (start < end) {
                uint64_t code = 0;
                for (int j = 0; j < k - 1; j++) {
                    code = (code << 2) | BASE_LUT[(unsigned char)seq[start + j]];
                }
                for (long long i = start; i < end; i++) {
                    code = ((code << 2) | BASE_LUT[(unsigned char)seq[i + k - 1]]) & mask;
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    counts[code]++;
                }
            }
        }
        return;
    }

    /* Serial fallback: single O(n) rolling-code pass */
    {
        uint64_t code = 0;
        for (int j = 0; j < k - 1; j++) {
            code = (code << 2) | BASE_LUT[(unsigned char)seq[j]];
        }
        for (long long i = 0; i < numWindows; i++) {
            code = ((code << 2) | BASE_LUT[(unsigned char)seq[i + k - 1]]) & mask;
            counts[code]++;
        }
    }
}
```

PREDICTION: 8

## MEASUREMENT

No tools (including `kmer_bench`) were available to invoke in this session — the environment explicitly stated "No tools are available in this session." I therefore could not execute the benchmark harness myself and have no empirical timing numbers to report. The prediction above (~8x over the naive `O(n·k)` baseline for a k in the range where the naive cost is dominated by re-derivation, e.g. k≈8–16) is an estimate based on: (a) the O(k) → O(1) algorithmic reduction per position from the rolling code, which alone should yield a factor close to k for k-scan cost, tempered by memory-bound counting writes that don't shrink; and (b) additional but partial multicore scaling from the two OpenMP paths, which is capped by memory bandwidth/atomic contention rather than core count. I did not get to run and refine this against real measurements within this session, so this should be treated as an unverified estimate rather than a confirmed result.

## VERDICT

Correctness: the implementation preserves the exact contract (`void kernel(int n, const char*, int k, uint64_t*)`, high-base-first 2-bit packing, additive updates into a pre-zeroed table) across all three code paths (local-table-reduction, atomic-parallel, serial), and each path independently produces the same per-window code sequence as the reference naive computation, just computed incrementally instead of recomputed from scratch.

Performance: unverified in this session due to no tool access — I cannot honestly claim a measured speedup. The algorithmic case (O(n) vs O(n·k) scanning) is sound and should be a genuine win for any k>1, and the parallel paths are gated by memory-safety heuristics to avoid regressions from over-allocating thread-local tables or excessive atomic contention; but "at most four times measure-and-improve" could not be exercised, so this artifact is a best-effort, un-benchmarked candidate rather than a confirmed fastest version.