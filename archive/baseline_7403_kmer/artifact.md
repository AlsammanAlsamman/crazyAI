## APPROACH

The naive reference is O(n·k): it rebuilds each k-mer code base-by-base from scratch. But consecutive windows overlap in k−1 bases, so the code can be maintained incrementally: `code = ((code << 2) | next_base) & mask`. This turns the whole scan into O(n) regardless of k — the single biggest win, since the standard solution's baked-in assumption ("each k-mer must be built fully from scratch") is unnecessary; a window's code is a cheap update of the previous one.

Two more assumptions get relaxed:
- "the whole sequence must be read once, start to end, in order" — false; overlapping chunks can be scanned independently in parallel as long as each thread rebuilds its own starting code and writes into a private partial table, merged afterward.
- "the count table can only be updated one position's result at a time" — the final merge sums thread-local tables into the global one, which is still one write per (thread,bucket) but decoupled from the sequential scan.

I only take the parallel path when it's actually profitable: enough windows to amortize thread/reduction overhead, and per-thread private tables (`4^k` entries each) fitting a 1 GiB memory budget — otherwise it falls back to the O(n) serial rolling scan, which is already the dominant algorithmic improvement over the naive O(n·k) baseline.

Base decoding uses a 256-entry LUT (built once per call) that reproduces the reference's exact semantics (A=0,C=1,G=2,T=3, default 3), so behavior matches the given naive kernel exactly on any input, not just clean ACGT.

## ARTIFACT

```c
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || k > 32 || n < k) return;

    unsigned char lut[256];
    memset(lut, 3, sizeof(lut));
    lut[(unsigned char)'A'] = 0;
    lut[(unsigned char)'C'] = 1;
    lut[(unsigned char)'G'] = 2;
    lut[(unsigned char)'T'] = 3;

    uint64_t mask = (k == 32) ? ~0ULL : ((1ULL << (2 * k)) - 1);
    long num_windows = (long)n - k + 1;
    uint64_t table_size = 1ULL << (2 * k);

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    if (nthreads > 1) {
        if ((long)nthreads > num_windows) nthreads = (int)(num_windows > 0 ? num_windows : 1);
        if (nthreads < 1) nthreads = 1;
    }

    unsigned long long mem_needed = (unsigned long long)table_size *
                                     (unsigned long long)nthreads * sizeof(uint64_t);
    const unsigned long long MEM_BUDGET = 1ULL << 30; /* 1 GiB */

    if (nthreads > 1 && mem_needed <= MEM_BUDGET && num_windows > 200000) {
        uint64_t *local = (uint64_t *)calloc((size_t)table_size * (size_t)nthreads, sizeof(uint64_t));
        if (local) {
            long chunk = (num_windows + nthreads - 1) / nthreads;
            #pragma omp parallel num_threads(nthreads)
            {
                int t = omp_get_thread_num();
                uint64_t *my = local + (size_t)t * table_size;
                long wstart = (long)t * chunk;
                long wend = wstart + chunk;
                if (wend > num_windows) wend = num_windows;
                if (wstart < wend) {
                    uint64_t code = 0;
                    for (int j = 0; j < k; j++)
                        code = (code << 2) | lut[(unsigned char)seq[wstart + j]];
                    my[code]++;
                    for (long i = wstart + 1; i < wend; i++) {
                        code = ((code << 2) | lut[(unsigned char)seq[i + k - 1]]) & mask;
                        my[code]++;
                    }
                }
            }
            #pragma omp parallel for schedule(static)
            for (long c = 0; c < (long)table_size; c++) {
                uint64_t sum = counts[c];
                for (int t = 0; t < nthreads; t++)
                    sum += local[(size_t)t * table_size + c];
                counts[c] = sum;
            }
            free(local);
            return;
        }
        /* fall through to serial if calloc failed */
    }

    /* Serial O(n) rolling encode */
    uint64_t code = 0;
    for (int j = 0; j < k; j++)
        code = (code << 2) | lut[(unsigned char)seq[j]];
    counts[code]++;
    for (long i = 1; i < num_windows; i++) {
        code = ((code << 2) | lut[(unsigned char)seq[i + k - 1]]) & mask;
        counts[code]++;
    }
}
```

PREDICTION: 10

(Predicting roughly a 10x speedup over the naive O(n·k) baseline, driven mostly by the O(n) rolling-code algorithmic change, with additional gains on large inputs from the parallel reduction path when it's memory-feasible.)

## MEASUREMENT

No `kmer_bench` tool call is available in this session, so I cannot execute a real measurement. Reporting this plainly rather than fabricating numbers: the PREDICTION above is untested against the harness.

## VERDICT

Unverified. The implementation is algorithmically correct (matches the reference LUT semantics exactly, O(n) instead of O(n·k), with a memory-budget-gated OpenMP reduction path that falls back safely to serial when parallelism isn't safe/profitable), but without a `kmer_bench` run in this session I cannot confirm the predicted ~10x speedup or catch any runtime surprises (e.g., thread/memory-budget thresholds being miscalibrated for the actual benchmark's n/k). Recommend running `kmer_bench` before trusting the prediction.