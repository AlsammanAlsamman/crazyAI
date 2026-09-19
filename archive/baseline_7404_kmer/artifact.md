# APPROACH

The naive kernel is O(n·k): every window re-derives its 2k-bit code from scratch, discarding all overlap with the previous window even though k-1 of the k bases are shared. The contract only fixes the *table semantics* (code = k bases packed 2 bits each, high base first, 4^k-entry pre-zeroed table) — it says nothing about how the code has to be produced or how the table has to be updated. So every one of the "assumptions" the naive solution makes is droppable:

1. **Rolling code, O(n) total**: maintain `code` across the scan; each step does `code = ((code<<2) | base) & mask` instead of re-reading k bases. This alone removes the factor of k.
2. **Branchless byte→code lookup**: a 256-entry table built once, instead of a `switch` per base, so the inner loop is a straight-line load/shift/or/mask/increment — trivially unrolled/vectorizable by the compiler and free of branch mispredictions.
2 and 1 together turn the "each k-mer built from scratch" and "every base contributes independently" assumptions into "each base contributes one O(1) rolling update, shared across all subsequent windows."
3. **OpenMP data-parallelism**: split the range of window-start positions into contiguous chunks per thread. Each thread rolls its own code independently (needs only k-1 bootstrap bases at its chunk start — O(k) one-time cost, negligible) and writes into a **private** local histogram to avoid write contention/false sharing on `counts`. Local histograms are reduced into the shared table once per thread at the end. This drops the "read the sequence once, in order, single stream" and "update the table one position at a time" assumptions.
4. **Safety gate on parallel path**: local histograms cost 4^k·8 bytes per thread. For large k this would blow up memory, so parallel local-histogram mode is only used when the per-thread table is small (≤64 MiB) and there's enough work (>200k windows) to amortize thread/merge overhead; otherwise it falls back to the single-threaded O(n) rolling scan (still far faster than naive), with an atomic-increment fallback if a local table's calloc fails.

# ARTIFACT

```c
#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || n < k) return;

    unsigned char code_of[256];
    for (int c = 0; c < 256; c++) code_of[c] = 3; /* same default as reference for non-ACGT */
    code_of[(unsigned char)'A'] = 0;
    code_of[(unsigned char)'C'] = 1;
    code_of[(unsigned char)'G'] = 2;
    code_of[(unsigned char)'T'] = 3;

    int shift = 2 * k;
    uint64_t mask = (shift >= 64) ? ~0ULL : ((1ULL << shift) - 1ULL);
    int64_t num_windows = (int64_t)n - (int64_t)k + 1;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#endif

    int table_ok = (shift < 63);
    uint64_t table_entries = table_ok ? (1ULL << shift) : 0ULL;
    size_t table_bytes = table_ok ? (size_t)table_entries * sizeof(uint64_t) : (size_t)-1;

    int use_parallel = (nthreads > 1) && (num_windows > 200000) && table_ok &&
                        (table_bytes <= (64ULL << 20));

    if (!use_parallel) {
        uint64_t code = 0;
        for (int j = 0; j < k - 1; j++) code = (code << 2) | code_of[(unsigned char)seq[j]];
        for (int64_t i = 0; i < num_windows; i++) {
            code = ((code << 2) | code_of[(unsigned char)seq[i + k - 1]]) & mask;
            counts[code]++;
        }
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int64_t chunk = (num_windows + nt - 1) / nt;
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > num_windows) end = num_windows;

        if (start < end) {
            uint64_t *local = (uint64_t*)calloc((size_t)table_entries, sizeof(uint64_t));
            if (local) {
                uint64_t code = 0;
                for (int j = 0; j < k - 1; j++)
                    code = (code << 2) | code_of[(unsigned char)seq[start + j]];
                for (int64_t i = start; i < end; i++) {
                    code = ((code << 2) | code_of[(unsigned char)seq[i + k - 1]]) & mask;
                    local[code]++;
                }
                #pragma omp critical
                {
                    for (uint64_t c = 0; c < table_entries; c++)
                        counts[c] += local[c];
                }
                free(local);
            } else {
                uint64_t code = 0;
                for (int j = 0; j < k - 1; j++)
                    code = (code << 2) | code_of[(unsigned char)seq[start + j]];
                for (int64_t i = start; i < end; i++) {
                    code = ((code << 2) | code_of[(unsigned char)seq[i + k - 1]]) & mask;
                    #pragma omp atomic
                    counts[code]++;
                }
            }
        }
    }
#endif
}
```

PREDICTION: 12

# MEASUREMENT

No `kmer_bench`/`kmer_contract` (or any other benchmarking) tool was actually available in this session — the tool list I was given contains only Claude Docs, Gmail, Google Drive, and Slack tools, none of which can compile or run C. I therefore could not execute the harness myself and have no measured numbers to report. The prediction above (≈12× over the naive `-O3 -march=native` baseline) is reasoning-only, based on: removing the redundant O(k) re-encode per window (the dominant factor for realistic k in the 8–16 range) plus a modest additional multiplier from OpenMP parallelism when the per-thread histogram fits comfortably in cache/RAM. The actual figure will depend on the harness's n, k, and core count, which I do not know.

# VERDICT

Correctness: the kernel is a straightforward algebraic rewrite of the reference (rolling 2-bit-per-base code instead of full re-encode, identical default-to-3 mapping for non-ACGT bytes, identical code packing high-base-first) plus a standard partition/private-accumulate/reduce parallelization — both paths should produce bit-identical counts to the naive reference for any valid input, and I've checked chunk-boundary bootstrap indices stay in bounds. Performance claim is **unverified in this session** since no measurement tool was actually accessible despite the problem statement listing one; I'm reporting that plainly rather than fabricating a MEASUREMENT result.