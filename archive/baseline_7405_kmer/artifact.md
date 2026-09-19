# APPROACH

The naive solution's four "load-bearing" assumptions are all removable:

1. **"Each k-mer must be built from scratch"** — false. Consecutive k-mers overlap in k-1 bases, so the code can be maintained incrementally: `code = ((code << 2) | base) & mask`. This turns the O(n·k) inner loop into O(n) total work, examining each base exactly once instead of k times.
2. **"One window discarded before the next begins"** — the rolling code carries state across windows, so there's no re-scan.
3. **"Table updated one position at a time / sequence read once in order"** — I keep sequential-order semantics per thread (required for correctness of the rolling state) but split the sequence into independent contiguous ranges across threads, each maintaining its own rolling code (bootstrapped from the k-1 bases preceding its range) and its own private count table, merged at the end. This removes the single-thread constraint without violating per-thread ordering.
4. Memory safety valve: private per-thread tables are only used when `4^k` is small enough that N_threads private copies stay within a fixed budget (256MB) and there's enough work per thread (≥4096 windows) to amortize thread launch/reduction cost; otherwise it falls back to the single-thread O(n) rolling pass, which is still asymptotically superior to the naive O(n·k) baseline.
5. Base decoding uses a precomputed 256-entry lookup table instead of a branchy `switch`, matching the reference's default-to-3 behavior exactly (including for `T` and any non-ACG byte).

# ARTIFACT

```c
#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Matches reference: 'A'->0, 'C'->1, 'G'->2, everything else (incl. 'T') -> 3 */
static const unsigned char BASE_CODE[256] = {
    [0 ... 255] = 3,
    ['A'] = 0, ['C'] = 1, ['G'] = 2
};

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0) {
        if (k == 0) {
            long long windows = (long long)n - (long long)k + 1;
            if (windows > 0) counts[0] = (uint64_t)windows;
        }
        return;
    }

    long long total_windows = (long long)n - (long long)k + 1;
    if (total_windows <= 0) return;

    uint64_t mask = (k >= 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);
    uint64_t table_size = (k >= 32) ? 0 : (1ULL << (2 * k));

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif

    int nthreads = 1;
    if (max_threads > 1 && k < 32 && table_size > 0) {
        const uint64_t MAX_PRIVATE_BYTES = 256ULL * 1024 * 1024;
        uint64_t table_bytes = table_size * (uint64_t)sizeof(uint64_t);
        long long by_mem = (long long)(table_bytes > 0 ? (MAX_PRIVATE_BYTES / table_bytes) : 0);
        long long by_work = total_windows / 4096;
        long long cap = max_threads;
        if (by_mem < cap) cap = by_mem;
        if (by_work < cap) cap = by_work;
        if (cap > max_threads) cap = max_threads;
        if (cap < 1) cap = 1;
        nthreads = (int)cap;
    }

    if (nthreads <= 1) {
        uint64_t code = 0;
        for (int j = 0; j < k - 1; j++)
            code = (code << 2) | BASE_CODE[(unsigned char)seq[j]];
        const char *p = seq + (k - 1);
        for (long long i = 0; i < total_windows; i++) {
            code = ((code << 2) | BASE_CODE[(unsigned char)p[i]]) & mask;
            counts[code]++;
        }
        return;
    }

#ifdef _OPENMP
    uint64_t **priv = (uint64_t **)malloc(sizeof(uint64_t *) * (size_t)nthreads);
    priv[0] = counts; /* already zeroed by caller */
    for (int t = 1; t < nthreads; t++)
        priv[t] = (uint64_t *)calloc((size_t)table_size, sizeof(uint64_t));

    #pragma omp parallel num_threads(nthreads)
    {
        int t = omp_get_thread_num();
        long long lo = (total_windows * t) / nthreads;
        long long hi = (total_windows * (t + 1)) / nthreads;
        uint64_t *my_counts = priv[t];
        if (hi > lo) {
            uint64_t code = 0;
            for (long long j = 0; j < k - 1; j++)
                code = (code << 2) | BASE_CODE[(unsigned char)seq[lo + j]];
            const char *p = seq + (lo + (k - 1));
            for (long long i = 0; i < hi - lo; i++) {
                code = ((code << 2) | BASE_CODE[(unsigned char)p[i]]) & mask;
                my_counts[code]++;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (long long c = 0; c < (long long)table_size; c++) {
        uint64_t sum = counts[c];
        for (int t = 1; t < nthreads; t++)
            sum += priv[t][c];
        counts[c] = sum;
    }

    for (int t = 1; t < nthreads; t++)
        free(priv[t]);
    free(priv);
#endif
}
```

PREDICTION: 10

# MEASUREMENT

No tool execution is available in this session (explicit environment note: "No tools are available in this session... the pipeline will compile and measure it"), so I could not invoke `kmer_bench` myself. The prediction above (≈10×) is a reasoned estimate based on: naive cost ∝ n·k vs. rolling cost ∝ n (a factor-of-k algorithmic win for any k beyond trivial size, e.g. k=8–16 gives 8–16× on the encode step alone from algorithmic complexity change), moderated downward for k where the table is small (little multithreading benefit, cache-resident anyway so naive isn't catastrophically slow) and for very large k where private-table multithreading is disabled by the memory guard and only the O(n) single-thread win applies. Actual measured speedup should be reported by the pipeline against this stated prediction.

# VERDICT

Correctness: the rolling-hash update is a standard, provably equivalent reformulation of the reference's from-scratch encoding (both compute the same base-2-bits-high-first packing for every window; verified by construction — the low 2k bits of the incrementally shifted-and-masked code equal the freshly-built code for the same window), and the lookup table reproduces the reference's exact char→code mapping including its default-to-3 fallback. The threaded variant only changes *how* work is scheduled (independent contiguous ranges with correctly bootstrapped rolling state), not *what* is computed, and results are summed exactly via the reduction pass — so counts should be bit-identical to the reference regardless of thread count. Performance verdict is pending the pipeline's actual `kmer_bench` run against the stated prediction.