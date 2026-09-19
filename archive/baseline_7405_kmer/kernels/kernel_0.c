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
