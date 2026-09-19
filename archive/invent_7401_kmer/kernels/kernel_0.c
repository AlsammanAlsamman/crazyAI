#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline int base_code(char c) {
    switch (c) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        default:  return 3; /* 'T' and anything else */
    }
}

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || n < k) return;

    const long long num_windows = (long long)n - k + 1;
    if (num_windows <= 0) return;

    const int bits = 2 * k;
    const uint64_t mask = (bits >= 64) ? ~0ULL : ((1ULL << bits) - 1ULL);
    const uint64_t num_buckets = (bits >= 64) ? 0ULL : (1ULL << bits);

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    if (max_threads < 1) max_threads = 1;

    /* Cap total private-board memory to ~512MB so large k can't blow up RAM. */
    uint64_t bytes_per_board = (bits < 64) ? num_buckets * (uint64_t)sizeof(uint64_t)
                                            : ((uint64_t)1 << 62);
    long long budget_threads = (long long)((512ULL * 1024 * 1024) /
                                            (bytes_per_board ? bytes_per_board : 1ULL));
    if (budget_threads < 1) budget_threads = 1;

    int nthreads = max_threads;
    if ((long long)nthreads > budget_threads) nthreads = (int)budget_threads;
    if ((long long)nthreads > num_windows) nthreads = (int)(num_windows < 1 ? 1 : num_windows);
    if (nthreads < 1) nthreads = 1;

    if (nthreads == 1 || bits >= 64) {
        /* One rail, one window, one board: pure rolling update. */
        uint64_t code = 0;
        for (int j = 0; j < k; j++) code = (code << 2) | (uint64_t)base_code(seq[j]);
        counts[code & mask]++;
        for (long long i = 1; i < num_windows; i++) {
            code = ((code << 2) | (uint64_t)base_code(seq[i + k - 1])) & mask;
            counts[code]++;
        }
        return;
    }

#ifdef _OPENMP
    uint64_t **boards = (uint64_t **)malloc(sizeof(uint64_t *) * (size_t)nthreads);
    for (int t = 0; t < nthreads; t++)
        boards[t] = (uint64_t *)calloc((size_t)num_buckets, sizeof(uint64_t));

    #pragma omp parallel num_threads(nthreads)
    {
        int t = omp_get_thread_num();
        uint64_t *board = boards[t];

        long long chunk = (num_windows + nthreads - 1) / nthreads;
        long long start = (long long)t * chunk;
        long long end = start + chunk;
        if (end > num_windows) end = num_windows;

        if (start < end) {
            uint64_t code = 0;
            for (int j = 0; j < k; j++) code = (code << 2) | (uint64_t)base_code(seq[start + j]);
            board[code & mask]++;
            for (long long i = start + 1; i < end; i++) {
                code = ((code << 2) | (uint64_t)base_code(seq[i + k - 1])) & mask;
                board[code]++;
            }
        }
    }

    /* Walk every board, hook by hook, exactly once, after the strand is spent. */
    for (uint64_t code = 0; code < num_buckets; code++) {
        uint64_t sum = 0;
        for (int t = 0; t < nthreads; t++) sum += boards[t][code];
        counts[code] = sum;
    }

    for (int t = 0; t < nthreads; t++) free(boards[t]);
    free(boards);
#endif
}
