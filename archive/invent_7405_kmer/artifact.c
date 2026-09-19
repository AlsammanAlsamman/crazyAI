#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline int base_code(char c) {
    switch (c) { case 'A': return 0; case 'C': return 1; case 'G': return 2; default: return 3; }
}

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || k > 31 || n < k) return;

    uint64_t mask = (1ULL << (2 * k)) - 1ULL;
    uint64_t table_size = 1ULL << (2 * k);
    long long total_windows = (long long)n - k + 1;

#ifdef _OPENMP
    /* Only fan out to private "deserts" when there are enough windows to
       amortize the setup, and each private table is small enough that
       duplicating it per thread doesn't blow up memory (SEED 2: mounds
       kept apart, reconciled only when the rain comes). */
    int nthreads = omp_get_max_threads();
    if (nthreads > 1 && total_windows > 4LL * k && table_size <= (1ULL << 24)) {
        if ((long long)nthreads > total_windows) nthreads = (int)total_windows;
        uint64_t **partial = (uint64_t **)malloc(sizeof(uint64_t *) * nthreads);
        for (int t = 0; t < nthreads; t++)
            partial[t] = (uint64_t *)calloc((size_t)table_size, sizeof(uint64_t));

        #pragma omp parallel num_threads(nthreads)
        {
            int t = omp_get_thread_num();
            long long chunk = (total_windows + nthreads - 1) / nthreads;
            long long start = (long long)t * chunk;
            long long end = start + chunk;
            if (end > total_windows) end = total_windows;
            if (start < end) {
                uint64_t *local = partial[t];
                uint64_t code = 0;
                /* Prime the running crest with the k-1 beads just before
                   this thread's first window; after that we never look
                   back at a raw bead again. */
                for (long long j = 0; j < k - 1; j++)
                    code = ((code << 2) | (uint64_t)base_code(seq[start + j])) & mask;
                for (long long w = start; w < end; w++) {
                    code = ((code << 2) | (uint64_t)base_code(seq[w + k - 1])) & mask;
                    local[code]++;
                }
            }
        }
        for (int t = 0; t < nthreads; t++) {
            uint64_t *local = partial[t];
            for (uint64_t c = 0; c < table_size; c++) counts[c] += local[c];
            free(local);
        }
        free(partial);
        return;
    }
#endif
    {
        uint64_t code = 0;
        int i;
        for (i = 0; i < k - 1; i++)
            code = ((code << 2) | (uint64_t)base_code(seq[i])) & mask;
        for (i = k - 1; i < n; i++) {
            code = ((code << 2) | (uint64_t)base_code(seq[i])) & mask;
            counts[code]++;
        }
    }
}
