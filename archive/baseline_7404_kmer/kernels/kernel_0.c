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
