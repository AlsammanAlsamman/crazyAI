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
