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
