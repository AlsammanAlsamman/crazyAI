#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static unsigned char g_b2c[256];
static int g_b2c_init = 0;

static void init_b2c(void) {
    for (int i = 0; i < 256; i++) g_b2c[i] = 3;
    g_b2c[(unsigned char)'A'] = 0;
    g_b2c[(unsigned char)'C'] = 1;
    g_b2c[(unsigned char)'G'] = 2;
    g_b2c_init = 1;
}

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (!g_b2c_init) init_b2c();
    if (k <= 0 || n < k) return;

    long total = (long)n - (long)k + 1; /* number of windows */
    uint64_t mask, entries;
    if (k >= 32) { mask = ~0ULL; entries = 0; /* not realistically reachable */ }
    else { mask = (1ULL << (2 * k)) - 1; entries = 1ULL << (2 * k); }

    const unsigned char *ub2c = g_b2c;
    const unsigned char *useq = (const unsigned char *)seq;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#endif
    if (total < 200000L) nthreads = 1; /* not worth parallel overhead */

    if (nthreads <= 1) {
        uint64_t code = 0;
        for (int j = 0; j < k - 1; j++) code = (code << 2) | ub2c[useq[j]];
        for (long i = 0; i < total; i++) {
            code = ((code << 2) | ub2c[useq[i + k - 1]]) & mask;
            counts[code]++;
        }
        return;
    }

    const uint64_t BUDGET_BYTES = (uint64_t)256 * 1024 * 1024; /* 256MB extra budget */
    int use_private = (k < 32) &&
        (entries <= (BUDGET_BYTES / (uint64_t)sizeof(uint64_t) / (uint64_t)nthreads));

    if (use_private) {
        uint64_t **locals = (uint64_t **)malloc(sizeof(uint64_t *) * (size_t)nthreads);
        for (int t = 0; t < nthreads; t++)
            locals[t] = (uint64_t *)calloc((size_t)entries, sizeof(uint64_t));

#ifdef _OPENMP
        #pragma omp parallel num_threads(nthreads)
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            long chunk = (total + nthreads - 1) / nthreads;
            long start = (long)tid * chunk;
            long end = start + chunk;
            if (end > total) end = total;
            if (start < end) {
                uint64_t *lc = locals[tid];
                uint64_t code = 0;
                for (int j = 0; j < k; j++) code = (code << 2) | ub2c[useq[start + j]];
                code &= mask;
                lc[code]++;
                for (long i = start + 1; i < end; i++) {
                    code = ((code << 2) | ub2c[useq[i + k - 1]]) & mask;
                    lc[code]++;
                }
            }
        }

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long c = 0; c < (long)entries; c++) {
            uint64_t sum = 0;
            for (int t = 0; t < nthreads; t++) sum += locals[t][c];
            counts[c] = sum;
        }
        for (int t = 0; t < nthreads; t++) free(locals[t]);
        free(locals);
    } else {
#ifdef _OPENMP
        #pragma omp parallel num_threads(nthreads)
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            long chunk = (total + nthreads - 1) / nthreads;
            long start = (long)tid * chunk;
            long end = start + chunk;
            if (end > total) end = total;
            if (start < end) {
                uint64_t code = 0;
                for (int j = 0; j < k; j++) code = (code << 2) | ub2c[useq[start + j]];
                code &= mask;
#ifdef _OPENMP
                #pragma omp atomic
#endif
                counts[code]++;
                for (long i = start + 1; i < end; i++) {
                    code = ((code << 2) | ub2c[useq[i + k - 1]]) & mask;
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    counts[code]++;
                }
            }
        }
    }
}
