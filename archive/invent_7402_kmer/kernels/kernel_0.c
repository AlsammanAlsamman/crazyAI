#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Every "bead" (base) is read once into a 2-bit color via this peg-table. */
static void build_lut(unsigned char lut[256]) {
    for (int i = 0; i < 256; i++) lut[i] = 3; /* default: dent-poison / T */
    lut[(unsigned char)'A'] = 0; /* shell */
    lut[(unsigned char)'C'] = 1; /* flower-plant */
    lut[(unsigned char)'G'] = 2; /* plank-grain */
    lut[(unsigned char)'T'] = 3; /* dent-poison */
}

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || n < k) return;

    unsigned char lut[256];
    build_lut(lut);

    const long long total_windows = (long long)n - (long long)k + 1;
    const uint64_t mask      = (k >= 32) ? ~0ULL : ((1ULL << (2 * k)) - 1);
    const uint64_t num_codes = (k >= 32) ? ~0ULL : (1ULL << (2 * k));

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
    if ((long long)nthreads > total_windows) nthreads = (int)total_windows;
    if (nthreads < 1) nthreads = 1;
#endif

    /* Give each crew its own courtyard (thread-local histogram) only when
       that courtyard is cheap enough to build (<=32MB); otherwise share
       one courtyard and use an atomic tally-notch. */
    int use_local = (nthreads > 1) && (num_codes <= (1ULL << 22));

#ifdef _OPENMP
    if (use_local) {
        uint64_t *local = (uint64_t *)calloc((size_t)nthreads * (size_t)num_codes, sizeof(uint64_t));
        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            uint64_t *my = local + (size_t)tid * (size_t)num_codes;
            long long chunk = (total_windows + nthreads - 1) / nthreads;
            long long start = (long long)tid * chunk;
            long long end = start + chunk;
            if (end > total_windows) end = total_windows;
            if (start < end) {
                uint64_t code = 0;
                for (int j = 0; j < k; j++)
                    code = (code << 2) | lut[(unsigned char)seq[start + j]];
                my[code]++;
                for (long long i = start + 1; i < end; i++) {
                    code = ((code << 2) | lut[(unsigned char)seq[i + k - 1]]) & mask;
                    my[code]++;
                }
            }
        }
        for (int t = 0; t < nthreads; t++) {
            uint64_t *my = local + (size_t)t * (size_t)num_codes;
            for (uint64_t c = 0; c < num_codes; c++) counts[c] += my[c];
        }
        free(local);
        return;
    }

    if (nthreads > 1) {
        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            long long chunk = (total_windows + nthreads - 1) / nthreads;
            long long start = (long long)tid * chunk;
            long long end = start + chunk;
            if (end > total_windows) end = total_windows;
            if (start < end) {
                uint64_t code = 0;
                for (int j = 0; j < k; j++)
                    code = (code << 2) | lut[(unsigned char)seq[start + j]];
                __atomic_fetch_add(&counts[code], 1, __ATOMIC_RELAXED);
                for (long long i = start + 1; i < end; i++) {
                    code = ((code << 2) | lut[(unsigned char)seq[i + k - 1]]) & mask;
                    __atomic_fetch_add(&counts[code], 1, __ATOMIC_RELAXED);
                }
            }
        }
        return;
    }
#endif

    /* Single oar-frame, single strand-walk: build the first window from
       scratch once, then slide it one bead at a time for the rest. */
    uint64_t code = 0;
    for (int j = 0; j < k; j++) code = (code << 2) | lut[(unsigned char)seq[j]];
    counts[code]++;
    for (long long i = 1; i < total_windows; i++) {
        code = ((code << 2) | lut[(unsigned char)seq[i + k - 1]]) & mask;
        counts[code]++;
    }
}
