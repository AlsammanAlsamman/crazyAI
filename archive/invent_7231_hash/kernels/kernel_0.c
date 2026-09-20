#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define HK_PRIME1 0x9E3779B185EBCA87ULL
#define HK_PRIME2 0xC2B2AE3D27D4EB4FULL
#define HK_PRIME3 0x165667B19E3779F9ULL
#define HK_PRIME4 0x85EBCA77C2B2AE63ULL
#define HK_PRIME5 0x27D4EB2F165667C5ULL

static inline uint64_t hk_rotl(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* final avalanche: every input bit must be able to flip every output bit */
static inline uint64_t hk_mix(uint64_t h) {
    h ^= h >> 33;
    h *= HK_PRIME2;
    h ^= h >> 29;
    h *= HK_PRIME3;
    h ^= h >> 32;
    return h;
}

/* hash one contiguous run of bytes under a given seed/brand */
static uint64_t hk_hash_range(const unsigned char *p, size_t n, uint64_t seed) {
    uint64_t h = seed ^ HK_PRIME5 ^ ((uint64_t)n * HK_PRIME1);

    while (n >= 8) {
        uint64_t k;
        memcpy(&k, p, 8);
        k *= HK_PRIME1; k = hk_rotl(k, 31); k *= HK_PRIME2;
        h ^= k;
        h = hk_rotl(h, 27) * HK_PRIME1 + HK_PRIME4;
        p += 8; n -= 8;
    }
    if (n >= 4) {
        uint32_t k32;
        memcpy(&k32, p, 4);
        h ^= (uint64_t)k32 * HK_PRIME1;
        h = hk_rotl(h, 23) * HK_PRIME2 + HK_PRIME3;
        p += 4; n -= 4;
    }
    while (n > 0) {
        h ^= (uint64_t)(*p) * HK_PRIME5;
        h = hk_rotl(h, 11) * HK_PRIME1;
        p++; n--;
    }
    return hk_mix(h);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* guard the small season: below this, calling other hands costs more
       than it saves, so press it alone */
    const size_t PARALLEL_THRESHOLD = 1u << 20; /* 1 MiB */

#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
#else
    int max_threads = 1;
#endif

    if (len < PARALLEL_THRESHOLD || max_threads <= 1) {
        return hk_hash_range(data, len, 1469598103934665603ULL);
    }

    int nchunks = max_threads;
    if ((size_t)nchunks > len) nchunks = (int)(len > 0 ? len : 1);

    size_t base = len / (size_t)nchunks;
    size_t rem  = len % (size_t)nchunks;

    uint64_t *partial = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)nchunks);
    if (!partial) {
        return hk_hash_range(data, len, 1469598103934665603ULL);
    }

    #pragma omp parallel for schedule(static) num_threads(nchunks)
    for (int i = 0; i < nchunks; i++) {
        size_t start    = (size_t)i * base + (size_t)((size_t)i < rem ? (size_t)i : rem);
        size_t this_len = base + ((size_t)i < rem ? 1 : 0);
        uint64_t seed = 1469598103934665603ULL ^ (HK_PRIME1 * (uint64_t)(i + 1));
        partial[i] = hk_hash_range(data + start, this_len, seed);
    }

    /* fold the gathers back together in dammed order, not finish order */
    uint64_t h = HK_PRIME5 ^ ((uint64_t)len * HK_PRIME2);
    for (int i = 0; i < nchunks; i++) {
        h ^= partial[i];
        h = hk_rotl(h, 29) * HK_PRIME1 + (uint64_t)i;
    }
    h = hk_mix(h);

    free(partial);
    return h;
}
