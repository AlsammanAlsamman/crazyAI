#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define XXH_PRIME64_1 0x9E3779B185EBCA87ULL
#define XXH_PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define XXH_PRIME64_3 0x165667B19E3779F9ULL
#define XXH_PRIME64_4 0x85EBCA77C2B2AE63ULL
#define XXH_PRIME64_5 0x27D4EB2F165667C5ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t xxh64_round(uint64_t acc, uint64_t input) {
    acc += input * XXH_PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= XXH_PRIME64_1;
    return acc;
}

static inline uint64_t xxh64_merge_round(uint64_t acc, uint64_t val) {
    val = xxh64_round(0, val);
    acc ^= val;
    acc = acc * XXH_PRIME64_1 + XXH_PRIME64_4;
    return acc;
}

/* Core serial xxHash64-style hash over a contiguous range with given seed. */
static uint64_t hash_range(const unsigned char *data, size_t len, uint64_t seed) {
    const unsigned char *p = data;
    const unsigned char *bEnd = data + len;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *limit = bEnd - 32;
        uint64_t v1 = seed + XXH_PRIME64_1 + XXH_PRIME64_2;
        uint64_t v2 = seed + XXH_PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - XXH_PRIME64_1;

        do {
            uint64_t k1, k2, k3, k4;
            memcpy(&k1, p, 8); p += 8;
            memcpy(&k2, p, 8); p += 8;
            memcpy(&k3, p, 8); p += 8;
            memcpy(&k4, p, 8); p += 8;
            v1 = xxh64_round(v1, k1);
            v2 = xxh64_round(v2, k2);
            v3 = xxh64_round(v3, k3);
            v4 = xxh64_round(v4, k4);
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = seed + XXH_PRIME64_5;
    }

    h64 += (uint64_t)len;

    while ((size_t)(bEnd - p) >= 8) {
        uint64_t k1;
        memcpy(&k1, p, 8);
        k1 = xxh64_round(0, k1);
        h64 ^= k1;
        h64 = rotl64(h64, 27) * XXH_PRIME64_1 + XXH_PRIME64_4;
        p += 8;
    }
    if ((size_t)(bEnd - p) >= 4) {
        uint32_t k1;
        memcpy(&k1, p, 4);
        h64 ^= (uint64_t)k1 * XXH_PRIME64_1;
        h64 = rotl64(h64, 23) * XXH_PRIME64_2 + XXH_PRIME64_3;
        p += 4;
    }
    while (p < bEnd) {
        h64 ^= (uint64_t)(*p) * XXH_PRIME64_5;
        h64 = rotl64(h64, 11) * XXH_PRIME64_1;
        p++;
    }

    h64 ^= h64 >> 33;
    h64 *= XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= XXH_PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}

#define PAR_THRESHOLD ((size_t)1 << 20)   /* 1 MiB: below this, go serial */
#define MIN_CHUNK     ((size_t)1 << 16)   /* 64 KiB minimum chunk size */
#define MAX_LANES     64

uint64_t kernel(const unsigned char *data, size_t len) {
    if (len < PAR_THRESHOLD) {
        return hash_range(data, len, 0);
    }

    int maxT = 1;
#ifdef _OPENMP
    maxT = omp_get_max_threads();
    if (maxT < 1) maxT = 1;
#endif

    size_t maxChunks = len / MIN_CHUNK;
    if (maxChunks < 1) maxChunks = 1;

    size_t T = (size_t)maxT;
    if (T > maxChunks) T = maxChunks;
    if (T > MAX_LANES) T = MAX_LANES;
    if (T < 1) T = 1;

    uint64_t partial[MAX_LANES];
    size_t base = len / T;

    {
        long long Ti = (long long)T;
        #pragma omp parallel for schedule(static) if(T > 1)
        for (long long i = 0; i < Ti; i++) {
            size_t start = (size_t)i * base;
            size_t end = ((size_t)i == T - 1) ? len : (start + base);
            uint64_t seed = XXH_PRIME64_1 * (uint64_t)(i + 1) ^ (uint64_t)len;
            partial[i] = hash_range(data + start, end - start, seed);
        }
    }

    uint64_t h64 = (uint64_t)len ^ XXH_PRIME64_5;
    for (size_t i = 0; i < T; i++) {
        uint64_t k1 = xxh64_round(0, partial[i]);
        h64 ^= k1;
        h64 = rotl64(h64, 27) * XXH_PRIME64_1 + XXH_PRIME64_4;
    }

    h64 ^= h64 >> 33;
    h64 *= XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= XXH_PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}
