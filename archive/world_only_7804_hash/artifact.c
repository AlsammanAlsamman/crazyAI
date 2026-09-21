#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

#define PRIME1 11400714785074694791ULL
#define PRIME2 14029467366897019727ULL
#define PRIME3 1609587929392839161ULL
#define PRIME4 9650029242287828579ULL
#define PRIME5 2870177450012600261ULL

static inline uint64_t read_u64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static uint64_t xxh64_block(const unsigned char *data, size_t len, uint64_t seed) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = seed + PRIME1 + PRIME2;
        uint64_t v2 = seed + PRIME2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME1;

        do {
            v1 += read_u64(p) * PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1; p += 8;
            v2 += read_u64(p) * PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1; p += 8;
            v3 += read_u64(p) * PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1; p += 8;
            v4 += read_u64(p) * PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1; p += 8;
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1; h64 ^= v1; h64 = h64 * PRIME1 + PRIME4;
        v2 *= PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1; h64 ^= v2; h64 = h64 * PRIME1 + PRIME4;
        v3 *= PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1; h64 ^= v3; h64 = h64 * PRIME1 + PRIME4;
        v4 *= PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1; h64 ^= v4; h64 = h64 * PRIME1 + PRIME4;
    } else {
        h64 = seed + PRIME5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = read_u64(p) * PRIME2;
        k1 = rotl64(k1, 31) * PRIME1;
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME1 + PRIME4;
        p += 8;
    }
    if (p + 4 <= end) {
        uint32_t v32;
        memcpy(&v32, p, 4);
        h64 ^= (uint64_t)v32 * PRIME1;
        h64 = rotl64(h64, 23) * PRIME2 + PRIME3;
        p += 4;
    }
    while (p < end) {
        h64 ^= (uint64_t)(*p) * PRIME5;
        h64 = rotl64(h64, 11) * PRIME1;
        p++;
    }

    h64 ^= h64 >> 33;
    h64 *= PRIME2;
    h64 ^= h64 >> 29;
    h64 *= PRIME3;
    h64 ^= h64 >> 32;

    return h64;
}

uint64_t kernel(const unsigned char *data, size_t len) {
#ifdef _OPENMP
    const size_t PAR_THRESHOLD = 1u << 20; /* 1 MiB */
    const int NCHUNKS = 8;
    if (len >= PAR_THRESHOLD) {
        uint64_t partial[8];
        size_t base = len / NCHUNKS;
        size_t rem = len % NCHUNKS;
        size_t offsets[9];
        offsets[0] = 0;
        for (int i = 0; i < NCHUNKS; i++) {
            size_t sz = base + (size_t)(i < (int)rem ? 1 : 0);
            offsets[i + 1] = offsets[i] + sz;
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NCHUNKS; i++) {
            partial[i] = xxh64_block(data + offsets[i], offsets[i + 1] - offsets[i],
                                      0x9E3779B97F4A7C15ULL + (uint64_t)i * PRIME5);
        }
        uint64_t h = PRIME5 + (uint64_t)len;
        for (int i = 0; i < NCHUNKS; i++) {
            h ^= partial[i];
            h = rotl64(h, 27) * PRIME1 + PRIME4;
        }
        h ^= h >> 33; h *= PRIME2; h ^= h >> 29; h *= PRIME3; h ^= h >> 32;
        return h;
    }
#endif
    return xxh64_block(data, len, 0);
}
