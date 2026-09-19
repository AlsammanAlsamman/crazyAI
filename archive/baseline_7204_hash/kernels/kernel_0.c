#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define PRIME64_1 11400714785074694791ULL
#define PRIME64_2 14029467366897019727ULL
#define PRIME64_3 1609587929392839161ULL
#define PRIME64_4 9650029242287828579ULL
#define PRIME64_5 2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t read64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t mergeRound64(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

static inline uint64_t avalanche64(uint64_t h) {
    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

/* Core xxHash64-style mixer: word-at-a-time, 4 parallel lanes. */
static uint64_t xxh64_core(const unsigned char *data, size_t len, uint64_t seed) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - PRIME64_1;

        do {
            v1 = round64(v1, read64(p)); p += 8;
            v2 = round64(v2, read64(p)); p += 8;
            v3 = round64(v3, read64(p)); p += 8;
            v4 = round64(v4, read64(p)); p += 8;
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = mergeRound64(h64, v1);
        h64 = mergeRound64(h64, v2);
        h64 = mergeRound64(h64, v3);
        h64 = mergeRound64(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = round64(0, read64(p));
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        uint32_t v;
        memcpy(&v, p, 4);
        h64 ^= (uint64_t)v * PRIME64_1;
        h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }
    while (p < end) {
        h64 ^= (uint64_t)(*p) * PRIME64_5;
        h64 = rotl64(h64, 11) * PRIME64_1;
        p++;
    }

    return avalanche64(h64);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* Deterministic fixed fan-out for large buffers only; small/medium
       buffers go straight through the single-threaded core, which is
       already far cheaper per byte than a byte-at-a-time multiply. */
    if (len >= (1u << 20)) {
        enum { NSEG = 4 };
        size_t seg_len = len / NSEG;
        seg_len -= seg_len % 32;           /* keep segment boundaries clean */
        if (seg_len == 0) seg_len = len;   /* defensive; unreachable at this size */

        size_t offs[NSEG], lens[NSEG];
        for (int i = 0; i < NSEG; i++) {
            offs[i] = (size_t)i * seg_len;
            lens[i] = (i == NSEG - 1) ? (len - offs[i]) : seg_len;
        }

        uint64_t hashes[NSEG];
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NSEG; i++) {
            uint64_t seed = (uint64_t)i * PRIME64_5 + 0x9E3779B97F4A7C15ULL;
            hashes[i] = xxh64_core(data + offs[i], lens[i], seed);
        }

        uint64_t acc = PRIME64_1 ^ (uint64_t)len;
        for (int i = 0; i < NSEG; i++) {
            acc ^= hashes[i];
            acc = rotl64(acc, 27) * PRIME64_1 + PRIME64_4;
        }
        return avalanche64(acc);
    }

    return xxh64_core(data, len, 0);
}
