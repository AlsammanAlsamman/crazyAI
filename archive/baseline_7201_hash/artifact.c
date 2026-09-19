#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

#define PRIME64_1 11400714785074694791ULL
#define PRIME64_2 14029467366897019727ULL
#define PRIME64_3 1609587929392839161ULL
#define PRIME64_4 9650029242287828579ULL
#define PRIME64_5 2870177450012600261ULL

static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t mergeround(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

static inline uint64_t avalanche(uint64_t h) {
    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

static inline uint64_t read64le(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint32_t read32le(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = PRIME64_1 + PRIME64_2;
        uint64_t v2 = PRIME64_2;
        uint64_t v3 = 0;
        uint64_t v4 = (uint64_t)0 - PRIME64_1;

        do {
            v1 = round64(v1, read64le(p)); p += 8;
            v2 = round64(v2, read64le(p)); p += 8;
            v3 = round64(v3, read64le(p)); p += 8;
            v4 = round64(v4, read64le(p)); p += 8;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = mergeround(h, v1);
        h = mergeround(h, v2);
        h = mergeround(h, v3);
        h = mergeround(h, v4);
    } else {
        h = PRIME64_5;
    }

    h += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = round64(0, read64le(p));
        h ^= k1;
        h = rotl64(h, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)read32le(p) * PRIME64_1;
        h = rotl64(h, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * PRIME64_5;
        h = rotl64(h, 11) * PRIME64_1;
        p++;
    }

    return avalanche(h);
}
