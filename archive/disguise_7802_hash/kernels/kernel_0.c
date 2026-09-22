#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t read64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t read32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

#define P1 11400714785074694791ULL
#define P2 14029467366897019727ULL
#define P3 1609587929392839161ULL
#define P4 9650029242287828579ULL
#define P5 2870177450012600261ULL

static inline uint64_t xxh_round(uint64_t acc, uint64_t input) {
    acc += input * P2;
    acc = rotl64(acc, 31);
    acc *= P1;
    return acc;
}
static inline uint64_t xxh_merge_round(uint64_t acc, uint64_t val) {
    val = xxh_round(0, val);
    acc ^= val;
    acc = acc * P1 + P4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h64;
    const uint64_t seed = 0;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = seed + P1 + P2;
        uint64_t v2 = seed + P2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - P1;

        do {
            v1 = xxh_round(v1, read64(p)); p += 8;
            v2 = xxh_round(v2, read64(p)); p += 8;
            v3 = xxh_round(v3, read64(p)); p += 8;
            v4 = xxh_round(v4, read64(p)); p += 8;
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh_merge_round(h64, v1);
        h64 = xxh_merge_round(h64, v2);
        h64 = xxh_merge_round(h64, v3);
        h64 = xxh_merge_round(h64, v4);
    } else {
        h64 = seed + P5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        h64 ^= xxh_round(0, read64(p));
        h64 = rotl64(h64, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h64 ^= (uint64_t)read32(p) * P1;
        h64 = rotl64(h64, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h64 ^= (uint64_t)(*p) * P5;
        h64 = rotl64(h64, 11) * P1;
        p += 1;
    }

    h64 ^= h64 >> 33;
    h64 *= P2;
    h64 ^= h64 >> 29;
    h64 *= P3;
    h64 ^= h64 >> 32;

    return h64;
}
