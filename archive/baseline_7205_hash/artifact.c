#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

#define PRIME64_1 0x9E3779B185EBCA87ULL
#define PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define PRIME64_3 0x165667B19E3779F9ULL
#define PRIME64_4 0x85EBCA77C2B2AE63ULL
#define PRIME64_5 0x27D4EB2F165667C5ULL

static inline uint64_t read_u64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint32_t read_u32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t merge_round(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h;
    const uint64_t seed = 0;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME64_1;

        do {
            v1 = round64(v1, read_u64(p)); p += 8;
            v2 = round64(v2, read_u64(p)); p += 8;
            v3 = round64(v3, read_u64(p)); p += 8;
            v4 = round64(v4, read_u64(p)); p += 8;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = merge_round(h, v1);
        h = merge_round(h, v2);
        h = merge_round(h, v3);
        h = merge_round(h, v4);
    } else {
        h = seed + PRIME64_5;
    }

    h += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = round64(0, read_u64(p));
        h ^= k1;
        h = rotl64(h, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }

    if (p + 4 <= end) {
        h ^= (uint64_t)read_u32(p) * PRIME64_1;
        h = rotl64(h, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }

    while (p < end) {
        h ^= (uint64_t)(*p) * PRIME64_5;
        h = rotl64(h, 11) * PRIME64_1;
        p++;
    }

    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;

    return h;
}
