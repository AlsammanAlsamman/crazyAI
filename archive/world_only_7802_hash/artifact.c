#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t rd64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint32_t rd32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

#define P1 0x9E3779B185EBCA87ULL
#define P2 0xC2B2AE3D27D4EB4FULL
#define P3 0x165667B19E3779F9ULL
#define P4 0x85EBCA77C2B2AE63ULL
#define P5 0x27D4EB2F165667C5ULL

static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * P2;
    acc = rotl64(acc, 31);
    acc *= P1;
    return acc;
}

static inline uint64_t mergeround(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * P1 + P4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h;

    if (len >= 32) {
        uint64_t v1 = P1 + P2;
        uint64_t v2 = P2;
        uint64_t v3 = 0;
        uint64_t v4 = 0ULL - P1;
        const unsigned char *limit = end - 32;

        do {
            v1 = round64(v1, rd64(p));      p += 8;
            v2 = round64(v2, rd64(p));      p += 8;
            v3 = round64(v3, rd64(p));      p += 8;
            v4 = round64(v4, rd64(p));      p += 8;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = mergeround(h, v1);
        h = mergeround(h, v2);
        h = mergeround(h, v3);
        h = mergeround(h, v4);
    } else {
        h = P5;
    }

    h += (uint64_t)len;

    while ((size_t)(end - p) >= 8) {
        h ^= round64(0, rd64(p));
        h = rotl64(h, 27) * P1 + P4;
        p += 8;
    }
    if ((size_t)(end - p) >= 4) {
        h ^= (uint64_t)rd32(p) * P1;
        h = rotl64(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * P5;
        h = rotl64(h, 11) * P1;
        p++;
    }

    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;

    return h;
}
