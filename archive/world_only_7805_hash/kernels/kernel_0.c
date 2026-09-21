#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const uint64_t PRIME1 = 0x9E3779B185EBCA87ULL;
    const uint64_t PRIME2 = 0xC2B2AE3D27D4EB4FULL;
    const uint64_t PRIME3 = 0x165667B19E3779F9ULL;
    const uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
    const uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;

    size_t n = len;
    const unsigned char *p = data;
    uint64_t h;

    if (n >= 32) {
        uint64_t v1 = PRIME1 + PRIME2;
        uint64_t v2 = PRIME2;
        uint64_t v3 = 0;
        uint64_t v4 = 0ULL - PRIME1;

        while (n >= 32) {
            uint64_t x1, x2, x3, x4;
            memcpy(&x1, p,      8);
            memcpy(&x2, p + 8,  8);
            memcpy(&x3, p + 16, 8);
            memcpy(&x4, p + 24, 8);

            v1 += x1 * PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1;
            v2 += x2 * PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1;
            v3 += x3 * PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1;
            v4 += x4 * PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1;

            p += 32;
            n -= 32;
        }

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1; h ^= v1; h = h * PRIME1 + PRIME4;
        v2 *= PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1; h ^= v2; h = h * PRIME1 + PRIME4;
        v3 *= PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1; h ^= v3; h = h * PRIME1 + PRIME4;
        v4 *= PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1; h ^= v4; h = h * PRIME1 + PRIME4;
    } else {
        h = PRIME5;
    }

    h += (uint64_t)len;

    while (n >= 8) {
        uint64_t k1;
        memcpy(&k1, p, 8);
        k1 *= PRIME2; k1 = rotl64(k1, 31); k1 *= PRIME1;
        h ^= k1;
        h = rotl64(h, 27) * PRIME1 + PRIME4;
        p += 8;
        n -= 8;
    }

    if (n >= 4) {
        uint32_t k;
        memcpy(&k, p, 4);
        h ^= (uint64_t)k * PRIME1;
        h = rotl64(h, 23) * PRIME2 + PRIME3;
        p += 4;
        n -= 4;
    }

    while (n > 0) {
        h ^= (uint64_t)(*p) * PRIME5;
        h = rotl64(h, 11) * PRIME1;
        p++;
        n--;
    }

    h = fmix64(h);
    return h;
}
