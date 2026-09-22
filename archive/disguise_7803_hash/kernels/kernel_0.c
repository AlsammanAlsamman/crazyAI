#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

#define PRIME1 11400714785074694791ULL
#define PRIME2 14029467366897019727ULL
#define PRIME3 1609587929392839161ULL
#define PRIME4 9650029242287828579ULL
#define PRIME5 2870177450012600261ULL

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h;

    if (len >= 32) {
        uint64_t v1 = PRIME1 + PRIME2;
        uint64_t v2 = PRIME2;
        uint64_t v3 = 0;
        uint64_t v4 = (uint64_t)0 - PRIME1;

        const unsigned char *limit = end - 32;
        do {
            uint64_t k1, k2, k3, k4;
            memcpy(&k1, p, 8);
            memcpy(&k2, p + 8, 8);
            memcpy(&k3, p + 16, 8);
            memcpy(&k4, p + 24, 8);

            v1 += k1 * PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1;
            v2 += k2 * PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1;
            v3 += k3 * PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1;
            v4 += k4 * PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1;

            p += 32;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1; h ^= v1; h = h * PRIME1 + PRIME4;
        v2 *= PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1; h ^= v2; h = h * PRIME1 + PRIME4;
        v3 *= PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1; h ^= v3; h = h * PRIME1 + PRIME4;
        v4 *= PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1; h ^= v4; h = h * PRIME1 + PRIME4;
    } else {
        h = PRIME5;
    }

    h += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1;
        memcpy(&k1, p, 8);
        k1 *= PRIME2; k1 = rotl64(k1, 31); k1 *= PRIME1;
        h ^= k1;
        h = rotl64(h, 27) * PRIME1 + PRIME4;
        p += 8;
    }
    if (p + 4 <= end) {
        uint32_t k1;
        memcpy(&k1, p, 4);
        h ^= (uint64_t)k1 * PRIME1;
        h = rotl64(h, 23) * PRIME2 + PRIME3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * PRIME5;
        h = rotl64(h, 11) * PRIME1;
        p++;
    }

    h ^= h >> 33;
    h *= PRIME2;
    h ^= h >> 29;
    h *= PRIME3;
    h ^= h >> 32;

    return h;
}
