#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len) {
    static const uint64_t P1 = 11400714785074694791ULL;
    static const uint64_t P2 = 14029467366897019727ULL;
    static const uint64_t P3 = 1609587929392839161ULL;
    static const uint64_t P4 = 9650029242287828579ULL;
    static const uint64_t P5 = 2870177450012600261ULL;

    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        /* Four independent "helpers", each with their own bowl (accumulator).
           These chains are data-independent, so the CPU can run their
           multiplies concurrently instead of waiting on one serial chain. */
        uint64_t v1 = P1 + P2;
        uint64_t v2 = P2;
        uint64_t v3 = 0;
        uint64_t v4 = (uint64_t)0 - P1;

        do {
            uint64_t k1, k2, k3, k4;
            memcpy(&k1, p,      8);
            memcpy(&k2, p + 8,  8);
            memcpy(&k3, p + 16, 8);
            memcpy(&k4, p + 24, 8);
            p += 32;

            v1 += k1 * P2; v1 = rotl64(v1, 31); v1 *= P1;
            v2 += k2 * P2; v2 = rotl64(v2, 31); v2 *= P1;
            v3 += k3 * P2; v3 = rotl64(v3, 31); v3 *= P1;
            v4 += k4 * P2; v4 = rotl64(v4, 31); v4 *= P1;
        } while (p <= limit);

        /* Pour all the mini-swirls together and give it hard final stirs. */
        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= P2; v1 = rotl64(v1, 31); v1 *= P1; h64 ^= v1; h64 = h64 * P1 + P4;
        v2 *= P2; v2 = rotl64(v2, 31); v2 *= P1; h64 ^= v2; h64 = h64 * P1 + P4;
        v3 *= P2; v3 = rotl64(v3, 31); v3 *= P1; h64 ^= v3; h64 = h64 * P1 + P4;
        v4 *= P2; v4 = rotl64(v4, 31); v4 *= P1; h64 ^= v4; h64 = h64 * P1 + P4;
    } else {
        h64 = P5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1;
        memcpy(&k1, p, 8);
        k1 *= P2; k1 = rotl64(k1, 31); k1 *= P1;
        h64 ^= k1;
        h64 = rotl64(h64, 27) * P1 + P4;
        p += 8;
    }

    if (p + 4 <= end) {
        uint32_t k1;
        memcpy(&k1, p, 4);
        h64 ^= (uint64_t)k1 * P1;
        h64 = rotl64(h64, 23) * P2 + P3;
        p += 4;
    }

    while (p < end) {
        h64 ^= (uint64_t)(*p) * P5;
        h64 = rotl64(h64, 11) * P1;
        p++;
    }

    /* Final hard twist-stir: full avalanche mix. */
    h64 ^= h64 >> 33;
    h64 *= P2;
    h64 ^= h64 >> 29;
    h64 *= P3;
    h64 ^= h64 >> 32;

    return h64;
}
