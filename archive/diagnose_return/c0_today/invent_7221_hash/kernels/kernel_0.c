#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* --- the four beaks: four distinct gauge constants --- */
#define P1 11400714785074694791ULL
#define P2 14029467366897019727ULL
#define P3  1609587929392839161ULL
#define P4  9650029242287828579ULL
#define P5  2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;          /* little-endian host */
}
static inline uint32_t ld32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

/* one fold: angle set by the mark (w) and the crease before it (acc) */
static inline uint64_t fold(uint64_t acc, uint64_t w) {
    acc += w * P2;
    acc  = rotl64(acc, 31);
    acc *= P1;
    return acc;
}

/* scrambled, asymmetric merge: lanes must NOT be interchangeable */
static inline uint64_t merge(uint64_t h, uint64_t v) {
    v  = fold(0, v);
    h ^= v;
    h  = h * P1 + P4;
    return h;
}

/* the water's edge: thin the wad down to one dense coin */
static inline uint64_t avalanche(uint64_t h) {
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p   = data;
    const unsigned char *restrict end = data + len;
    uint64_t h;

    /* REGIME CHECK: "pink or orange, doesn't matter, only that it's
       large enough." Too small to give every bird a corner -> one bird. */
    if (len >= 32) {
        const unsigned char *const limit = end - 32;
        uint64_t v1 = P1 + P2;      /* four different beaks */
        uint64_t v2 = P2;
        uint64_t v3 = 0;
        uint64_t v4 = 0 - P1;

        /* enormous pile: each bird takes two corners per pass.
           Identical result, better scheduling. */
        if (len >= 64) {
            const unsigned char *const limit64 = end - 64;
            do {
                __builtin_prefetch(p + 512, 0, 0);
                v1 = fold(v1, ld64(p +  0));
                v2 = fold(v2, ld64(p +  8));
                v3 = fold(v3, ld64(p + 16));
                v4 = fold(v4, ld64(p + 24));
                v1 = fold(v1, ld64(p + 32));
                v2 = fold(v2, ld64(p + 40));
                v3 = fold(v3, ld64(p + 48));
                v4 = fold(v4, ld64(p + 56));
                p += 64;
            } while (p <= limit64);
        }
        while (p <= limit) {
            v1 = fold(v1, ld64(p +  0));
            v2 = fold(v2, ld64(p +  8));
            v3 = fold(v3, ld64(p + 16));
            v4 = fold(v4, ld64(p + 24));
            p += 32;
        }

        /* all four beaks must agree before the crease is set */
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = merge(h, v1);
        h = merge(h, v2);
        h = merge(h, v3);
        h = merge(h, v4);
    } else {
        h = P5;                     /* FALLBACK PATH: a single bird */
    }

    h += (uint64_t)len;             /* the one small suitcase */

    /* trim the remaining scraps */
    while (p + 8 <= end) {
        h ^= fold(0, ld64(p));
        h  = rotl64(h, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)ld32(p) * P1;
        h  = rotl64(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * P5;
        h  = rotl64(h, 11) * P1;
        p += 1;
    }

    /* nothing else survives: only the coin */
    return avalanche(h);
}
