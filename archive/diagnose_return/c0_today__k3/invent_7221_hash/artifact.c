#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The four birds are different birds: four distinct wire gauges. */
#define P1 11400714785074694791ULL
#define P2 14029467366897019727ULL
#define P3  1609587929392839161ULL
#define P4  9650029242287828579ULL
#define P5  2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* unaligned reads; gcc -O3 folds these to single loads */
static inline uint64_t rd8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t rd4(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

/* one fold pressed against one beak: lane value folded into that bird */
static inline uint64_t beak(uint64_t acc, uint64_t lane) {
    acc += lane * P2;
    acc  = rotl64(acc, 31);
    acc *= P1;
    return acc;
}

/* "only when all four agree": a bird's reading folded into the agreement,
   deliberately scrambled so the birds cannot be interchanged */
static inline uint64_t agree(uint64_t h, uint64_t bird) {
    uint64_t v = beak(0, bird);
    h ^= v;
    h  = h * P1 + P4;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p   = data;
    const unsigned char * restrict end = data + len;
    uint64_t h;

    if (len >= 32) {
        /* the sheet is large enough: lay out all four birds */
        const unsigned char *limit = end - 32;
        uint64_t b1 = P1 + P2;
        uint64_t b2 = P2;
        uint64_t b3 = 0;
        uint64_t b4 = (uint64_t)0 - P1;

        /* four beaks at once: four independent chains, 32 bytes per pass */
        do {
            b1 = beak(b1, rd8(p +  0));
            b2 = beak(b2, rd8(p +  8));
            b3 = beak(b3, rd8(p + 16));
            b4 = beak(b4, rd8(p + 24));
            p += 32;
        } while (p <= limit);

        /* the birds disagree on purpose: distinct rotations, no symmetry */
        h = rotl64(b1, 1) + rotl64(b2, 7) + rotl64(b3, 12) + rotl64(b4, 18);
        h = agree(h, b1);
        h = agree(h, b2);
        h = agree(h, b3);
        h = agree(h, b4);
    } else {
        /* too few marks for four gauges: one sheet, one gauge */
        h = P5;
    }

    /* the one small suitcase carried under with the body */
    h += (uint64_t)len;

    /* scraps trimmed off the sheet */
    while (p + 8 <= end) {
        h ^= beak(0, rd8(p));
        h  = rotl64(h, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)rd4(p) * P1;
        h  = rotl64(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p++) * P5;
        h  = rotl64(h, 11) * P1;
    }

    /* the water's edge: small, small, small -> one coin. Three presses, then stop. */
    h ^= h >> 33; h *= P2;
    h ^= h >> 29; h *= P3;
    h ^= h >> 32;
    return h;
}
