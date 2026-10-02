#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Four wire birds. Each bird's beak is a different gauge constant. */
#define BIRD1 11400714785074694791ULL /* 0x9E3779B185EBCA87 */
#define BIRD2 14029467366897019727ULL /* 0xC2B2AE3D27D4EB4F */
#define BIRD3  1609587929392839161ULL /* 0x165667B19E3779F9 */
#define BIRD4  9650029242287828579ULL /* 0x85EBCA77C2B2AE63 */
#define SCRAP  2870177450012600261ULL /* 0x27D4EB2F165667C5 */

/* spiral / snail-shell markings, lined up wrong on purpose */
static inline uint64_t spiral(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* one fold: angle set by the mark AND the crease before it, then refolded tighter */
static inline uint64_t fold(uint64_t crease, uint64_t mark) {
    crease += mark * BIRD2;
    crease  = spiral(crease, 31);
    crease *= BIRD1;
    return crease;
}

/* a beak agreeing: the bird's crease is consumed into the common wad and destroyed */
static inline uint64_t agree(uint64_t wad, uint64_t crease) {
    crease = fold(0, crease);
    wad   ^= crease;
    wad    = wad * BIRD1 + BIRD4;
    return wad;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;
    const unsigned char *end = data + len;
    uint64_t wad;

    /* "only that it's large enough" -- the native's own regime test */
    if (len >= 32) {
        /* big sheet: four birds, each folding its own stripe, all at once */
        uint64_t b0 = BIRD1 + BIRD2;
        uint64_t b1 = BIRD2;
        uint64_t b2 = 0;
        uint64_t b3 = (uint64_t)0 - BIRD1;
        const unsigned char *limit = end - 32;

        do {
            uint64_t m0, m1, m2, m3;
            memcpy(&m0, p,      8);
            memcpy(&m1, p +  8, 8);
            memcpy(&m2, p + 16, 8);
            memcpy(&m3, p + 24, 8);
            b0 = fold(b0, m0);          /* four independent crease chains, */
            b1 = fold(b1, m1);          /* no bird waits on another bird   */
            b2 = fold(b2, m2);
            b3 = fold(b3, m3);
            p += 32;
        } while (p <= limit);

        /* nothing counts as set until all four beaks agree */
        wad = spiral(b0, 1) + spiral(b1, 7) + spiral(b2, 12) + spiral(b3, 18);
        wad = agree(wad, b0);
        wad = agree(wad, b1);
        wad = agree(wad, b2);
        wad = agree(wad, b3);
    } else {
        /* small pile: no sheet unfolded, one bird, strictly serial (the fallback) */
        wad = SCRAP;
    }

    /* the one small suitcase that goes under with it */
    wad += (uint64_t)len;

    /* leftover marks, folded in order, never skipped */
    while (end - p >= 8) {
        uint64_t k; memcpy(&k, p, 8);
        k    = fold(0, k);
        wad ^= k;
        wad  = spiral(wad, 27) * BIRD1 + BIRD4;
        p += 8;
    }
    if (end - p >= 4) {
        uint32_t k; memcpy(&k, p, 4);
        wad ^= (uint64_t)k * BIRD1;
        wad  = spiral(wad, 23) * BIRD2 + BIRD3;
        p += 4;
    }
    while (p < end) {
        wad ^= (uint64_t)(*p) * SCRAP;
        wad  = spiral(wad, 11) * BIRD1;
        p++;
    }

    /* the water's edge: hold it down until it is small, small, small */
    wad ^= wad >> 33;
    wad *= BIRD2;
    wad ^= wad >> 29;
    wad *= BIRD3;
    wad ^= wad >> 32;

    /* every scrap and every reading that didn't hold is already gone --
       nothing but this coin leaves the function */
    return wad;
}
