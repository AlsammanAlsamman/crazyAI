#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* the four birds' gauges, and the stone at the water's edge */
#define B1 11400714785074694791ULL
#define B2 14029467366897019727ULL
#define B3  1609587929392839161ULL
#define B4  9650029242287828579ULL
#define B5  2870177450012600261ULL

/* a crease: an angle, not a product */
static inline uint64_t crease(uint64_t x, int a) {
    return (x << a) | (x >> (64 - a));
}
static inline uint64_t mark8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, sizeof v); return v;   /* one mov under -O3 */
}
static inline uint32_t mark4(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, sizeof v); return v;
}
/* one fold: the new angle is set by the mark AND the crease before it */
static inline uint64_t fold(uint64_t acc, uint64_t m) {
    return crease(acc + m * B2, 31) * B1;
}
/* the beaks disagreed: refold tighter before this bird joins the wad */
static inline uint64_t refold(uint64_t h, uint64_t v) {
    return (h ^ fold(0, v)) * B1 + B4;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p   = data;
    const unsigned char *end = data + len;
    uint64_t h;

    if (len >= 32) {                 /* "only that it's large enough" */
        const unsigned char *limit = end - 32;
        /* four birds, four different beaks - never seeded alike */
        uint64_t v1 = B1 + B2, v2 = B2, v3 = 0, v4 = (uint64_t)0 - B1;
        do {
            /* one crease, pressed against all four beaks at once;
               no bird waits on another bird's reading */
            v1 = fold(v1, mark8(p +  0));
            v2 = fold(v2, mark8(p +  8));
            v3 = fold(v3, mark8(p + 16));
            v4 = fold(v4, mark8(p + 24));
            p += 32;
        } while (p <= limit);        /* never skipping, never looking ahead */

        /* spiral and snail-shell lined up WRONG on purpose: distinct
           angles + an ordered merge, so no two piles fold the same way */
        h = crease(v1, 1) + crease(v2, 7) + crease(v3, 12) + crease(v4, 18);
        h = refold(h, v1);
        h = refold(h, v2);
        h = refold(h, v3);
        h = refold(h, v4);
    } else {
        h = B5;                      /* too small for the birds: one chain */
    }

    h += (uint64_t)len;              /* the one small suitcase goes under too */

    /* the last few marks, still strictly in order, still in bounds */
    while (end - p >= 8) {
        h ^= fold(0, mark8(p));
        h  = crease(h, 27) * B1 + B4;
        p += 8;
    }
    if (end - p >= 4) {
        h ^= (uint64_t)mark4(p) * B1;
        h  = crease(h, 23) * B2 + B3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * B5;
        h  = crease(h, 11) * B1;
        p++;
    }

    /* the water's edge: hold it down until it is small, small, small */
    h ^= h >> 33; h *= B2;
    h ^= h >> 29; h *= B3;
    h ^= h >> 32;
    return h;                        /* the coin. every scrap died above. */
}
