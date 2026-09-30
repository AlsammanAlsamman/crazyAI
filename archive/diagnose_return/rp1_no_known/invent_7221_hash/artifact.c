#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the four wire birds: each beak its own prime, its own merge angle ---- */
#define BEAK0 0x9E3779B97F4A7C15ULL
#define BEAK1 0xC2B2AE3D27D4EB4FULL
#define BEAK2 0x165667B19E3779F9ULL
#define BEAK3 0x27D4EB2F165667C5ULL
#define BASIS 1469598103934665603ULL            /* the blank sheet */

#define ROTC(x,c) (((x) << (c)) | ((x) >> (64 - (c))))   /* fixed beak angle, 1..63 */

static inline uint64_t read_mark(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, sizeof v); return v;      /* one mark, read in order */
}

/* ONE FOLD.  The angle is decided by two things together -- the mark itself and
   the crease left by the fold before it -- and then the corner is pressed
   against this bird's beak.  Angle is forced into 16..47 so it is never 0/64. */
static inline uint64_t press(uint64_t crease, uint64_t mark, uint64_t beak) {
    unsigned a = (unsigned)((mark ^ crease) & 31u) + 16u;
    uint64_t bent = (crease << a) | (crease >> (64u - a));
    return (bent ^ mark) * beak;
}

/* THE WATER'S EDGE.  Thin the whole wad down to one hard dense corner no
   bigger than a coin; nothing else survives to be compared next time. */
static inline uint64_t thin_to_coin(uint64_t x) {
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 29; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 32;
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* "pink or orange, doesn't matter, only that it's LARGE ENOUGH":
       a small pile never gets the big sheet or the four-bird rig -- one beak,
       mark by mark, then straight to the water.  This is the fallback path. */
    if (len < 32) {
        uint64_t c = BASIS ^ (BEAK0 * (uint64_t)len);   /* thickness of the wad */
        size_t i = 0;
        for (; i + 8 <= len; i += 8) c = press(c, read_mark(p + i), BEAK1);
        for (; i < len; i++)         c = press(c, (uint64_t)p[i], BEAK1);
        return thin_to_coin(c);
    }

    /* four birds, four creases in progress at once */
    uint64_t l0 = BASIS ^ (BEAK0 * (uint64_t)len);
    uint64_t l1 = BASIS + BEAK1;
    uint64_t l2 = BASIS ^ BEAK2;
    uint64_t l3 = BASIS - BEAK3;

    size_t i = 0;
    const size_t blocks = len & ~(size_t)31;

    /* each successive mark is pressed against a DIFFERENT beak, cycling 0,1,2,3 */
    for (; i < blocks; i += 32) {
        const uint64_t m0 = read_mark(p + i);
        const uint64_t m1 = read_mark(p + i +  8);
        const uint64_t m2 = read_mark(p + i + 16);
        const uint64_t m3 = read_mark(p + i + 24);
        l0 = press(l0, m0, BEAK0);
        l1 = press(l1, m1, BEAK1);
        l2 = press(l2, m2, BEAK2);
        l3 = press(l3, m3, BEAK3);
    }

    /* the scraps trimmed off the edge -- still round-robin across the beaks */
    size_t rem = len - i;
    if (rem >= 8) { l0 = press(l0, read_mark(p + i), BEAK0); i += 8; rem -= 8; }
    if (rem >= 8) { l1 = press(l1, read_mark(p + i), BEAK1); i += 8; rem -= 8; }
    if (rem >= 8) { l2 = press(l2, read_mark(p + i), BEAK2); i += 8; rem -= 8; }
    if (rem)      { l3 = press(l3, read_mark(p + len - 8), BEAK3); } /* len>=32: in range */

    /* only when all four beaks agree does the crease count as set:
       distinct angles, so no two piles line up cleanly by symmetry. */
    uint64_t x = ROTC(l0,1) + ROTC(l1,7) + ROTC(l2,12) + ROTC(l3,18);
    return thin_to_coin(x);
}
