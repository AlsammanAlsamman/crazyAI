#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The four wooden numbered keeps mounted at the desk's edge. */
#define KEEP1 0x9E3779B185EBCA87ULL
#define KEEP2 0xC2B2AE3D27D4EB4FULL
#define KEEP3 0x165667B19E3779F9ULL
#define KEEP4 0x27D4EB2F165667C5ULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}

/* turn the stone a quarter through the chalk-bank groove: 64/4 = 16 bits */
static inline uint64_t turn_quarter(uint64_t x) { return rotl64(x, 16); }

/* read eight flat marks as one weight -- the flat thing invested into three */
static inline uint64_t weight8(const unsigned char *p) {
    uint64_t w; memcpy(&w, p, sizeof w); return w;
}

/* one fold: turn the stone a quarter, drop the weight into the same seated
   place (so it lands on already-turned callus, never a clean face), and let
   the stone's memory of every prior press bend how deep this one goes.
   Rotate, xor-constant-free injection, and odd multiply are each bijections,
   so no two folding-paths that started differently walk the same last step. */
static inline uint64_t press(uint64_t stone, uint64_t w, uint64_t keep) {
    return (turn_quarter(stone) ^ w) * keep;
}

/* lift the stone off the desk and read its seated number against the keeps.
   Off-chain, fixed constants, once.  This is MurmurHash3's fmix64, unchanged:
   the fold's multiplies diffuse only upward, and the >>33 steps supply the
   downward diffusion the fold cannot. */
static inline uint64_t read_keeps(uint64_t x) {
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 33;
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    const unsigned char *const end = data + len;
    uint64_t seated;

    /* Regime test: one glance down the numbered groove.  A pile shorter than
       one full revolution of the stone (4 faces x 8 marks = 32) never gets
       the four-faced layout -- it goes straight down the single-seated-face
       path below, so short piles pay nothing for the wide desk. */
    if (len >= 32) {
        /* the four faces of the die-stone: 256 bits on the desk, of which
           only 64 will ever leave it.  The rest is groove-dust. */
        uint64_t f0 = KEEP1 + KEEP2;
        uint64_t f1 = KEEP2;
        uint64_t f2 = 0ULL;
        uint64_t f3 = 0ULL - KEEP1;

        const unsigned char *const stop = end - 32;
        do {
            /* the stone turns once per mark-weight; mark i seats on face i%4.
               Four independent callus-chains, no reset, nothing washed clean. */
            f0 = press(f0, weight8(p +  0), KEEP1);
            f1 = press(f1, weight8(p +  8), KEEP2);
            f2 = press(f2, weight8(p + 16), KEEP3);
            f3 = press(f3, weight8(p + 24), KEEP4);
            p += 32;
        } while (p <= stop);

        /* lift the stone: collapse the four faces onto the one seated face.
           Rotations 1/7/12/18 and this merge shape are XXH64's, validated. */
        seated = rotl64(f0, 1) + rotl64(f1, 7) + rotl64(f2, 12) + rotl64(f3, 18);
        seated = press(seated, f0, KEEP1);
        seated = press(seated, f1, KEEP2);
        seated = press(seated, f2, KEEP3);
        seated = press(seated, f3, KEEP4);
    } else {
        seated = KEEP1 + KEEP2;   /* short pile: the seated face alone */
    }

    /* whatever marks remain, keep pressing them into the seated face */
    while (end - p >= 8) { seated = press(seated, weight8(p), KEEP1); p += 8; }
    if (end - p >= 4) {
        uint32_t v; memcpy(&v, p, 4);
        seated = press(seated, (uint64_t)v, KEEP2);
        p += 4;
    }
    while (p < end) { seated = press(seated, (uint64_t)(*p), KEEP3); p++; }

    /* the stone turned once for every mark: the count of folds is itself
       read off against the keeps, so piles of different length cannot seat
       alike even when their marks agree. */
    seated = press(seated, (uint64_t)len, KEEP4);

    /* only the last seated number ever leaves the desk. */
    return read_keeps(seated);
}
