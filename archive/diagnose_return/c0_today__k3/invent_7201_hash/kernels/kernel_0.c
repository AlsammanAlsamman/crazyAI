#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the wheel (forward only: rotate, never shift) ---- */
static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t load64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint64_t load32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return (uint64_t)v;
}

#define BASIS 1469598103934665603ULL          /* FNV-1a offset basis, kept as a nod */
#define P1    0x9E3779B185EBCA87ULL           /* the shared slate wheel's multiplier */

/* seven hungers, one per bell-jar (all odd) */
#define PL0 0xC2B2AE3D27D4EB4FULL
#define PL1 0x165667B19E3779F9ULL
#define PL2 0x85EBCA77C2B2AE63ULL
#define PL3 0x27D4EB2F165667C5ULL
#define PL4 0xD6E8FEB86659FD93ULL
#define PL5 0xA3B195354A39B70DULL
#define PL6 0x589965CC75374CC3ULL

/* six frost-flowers, no two ever alike */
#define M1 0x9E3779B97F4A7C15ULL
#define M2 0xBF58476D1CE4E5B9ULL
#define M3 0x94D049BB133111EBULL
#define M4 0x2545F4914F6CDD1DULL
#define M5 0x7FB5D329728EA185ULL
#define M6 0x1C69B3F74AC4AE35ULL

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    size_t rem = len;
    uint64_t h;

    /* ---- weigh the pile: is it thick enough to cover the wire for seven jars? ---- */
    if (rem >= 112) {
        /* seven troughs, seven hungers */
        uint64_t a0 = BASIS + PL0 + PL1;
        uint64_t a1 = BASIS + PL1;
        uint64_t a2 = BASIS;
        uint64_t a3 = BASIS - PL0;
        uint64_t a4 = BASIS + PL2;
        uint64_t a5 = BASIS - PL2;
        uint64_t a6 = BASIS ^ PL3;

        /* one dip of the wire = 56 marks; no lane waits for another to cool */
        do {
            a0 = rotl64(a0 + load64(p +  0) * PL0, 31) * P1;
            a1 = rotl64(a1 + load64(p +  8) * PL1, 29) * P1;
            a2 = rotl64(a2 + load64(p + 16) * PL2, 33) * P1;
            a3 = rotl64(a3 + load64(p + 24) * PL3, 27) * P1;
            a4 = rotl64(a4 + load64(p + 32) * PL4, 37) * P1;
            a5 = rotl64(a5 + load64(p + 40) * PL5, 23) * P1;
            a6 = rotl64(a6 + load64(p + 48) * PL6, 35) * P1;
            p += 56; rem -= 56;
        } while (rem >= 56);

        /* ---- the lattice: seven jars fold to one in exactly six frost-flowers ---- */
        h = a0;
        h = (h ^ rotl64(a1 * M1, 29)) * P1;
        h = (h ^ rotl64(a2 * M2, 31)) * P1;
        h = (h ^ rotl64(a3 * M3, 33)) * P1;
        h = (h ^ rotl64(a4 * M4, 27)) * P1;
        h = (h ^ rotl64(a5 * M5, 35)) * P1;
        h = (h ^ rotl64(a6 * M6, 25)) * P1;
    } else {
        /* thin pile: keep one jar awake, same wheel, same basin */
        h = BASIS + PL3;
    }

    h += (uint64_t)len;                       /* the pile's own weight is a mark too */

    /* ---- the last marks, read one at a time ---- */
    while (rem >= 8) {
        h ^= rotl64(load64(p) * PL0, 31) * P1;
        h  = rotl64(h, 27) * PL2 + PL3;
        p += 8; rem -= 8;
    }
    if (rem >= 4) {
        h ^= load32(p) * P1;
        h  = rotl64(h, 23) * PL1 + PL2;
        p += 4; rem -= 4;
    }
    while (rem > 0) {
        h ^= (uint64_t)(*p) * PL4;
        h  = rotl64(h, 11) * P1;
        p++; rem--;
    }

    /* ---- the brine basin: wait until the frost-lace stops spreading ---- */
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;
    return h;                                 /* only the tin sketch survives */
}
