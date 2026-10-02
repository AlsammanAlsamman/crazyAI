#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The cup is never pure: xxHash64's odd primes. */
#define P1 0x9E3779B185EBCA87ULL
#define P2 0xC2B2AE3D27D4EB4FULL
#define P3 0x165667B19E3779F9ULL
#define P4 0x85EBCA77C2B2AE63ULL
#define P5 0x27D4EB2F165667C5ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t rd64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint64_t rd32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return (uint64_t)v;
}

/* one pull on a cup: the pulled word bends the cup, the cup's new color stays */
static inline uint64_t pull(uint64_t acc, uint64_t w) {
    return rotl64(acc + w * P2, 31) * P1;
}
/* blending the four cups into one */
static inline uint64_t blend(uint64_t h, uint64_t v) {
    v = pull(0, v);
    h ^= v;
    return h * P1 + P4;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    const unsigned char *const end = p + len;
    uint64_t h;

    /* --- weigh the pile at the door: which regime am I in? --- */
    if (len >= 32) {
        /* heavy pile: four cups side by side, four unbroken pours at once */
        uint64_t v1 = P1 + P2, v2 = P2, v3 = 0, v4 = (uint64_t)0 - P1;
        const unsigned char *const limit = end - 32;
        do {
            v1 = pull(v1, rd64(p +  0));
            v2 = pull(v2, rd64(p +  8));
            v3 = pull(v3, rd64(p + 16));
            v4 = pull(v4, rd64(p + 24));
            p += 32;
        } while (p <= limit);
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = blend(h, v1);
        h = blend(h, v2);
        h = blend(h, v3);
        h = blend(h, v4);
    } else {
        /* light pile: one cup is enough, and it is already sickened */
        h = P5;
    }

    h += (uint64_t)len;   /* the size of the pile is itself a mark */

    /* the dregs: whatever did not fill a full round of cups */
    while (p + 8 <= end) { h ^= pull(0, rd64(p)); h = rotl64(h, 27) * P1 + P4; p += 8; }
    if    (p + 4 <= end) { h ^= rd32(p) * P1;     h = rotl64(h, 23) * P2 + P3; p += 4; }
    while (p < end)      { h ^= (uint64_t)(*p) * P5; h = rotl64(h, 11) * P1;   p += 1; }

    /* --- the organs: each bends, discards half, keeps what refuses to sit still --- */
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;   /* organ 1 */
    h ^= h >> 29; h *= 0xC4CEB9FE1A85EC53ULL;   /* organ 2 */
    h ^= h >> 32; h *= 0x9E3779B97F4A7C15ULL;   /* organ 3 */
    if (len >= 16) {                            /* a small sickness needs less compensating */
        h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;   /* organ 4 */
        h ^= h >> 27; h *= 0x94D049BB133111EBULL;   /* organ 5 */
        h ^= h >> 31; h *= 0x2545F4914F6CDD1DULL;   /* organ 6 */
        h ^= h >> 32; h *= 0xD6E8FEB86659FD93ULL;   /* organ 7 */
    }
    h ^= h >> 32;                               /* the garden door: the last discard */
    return h;
}
