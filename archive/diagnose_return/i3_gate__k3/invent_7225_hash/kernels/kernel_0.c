#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the cup: "never pure" => a blend of eight parts ---------------- */
#define BASIS 0x9E3779B97F4A7C15ULL
static const uint64_t PRIME_POUR = 0x9E3779B185EBCA87ULL;
static const uint64_t PM1 = 0xC2B2AE3D27D4EB4FULL;
static const uint64_t PM2 = 0x165667B19E3779F9ULL;
static const uint64_t PM3 = 0x85EBCA77C2B2AE63ULL;

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));           /* r is always 1..59 */
}

/* one pull: the mark enters, the cup's colour turns, that colour is the
   seed of the next pull.  Deliberately cheap - ONE multiply per 8 bytes -
   because all avalanche duty is discharged by the organs below. */
static inline uint64_t pull(uint64_t l, uint64_t w) {
    return rotl64(l ^ w, 29) * PRIME_POUR;
}

/* ---- the seven organs ------------------------------------------------
   each bends, discards half of what it received, and passes on only
   what refuses to sit still.  Organs 1-5 are splitmix64's validated
   finalizer; 6-7 are the extra stir-and-fold the native insists on. */
static inline uint64_t seven_organs(uint64_t h) {
    h ^= h >> 30;                   /* 1: fold, keep only what differs */
    h *= 0xBF58476D1CE4E5B9ULL;     /* 2: bend                         */
    h ^= h >> 27;                   /* 3: fold                         */
    h *= 0x94D049BB133111EBULL;     /* 4: bend                         */
    h ^= h >> 31;                   /* 5: fold                         */
    h *= 0x9E3779B97F4A7C15ULL;     /* 6: bend                         */
    h ^= h >> 32;                   /* 7: fold                         */
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t h;

    if (len >= 64) {
        /* --- the tavern's pile: eight cups, poured side by side ----- */
        uint64_t l0 = BASIS + PM1, l1 = BASIS - PM1;
        uint64_t l2 = BASIS + PM2, l3 = BASIS - PM2;
        uint64_t l4 = BASIS + PM3, l5 = BASIS - PM3;
        uint64_t l6 = BASIS ^ PM1, l7 = BASIS ^ PM2;

        size_t nblk = len >> 6;
        for (size_t i = 0; i < nblk; i++) {
            l0 = pull(l0, ld64(p +  0));
            l1 = pull(l1, ld64(p +  8));
            l2 = pull(l2, ld64(p + 16));
            l3 = pull(l3, ld64(p + 24));
            l4 = pull(l4, ld64(p + 32));
            l5 = pull(l5, ld64(p + 40));
            l6 = pull(l6, ld64(p + 48));
            l7 = pull(l7, ld64(p + 56));
            p += 64;
        }

        /* the dregs - every mark exactly once, never twice */
        size_t rem = len & 63;
        uint64_t t = BASIS;
        while (rem >= 8) { t = pull(t, ld64(p)); p += 8; rem -= 8; }
        if (rem) {
            uint64_t w = 0;
            for (size_t i = 0; i < rem; i++) w |= (uint64_t)p[i] << (8 * i);
            t = pull(t, w);
        }

        /* pour the eight cups back into one - position-sensitive */
        h  = rotl64(l0,  1) + rotl64(l1,  7) + rotl64(l2, 12) + rotl64(l3, 18)
           + rotl64(l4, 23) + rotl64(l5, 29) + rotl64(l6, 34) + rotl64(l7, 40);
        h ^= rotl64(l0, 11) * PM1;
        h ^= rotl64(l1, 19) * PM2;
        h ^= rotl64(l2, 27) * PM3;
        h ^= rotl64(l3, 35) * PM1;
        h ^= rotl64(l4, 43) * PM2;
        h ^= rotl64(l5, 51) * PM3;
        h ^= rotl64(l6, 59) * PM1;
        h ^= rotl64(l7,  5) * PM2;
        h ^= t;

    } else if (len >= 8) {
        /* --- a pile for one hand: a single cup ---------------------- */
        uint64_t l = BASIS;
        size_t rem = len;
        while (rem >= 8) { l = pull(l, ld64(p)); p += 8; rem -= 8; }
        if (rem) {
            uint64_t w = 0;
            for (size_t i = 0; i < rem; i++) w |= (uint64_t)p[i] << (8 * i);
            l = pull(l, w);
        }
        h = l;

    } else {
        /* --- a handful of marks: no pour at all.  The organs alone.
               (guard: the finalizer's fixed cost is never stacked on
                top of a pour that cannot pay for itself.)            */
        uint64_t w = 0;
        for (size_t i = 0; i < len; i++) w |= (uint64_t)p[i] << (8 * i);
        h = BASIS ^ w ^ ((uint64_t)len << 56);   /* len<8 => bits 56.. free */
    }

    h += (uint64_t)len * PM3;
    return seven_organs(h);
}
