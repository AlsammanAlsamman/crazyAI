#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the die-stone ------------------------------------------------------
   Four faces (256-bit state), never reset, folded once per 32-byte block.
   One fold = quarter turn of each face (ROTL 16), seat the next weights into
   the already-turned positions (+=), then let each face's callus bend its
   neighbour (pairwise, then crossed).  The whole block map is a BIJECTION of
   the 256-bit state, so differing fold-paths can never re-converge:
     given (v0',v1'): v1 = ROTR(v1' ^ v0', 23), v0 = v0' - v1   (and likewise
     for the (v2,v3) pair); the crossing and the turns invert trivially.
   No multiplication touches a data byte.  At the end the stone is read off
   against the wooden numbered keeps -- xxHash64's validated merge, tail and
   avalanche, unchanged -- and the other 192 bits are swept away.
   Regime check: a pile that does not fill the groove (len < 32) never wakes
   the extra faces and falls back to the plain single-face path.          */

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

#define HP1 11400714785074694791ULL
#define HP2 14029467366897019727ULL
#define HP3  1609587929392839161ULL
#define HP4  9650029242287828579ULL
#define HP5  2870177450012600261ULL

static inline uint64_t hld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t hld32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
/* one wooden keep */
static inline uint64_t hkeep(uint64_t v) {
    v *= HP2; v = ROTL64(v, 31); v *= HP1; return v;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    const unsigned char *const end  = p + len;
    uint64_t h;

    if (len >= 32) {                      /* the pile fills the groove */
        uint64_t v0 = HP1 + HP2;
        uint64_t v1 = HP2;
        uint64_t v2 = 0;
        uint64_t v3 = (uint64_t)0 - HP1;
        const unsigned char *const limit = end - 32;
        do {
            uint64_t w0 = hld64(p);
            uint64_t w1 = hld64(p +  8);
            uint64_t w2 = hld64(p + 16);
            uint64_t w3 = hld64(p + 24);

            /* quarter turn, weight seated into the turned position */
            v0 = ROTL64(v0 + w0, 16);
            v1 = ROTL64(v1 + w1, 16);
            v2 = ROTL64(v2 + w2, 16);
            v3 = ROTL64(v3 + w3, 16);

            /* the callus: each face bends its neighbour, then the pairs cross */
            v0 += v1;  v1 = ROTL64(v1, 23) ^ v0;
            v2 += v3;  v3 = ROTL64(v3, 23) ^ v2;
            v0 += v3;  v2 += v1;

            p += 32;
        } while (p <= limit);

        /* read the stone off the numbered keeps; sweep the rest away */
        h  = ROTL64(v0, 1) + ROTL64(v1, 7) + ROTL64(v2, 12) + ROTL64(v3, 18);
        h ^= hkeep(v0); h = h * HP1 + HP4;
        h ^= hkeep(v1); h = h * HP1 + HP4;
        h ^= hkeep(v2); h = h * HP1 + HP4;
        h ^= hkeep(v3); h = h * HP1 + HP4;
    } else {                              /* a handful: one face is enough */
        h = HP5;
    }

    h += (uint64_t)len;                   /* the count of turns is part of the reading */

    /* trailing marks, single face */
    while ((size_t)(end - p) >= 8) {
        h ^= hkeep(hld64(p));
        h  = ROTL64(h, 27) * HP1 + HP4;
        p += 8;
    }
    if ((size_t)(end - p) >= 4) {
        h ^= (uint64_t)hld32(p) * HP1;
        h  = ROTL64(h, 23) * HP2 + HP3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * HP5;
        h  = ROTL64(h, 11) * HP1;
        p += 1;
    }

    /* final avalanche */
    h ^= h >> 33; h *= HP2;
    h ^= h >> 29; h *= HP3;
    h ^= h >> 32;
    return h;
}
