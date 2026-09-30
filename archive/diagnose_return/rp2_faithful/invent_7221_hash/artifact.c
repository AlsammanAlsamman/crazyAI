#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- spiral & snail-shell markings: golden-ratio odd constants, deliberately misaligned ---- */
#define S0 0x9E3779B97F4A7C15ULL
#define S1 0xC2B2AE3D27D4EB4FULL
#define S2 0x165667B19E3779F9ULL
#define S3 0x27D4EB2F165667C5ULL

/* ---- the four wire birds' beaks: the gauges the crease is pressed against ---- */
#define BEAK0 0xFF51AFD7ED558CCDULL
#define BEAK1 0xC4CEB9FE1A85EC53ULL
#define BEAK2 0xBF58476D1CE4E5B9ULL
#define BEAK3 0x94D049BB133111EBULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> ((64u - r) & 63u));
}
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }

/* ONE FOLD PER MARK.  angle = f(mark, crease before it).  bijective in b for fixed w,
   so two different piles can never fold down onto the same crease: "line up wrong on purpose". */
#define FOLD(b, w, r) ((b) = rotl64((b) ^ (w), (r)) + (w))

/* "only when all four beaks agree does the crease count as set", then the water's edge:
   hold it down until it is small, small, small -> one hard dense corner, the coin. */
static inline uint64_t set_and_thin(uint64_t b0, uint64_t b1, uint64_t b2, uint64_t b3,
                                    uint64_t len)
{
    uint64_t t = S0 ^ rotl64(len, 27);
    t = (t ^ b0) * BEAK0; t = rotl64(t, 31);
    t = (t ^ b1) * BEAK1; t = rotl64(t, 29);
    t = (t ^ b2) * BEAK2; t = rotl64(t, 27);
    t = (t ^ b3) * BEAK3;
    t ^= t >> 33;  t *= BEAK0;
    t ^= t >> 29;  t *= BEAK1;
    t ^= t >> 32;
    return t;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *p = data;
    size_t n = len;

    /* the four wire birds, seeded with the size of the pile */
    uint64_t b0 = S0 ^ (uint64_t)len;
    uint64_t b1 = S1;
    uint64_t b2 = S2;
    uint64_t b3 = S3 ^ rotl64((uint64_t)len, 32);

    /* choose the sheet: pink or orange, doesn't matter - only that it is large enough
       for the pile of marks.  This is the runtime regime test. */
    if (n >= 64) {
        /* ---- LARGE SHEET: one cache line per pass, each fold to a different bird ---- */
        do {
            uint64_t w0 = ld64(p),      w1 = ld64(p + 8),  w2 = ld64(p + 16), w3 = ld64(p + 24);
            uint64_t w4 = ld64(p + 32), w5 = ld64(p + 40), w6 = ld64(p + 48), w7 = ld64(p + 56);
            FOLD(b0, w0, 13); FOLD(b1, w1, 29); FOLD(b2, w2, 41); FOLD(b3, w3, 53);
            FOLD(b0, w4, 17); FOLD(b1, w5, 31); FOLD(b2, w6, 43); FOLD(b3, w7, 59);
            p += 64; n -= 64;
        } while (n >= 64);

        if (n >= 32) {
            uint64_t w0 = ld64(p), w1 = ld64(p + 8), w2 = ld64(p + 16), w3 = ld64(p + 24);
            FOLD(b0, w0, 13); FOLD(b1, w1, 29); FOLD(b2, w2, 41); FOLD(b3, w3, 53);
            p += 32; n -= 32;
        }
        /* at most three whole marks left, still one to each bird in turn */
        if (n >= 8) { uint64_t w = ld64(p); FOLD(b0, w, 23); p += 8; n -= 8; }
        if (n >= 8) { uint64_t w = ld64(p); FOLD(b1, w, 37); p += 8; n -= 8; }
        if (n >= 8) { uint64_t w = ld64(p); FOLD(b2, w, 47); p += 8; n -= 8; }
        /* the trimmed scrap: re-read the last whole mark (safe, len >= 64), never skip it */
        if (n)      { uint64_t w = ld64(data + len - 8); FOLD(b3, w, 7); }

        return set_and_thin(b0, b1, b2, b3, (uint64_t)len);
    }

    /* ---- SMALL SHEET: same folds, same four birds, no wide set-up, marks read one at a
       time to the end.  Fallback path for a pile too small to need the big sheet. ---- */
    {
        static const unsigned R[4] = { 13u, 29u, 41u, 53u };
        uint64_t bb[4];
        unsigned k = 0;
        bb[0] = b0; bb[1] = b1; bb[2] = b2; bb[3] = b3;

        while (n >= 8) {
            uint64_t w = ld64(p);
            bb[k] = rotl64(bb[k] ^ w, R[k]) + w;
            k = (k + 1u) & 3u; p += 8; n -= 8;
        }
        if (n) {                       /* 1..7 loose marks, gathered then folded once */
            uint64_t tw = 0;
            for (size_t i = 0; i < n; i++) tw = (tw << 8) ^ (uint64_t)p[i];
            tw ^= ((uint64_t)n << 56);  /* the count is part of the fold: no two piles alias */
            bb[k] = rotl64(bb[k] ^ tw, 11) + tw;
        }
        return set_and_thin(bb[0], bb[1], bb[2], bb[3], (uint64_t)len);
    }
}
