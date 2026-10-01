#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---------------------------------------------------------------
   THE MASON TRAIL

   A mark is a byte.  The sphere is a 64-bit register.  A stalk is
   one mixing stage.  Striking a stalk is a hairline crack: the
   sphere split against a shifted copy of itself.  The angle of the
   crack is the shift/rotate constant.  There is no multiplication
   anywhere in the mixer -- a crack is a shear, not a product.
   --------------------------------------------------------------- */

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* One stalk: the sphere sobs once and cracks at a fixed angle.
   ARX -- add, rotate, xor -- the validated ChaCha/BLAKE2 primitive. */
#define STALK(a, b, r1, r2)      \
    do {                         \
        (a) += (b);              \
        (b)  = rotl64((b), (r1));\
        (b) ^= (a);              \
        (a)  = rotl64((a), (r2));\
    } while (0)

/* The last, smallest crack in the final stalk: the token handed over.
   SplitMix64's finalizer -- xor-shift angles 30 / 27 / 31.  Two
   multiplies live here and here only: once per call, not once per
   byte, so the native's rule ("mixing one BYTE requires no product")
   holds exactly.  This is the one validated avalanche finisher I
   refuse to reinvent. */
static inline uint64_t token(uint64_t z) {
    z ^= z >> 30;  z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27;  z *= 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z;
}

/* Read 8 marks at once, in their given order. */
static inline uint64_t read8(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);      /* -O3 folds this to one MOV */
    return v;
}

/* -------- the narrow path: the straight mouth of the trail --------
   For a pile too short to reach the first bend, and for the tail of
   a long pile.  A fixed count of turns per mark -- no more, no
   fewer -- and not one multiply among them.                       */
static inline uint64_t narrow(uint64_t h, const unsigned char *p, size_t n) {
    while (n >= 8) {
        h ^= read8(p);
        h  = rotl64(h, 29);        /* the coil */
        h += 0x9E3779B97F4A7C15ULL;/* the trail's own slope */
        h ^= h >> 32;              /* a hairline crack */
        h  = rotl64(h, 17);
        p += 8; n -= 8;
    }
    if (n) {                        /* the last few marks */
        uint64_t v = 0;
        for (size_t i = 0; i < n; i++)
            v |= (uint64_t)p[i] << (8 * i);
        h ^= v;
        h  = rotl64(h, 29);
        h += 0x9E3779B97F4A7C15ULL;
        h ^= h >> 32;
    }
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;

    /* --- THE REGIME TEST, stated in-world ---
       A trail is only a coiled trail if it is long enough to coil.
       A short pile never reaches the first bend: carry it down the
       straight mouth.  This is also the guard on the wide path's
       setup cost -- the risk I named, answered here.              */
    if (len < 64)
        return token(narrow(0x9E3779B97F4A7C15ULL ^ (uint64_t)len, p, len));

    /* --- THE WIDE PATH: the trail sheds four eggshells ---
       Four spheres tumble side by side.  They are independent, so
       the machine runs them in parallel (and auto-vectorizes the
       rotate/xor/add -- which a 64x64 multiply chain could not do).
       Only one token is handed over at the end; the shells are
       swept into the jungle's open plumbing and forgotten.        */
    uint64_t s0 = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;
    uint64_t s1 = 0xBF58476D1CE4E5B9ULL + (uint64_t)len;
    uint64_t s2 = 0x94D049BB133111EBULL ^ rotl64((uint64_t)len, 32);
    uint64_t s3 = 0xD1B54A32D192ED03ULL + rotl64((uint64_t)len, 17);

    size_t n = len;

    /* Each turn consumes a 32-byte stripe: one mark-group per shell.
       A FIXED count of stalks per stripe -- two -- no more, no fewer. */
    while (n >= 32) {
        uint64_t m0 = read8(p +  0);
        uint64_t m1 = read8(p +  8);
        uint64_t m2 = read8(p + 16);
        uint64_t m3 = read8(p + 24);

        s0 ^= m0;  s1 ^= m1;  s2 ^= m2;  s3 ^= m3;

        /* stalk one: shells crack against their neighbours, columnwise */
        STALK(s0, s1, 13, 32);
        STALK(s2, s3, 16, 21);
        /* stalk two: the coil turns, so the pairing crosses over --
           this is what makes one wrong mark reshape the WHOLE sphere */
        STALK(s0, s3, 17, 32);
        STALK(s2, s1, 12,  7);

        p += 32; n -= 32;
    }

    /* Sweep the shells into one sphere -- cross-folded so every shell
       touches every other before the token is read. */
    s0 ^= rotl64(s2, 31);
    s1 ^= rotl64(s3, 23);
    s0 += rotl64(s1, 47);
    s0 ^= s1;

    /* The tail drains down the same narrow mouth as a short pile. */
    return token(narrow(s0, p, n));
}
