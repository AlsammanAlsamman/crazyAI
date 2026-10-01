#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---------------------------------------------------------------------------
   THE FOUR WIRE BIRDS.  b0..b3 are not four accumulators - they are four
   gauges of the SAME crease.  One pass of BEAKS leaves every bird a function
   of all four birds (word-level agreement); two passes agree at bit level.
   No lane ever holds an independent partial answer.
   This coupling pass is the SipRound of Aumasson & Bernstein (SipHash) -
   the validated four-word cross-check, arrived at rather than invented.
   --------------------------------------------------------------------------- */
#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

#define BEAKS(b0, b1, b2, b3) do {                                    \
    (b0) += (b1); (b1) = ROTL64((b1), 13); (b1) ^= (b0);              \
    (b0) = ROTL64((b0), 32);                                          \
    (b2) += (b3); (b3) = ROTL64((b3), 16); (b3) ^= (b2);              \
    (b0) += (b3); (b3) = ROTL64((b3), 21); (b3) ^= (b0);              \
    (b2) += (b1); (b1) = ROTL64((b1), 17); (b1) ^= (b2);              \
    (b2) = ROTL64((b2), 32);                                          \
} while (0)

/* read one mark: unaligned, little-endian, compiles to a single movq */
static inline uint64_t mark8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, sizeof(v)); return v;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    size_t n = len;

    /* The sheet is chosen for the pile, and which sheet was chosen is
       recorded on the token: differently sized piles start scrambled
       differently and can never fold down to the same corner. */
    uint64_t b0 = 0x736f6d6570736575ULL ^ (uint64_t)len;
    uint64_t b1 = 0x646f72616e646f6dULL;
    uint64_t b2 = 0x6c7967656e657261ULL ^ ROTL64((uint64_t)len ^ 0x9e3779b97f4a7c15ULL, 32);
    uint64_t b3 = 0x7465646279746573ULL;

    /* ===== BIG SHEET: one crease, four corners, four beaks, all at once ===== */
    if (n >= 32) {
        do {
            uint64_t m0 = mark8(p);
            uint64_t m1 = mark8(p +  8);
            uint64_t m2 = mark8(p + 16);
            uint64_t m3 = mark8(p + 24);

            b0 ^= m0; b1 ^= m1; b2 ^= m2; b3 ^= m3;  /* one corner per beak  */
            BEAKS(b0, b1, b2, b3);                   /* fold                 */
            BEAKS(b0, b1, b2, b3);                   /* refold tighter: now
                                                        all four agree       */
            b0 ^= m2; b1 ^= m3; b2 ^= m0; b3 ^= m1;  /* each mark must read
                                                        the same against a
                                                        second beak too      */
            p += 32; n -= 32;
        } while (n >= 32);
    }

    /* ===== SMALL SHEET: one mark per crease, pressed on both sides ===== */
    while (n >= 8) {
        uint64_t m = mark8(p);
        b3 ^= m;
        BEAKS(b0, b1, b2, b3);
        b0 ^= m;
        p += 8; n -= 8;
    }

    /* ===== the last, ragged mark (carries the pile's length) ===== */
    {
        uint64_t t = (uint64_t)len << 56;
        switch (n) {
            case 7: t |= (uint64_t)p[6] << 48;  /* fall through */
            case 6: t |= (uint64_t)p[5] << 40;  /* fall through */
            case 5: t |= (uint64_t)p[4] << 32;  /* fall through */
            case 4: t |= (uint64_t)p[3] << 24;  /* fall through */
            case 3: t |= (uint64_t)p[2] << 16;  /* fall through */
            case 2: t |= (uint64_t)p[1] <<  8;  /* fall through */
            case 1: t |= (uint64_t)p[0];        /* fall through */
            default: break;
        }
        b3 ^= t;
        BEAKS(b0, b1, b2, b3);
        b0 ^= t;
    }

    /* ===== THE WATER'S EDGE: hold it down small, small, small ===== */
    b2 ^= 0xffULL;
    BEAKS(b0, b1, b2, b3);
    BEAKS(b0, b1, b2, b3);
    BEAKS(b0, b1, b2, b3);

    /* 256 bits of sheet thinned to one dense corner; nothing else survives */
    uint64_t coin = b0 ^ b1 ^ b2 ^ b3;
    coin ^= coin >> 33;
    coin *= 0xff51afd7ed558ccdULL;
    coin ^= coin >> 33;
    coin *= 0xc4ceb9fe1a85ec53ULL;
    coin ^= coin >> 33;
    return coin;
}
