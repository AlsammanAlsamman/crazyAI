#include <stdint.h>
#include <stddef.h>

#define ROTL64(x, b) (((x) << (b)) | ((x) >> (64 - (b))))

/* ONE TUMBLE down the mason trail: four stalk-strikes.  Each strike is a
 * sob (add mod 2^64 -- its carry is the eggshell the trail sheds, swept
 * away on purpose), a turn of the coil (rotation by that stalk's fixed
 * angle), and the hairline crack read off at that angle (xor).
 * There is no multiplication anywhere on this trail. */
#define TUMBLE                                                            \
    do {                                                                  \
        v0 += v1; v1 = ROTL64(v1, 13); v1 ^= v0; v0 = ROTL64(v0, 32);     \
        v2 += v3; v3 = ROTL64(v3, 16); v3 ^= v2;                          \
        v0 += v3; v3 = ROTL64(v3, 21); v3 ^= v0;                          \
        v2 += v1; v1 = ROTL64(v1, 17); v1 ^= v2; v2 = ROTL64(v2, 32);     \
    } while (0)

/* Press the bale into the sphere's face, let it tumble a FIXED one turn,
 * then let the old face shrink and go.  No footprint is kept. */
#define PRESS(mexpr)                                                      \
    do {                                                                  \
        const uint64_t mm = (mexpr);                                      \
        v3 ^= mm; TUMBLE; v0 ^= mm;                                       \
    } while (0)

/* Gather 1..7 loose marks into the final bale, low mark first. */
#define GATHER(q, n)                                                      \
    do {                                                                  \
        switch (n) {                                                      \
        case 7: b |= (uint64_t)(q)[6] << 48; /* fall through */            \
        case 6: b |= (uint64_t)(q)[5] << 40; /* fall through */            \
        case 5: b |= (uint64_t)(q)[4] << 32; /* fall through */            \
        case 4: b |= (uint64_t)(q)[3] << 24; /* fall through */            \
        case 3: b |= (uint64_t)(q)[2] << 16; /* fall through */            \
        case 2: b |= (uint64_t)(q)[1] <<  8; /* fall through */            \
        case 1: b |= (uint64_t)(q)[0];       /* fall through */            \
        default: break;                                                   \
        }                                                                 \
    } while (0)

static inline uint64_t load_bale(const unsigned char *p)
{
    uint64_t m;
    __builtin_memcpy(&m, p, sizeof m);        /* one unaligned mov on x86-64 */
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) &&            \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    m = __builtin_bswap64(m);                 /* the trail runs low-mark-first */
#endif
    return m;
}

uint64_t kernel(const unsigned char *restrict data, size_t len)
{
    /* The sphere at the trail's high mouth: one body, four faces. */
    uint64_t v0 = 0x736f6d6570736575ULL ^ 0x0706050403020100ULL;
    uint64_t v1 = 0x646f72616e646f6dULL ^ 0x0f0e0d0c0b0a0908ULL;
    uint64_t v2 = 0x6c7967656e657261ULL ^ 0x0706050403020100ULL;
    uint64_t v3 = 0x7465646279746573ULL ^ 0x0f0e0d0c0b0a0908ULL;

    /* The size of the pile is pressed into the top of the last bale, so a
     * pile's length cannot be swept away with its dust. */
    uint64_t b = (uint64_t)len << 56;

    if (len >= 8) {
        /* ---- REGIME A: a pile I cannot lift.  Bale the marks by eights. */
        const unsigned char *p = data;
        size_t bales = len >> 3;

        if (bales >= 4) {                 /* four bales per walk of the trail */
            size_t walks = bales >> 2;
            bales &= 3;
            do {
                PRESS(load_bale(p));
                PRESS(load_bale(p + 8));
                PRESS(load_bale(p + 16));
                PRESS(load_bale(p + 24));
                p += 32;
            } while (--walks);
        }
        while (bales--) { PRESS(load_bale(p)); p += 8; }

        GATHER(p, len & 7);
    } else {
        /* ---- REGIME B: it fits in one hand.  No bales, no loop, no stride:
         * straight to the final bale and the closing cracks. */
        GATHER(data, len);
    }

    /* the last bale takes its one fixed tumble, like every other */
    v3 ^= b; TUMBLE; v0 ^= b;

    /* I do not keep the sphere.  I take the last, smallest crack it leaves
     * in the final stalk -- three closing tumbles, then the four faces
     * folded into one token.  Everything else goes into the plumbing. */
    v2 ^= 0xffULL;
    TUMBLE; TUMBLE; TUMBLE;
    return v0 ^ v1 ^ v2 ^ v3;
}
